"""
Candidate Pool Generator for CineMatch AI
-----------------------------------------
Implements multi-channel candidate retrieval, canonical cross-catalog identity resolution,
strict 4.5s shared network deadline budget (Policy A), and deterministic 100-candidate cap allocation.

Channel Extraction Quotas (Architecture Section 10.1):
1. Channel 1 (ML Content): Up to 40 candidates from local MovieLens vector space.
2. Channel 2 (ML Collaborative): Up to 20 candidates from item-item co-rating graph.
3. Channel 3 (TMDb Recommendations): Up to 25 candidates via /movie/{id}/recommendations.
4. Channel 4 (TMDb Similar): Up to 25 candidates via /movie/{id}/similar.
5. Channel 5 (TMDb Regional Discover): Up to 20 candidates via /discover/movie.

Deterministic 100-Candidate Cap Allocation Algorithm (Architecture Section 10.2):
- Phase A: Multi-source intersections (candidates retrieved by >= 2 independent channels).
- Phase B: Source-diversity floor reservations (within remaining capacity).
- Phase C: Residual allocation by channel score and canonical_id tie-breaker.
- Invariant Guarantee: len(final_pool) <= 100 always.
"""

import time
import math
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple

from unified_data_model import UnifiedMovie
from tmdb_client import get_tmdb_api_key, TMDB_API_BASE
from tmdb_search import normalize_tmdb_movie

CHANNEL_QUOTAS = {
    "ml_content": 40,
    "ml_collab": 20,
    "tmdb_recs": 25,
    "tmdb_sim": 25,
    "tmdb_discover": 20,
}

CHANNEL_FLOORS = {
    "ml_content": 20,
    "ml_collab": 10,
    "tmdb_recs": 15,
    "tmdb_sim": 10,
    "tmdb_discover": 10,
}

MAX_TOTAL_CANDIDATES = 100
NETWORK_BUDGET_SECONDS = 4.5
INITIAL_SOCKET_TIMEOUT = 3.0
MAX_RETRY_TIMEOUT = 1.25


class CandidateGenerator:
    """
    Retrieves, resolves, and deduplicates movie candidates across MovieLens and TMDb.
    """

    def __init__(
        self,
        data_dir: Optional[Path | str] = None,
        models_dir: Optional[Path | str] = None,
    ):
        base_path = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir else (base_path / "data" / "ml-latest-small")
        self.models_dir = Path(models_dir) if models_dir else (base_path / "models")

        # Identifier resolution maps loaded from links.csv
        self.ml_to_tmdb: Dict[int, str] = {}
        self.tmdb_to_ml: Dict[str, int] = {}
        self.ml_to_imdb: Dict[int, str] = {}
        self.imdb_to_ml: Dict[str, int] = {}
        self.imdb_to_tmdb: Dict[str, str] = {}

        # Cached MovieLens preprocessed lookup dataframe
        self.ml_preprocessed_df: Optional[pd.DataFrame] = None
        self._load_identifier_maps()

    def _load_identifier_maps(self) -> None:
        """Loads links.csv to construct canonical identifier translation graphs."""
        links_path = self.data_dir / "links.csv"
        if not links_path.exists():
            return

        links_df = pd.read_csv(links_path)
        for _, row in links_df.iterrows():
            ml_id = int(row["movieId"])
            
            # TMDb ID mapping
            if pd.notna(row.get("tmdbId")):
                try:
                    tmdb_id = str(int(row["tmdbId"]))
                    self.ml_to_tmdb[ml_id] = tmdb_id
                    self.tmdb_to_ml[tmdb_id] = ml_id
                except (ValueError, TypeError):
                    pass

            # IMDb ID mapping (formatted as ttXXXXXXX)
            if pd.notna(row.get("imdbId")):
                try:
                    raw_imdb = str(int(row["imdbId"]))
                    imdb_formatted = f"tt{raw_imdb.zfill(7)}"
                    self.ml_to_imdb[ml_id] = imdb_formatted
                    self.imdb_to_ml[imdb_formatted] = ml_id
                    if ml_id in self.ml_to_tmdb:
                        self.imdb_to_tmdb[imdb_formatted] = self.ml_to_tmdb[ml_id]
                except (ValueError, TypeError):
                    pass

    def _get_ml_preprocessed(self) -> Optional[pd.DataFrame]:
        """Lazy loads the preprocessed MovieLens catalog dataframe."""
        if self.ml_preprocessed_df is None:
            pkl_path = self.models_dir / "movies_preprocessed.pkl"
            if pkl_path.exists():
                try:
                    self.ml_preprocessed_df = pd.read_pickle(pkl_path)
                except Exception:
                    pass
        return self.ml_preprocessed_df

    def resolve_canonical_id(self, movie: UnifiedMovie) -> str:
        """
        Assigns immutable canonical_id according to the specification hierarchy:
        1. Primary: tmdb:{id}
        2. Secondary: imdb:{id}
        3. Tertiary: ml:{id}
        """
        # 1. Direct TMDb ID
        if movie.tmdb_id:
            return f"tmdb:{movie.tmdb_id}"

        # 2. Check if MovieLens ID resolves to TMDb ID
        if movie.movielens_movie_id and movie.movielens_movie_id in self.ml_to_tmdb:
            resolved_tmdb = self.ml_to_tmdb[movie.movielens_movie_id]
            object.__setattr__(movie, "tmdb_id", resolved_tmdb)
            return f"tmdb:{resolved_tmdb}"

        # 3. Check if IMDb ID resolves to TMDb ID
        if movie.imdb_id and movie.imdb_id in self.imdb_to_tmdb:
            resolved_tmdb = self.imdb_to_tmdb[movie.imdb_id]
            object.__setattr__(movie, "tmdb_id", resolved_tmdb)
            return f"tmdb:{resolved_tmdb}"

        # 4. IMDb standard
        if movie.imdb_id:
            return f"imdb:{movie.imdb_id}"

        # 5. MovieLens standard
        if movie.movielens_movie_id:
            return f"ml:{movie.movielens_movie_id}"

        return movie.primary_uid

    def enrich_entity_identity(self, movie: UnifiedMovie) -> UnifiedMovie:
        """
        Bridges missing identifiers across MovieLens, TMDb, and IMDb registries.
        Preserves original records while linking cross-catalog IDs.
        """
        # Resolve MovieLens ID if missing
        if movie.movielens_movie_id is None:
            if movie.tmdb_id and movie.tmdb_id in self.tmdb_to_ml:
                object.__setattr__(movie, "movielens_movie_id", self.tmdb_to_ml[movie.tmdb_id])
            elif movie.imdb_id and movie.imdb_id in self.imdb_to_ml:
                object.__setattr__(movie, "movielens_movie_id", self.imdb_to_ml[movie.imdb_id])

        # Resolve TMDb ID if missing
        if movie.tmdb_id is None:
            if movie.movielens_movie_id and movie.movielens_movie_id in self.ml_to_tmdb:
                object.__setattr__(movie, "tmdb_id", self.ml_to_tmdb[movie.movielens_movie_id])
            elif movie.imdb_id and movie.imdb_id in self.imdb_to_tmdb:
                object.__setattr__(movie, "tmdb_id", self.imdb_to_tmdb[movie.imdb_id])

        # Resolve IMDb ID if missing
        if not movie.imdb_id and movie.movielens_movie_id and movie.movielens_movie_id in self.ml_to_imdb:
            object.__setattr__(movie, "imdb_id", self.ml_to_imdb[movie.movielens_movie_id])

        # If entity has MovieLens ID, enrich with MovieLens ratings/tags from preprocessed catalog
        if movie.movielens_movie_id:
            object.__setattr__(movie, "metadata_source", "movielens_linked" if movie.tmdb_id else "movielens_only")
            df = self._get_ml_preprocessed()
            if df is not None:
                matches = df[df["movieId"] == movie.movielens_movie_id]
                if not matches.empty:
                    row = matches.iloc[0]
                    if movie.movielens_rating is None and row.get("vote_count", 0) > 0:
                        object.__setattr__(movie, "movielens_rating", float(row["avg_rating"]))
                        object.__setattr__(movie, "movielens_rating_count", int(row["vote_count"]))
                        object.__setattr__(movie, "collaborative_data_available", True)
                    if not movie.movielens_tags and isinstance(row.get("tag_list"), list):
                        object.__setattr__(movie, "movielens_tags", list(row["tag_list"]))

        return movie

    def merge_entities(self, primary: UnifiedMovie, secondary: UnifiedMovie) -> UnifiedMovie:
        """
        Deterministically fuses two UnifiedMovie records representing the same canonical entity.
        Strictly preserves MovieLens ratings, links TMDb media, and sets metadata_source.
        """
        # Determine base entity (prefer one that already has MovieLens links)
        base = primary if primary.movielens_movie_id else secondary
        other = secondary if primary.movielens_movie_id else primary

        ml_id = base.movielens_movie_id or other.movielens_movie_id
        tmdb_id = base.tmdb_id or other.tmdb_id
        imdb_id = base.imdb_id or other.imdb_id

        # Combine genres and keywords without duplicates
        all_genres = list(dict.fromkeys(base.genre_names + other.genre_names))
        all_keywords = list(dict.fromkeys(base.keywords + other.keywords))
        all_tags = list(dict.fromkeys(base.movielens_tags + other.movielens_tags))

        # Media paths
        poster = base.poster_path or other.poster_path
        backdrop = base.backdrop_path or other.backdrop_path

        # Ratings - strictly preserve source isolation
        tmdb_avg = base.tmdb_vote_average if base.tmdb_vote_average is not None else other.tmdb_vote_average
        tmdb_cnt = base.tmdb_vote_count if base.tmdb_vote_count is not None else other.tmdb_vote_count
        ml_avg = base.movielens_rating if base.movielens_rating is not None else other.movielens_rating
        ml_cnt = base.movielens_rating_count if base.movielens_rating_count is not None else other.movielens_rating_count

        source = "movielens_linked" if (ml_id and tmdb_id) else (base.metadata_source or other.metadata_source)

        merged = UnifiedMovie(
            tmdb_id=tmdb_id,
            movielens_movie_id=ml_id,
            imdb_id=imdb_id,
            title=base.title or other.title,
            original_title=base.original_title or other.original_title,
            overview=base.overview or other.overview,
            release_date=base.release_date or other.release_date,
            release_year=base.release_year or other.release_year,
            original_language=base.original_language or other.original_language,
            origin_country=base.origin_country or other.origin_country,
            genre_ids=base.genre_ids or other.genre_ids,
            genre_names=all_genres,
            keywords=all_keywords,
            poster_path=poster,
            backdrop_path=backdrop,
            tmdb_vote_average=tmdb_avg,
            tmdb_vote_count=tmdb_cnt,
            movielens_rating=ml_avg,
            movielens_rating_count=ml_cnt,
            movielens_tags=all_tags,
            metadata_source=source,
            collaborative_data_available=bool(ml_cnt and ml_cnt > 0),
        )
        return merged

    def _fetch_tmdb_channel(
        self,
        endpoint: str,
        params: Dict[str, Any],
        deadline: float,
        api_key: str,
    ) -> List[Dict[str, Any]]:
        """
        Executes a TMDb API request with the strict Policy A deadline and retry model.
        Returns empty list on 429, timeout, or deadline expiration.
        """
        t_remaining = deadline - time.time()
        if t_remaining <= 0.1:
            return []

        full_params = dict(params)
        full_params["api_key"] = api_key

        socket_timeout = min(INITIAL_SOCKET_TIMEOUT, t_remaining)

        try:
            resp = requests.get(endpoint, params=full_params, timeout=socket_timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("results", []) if isinstance(data, dict) else []
            elif resp.status_code == 429:
                # Do not retry on 429
                return []
            elif resp.status_code in [502, 503, 504]:
                # Retry once if time permits
                t_remaining = deadline - time.time()
                if t_remaining > 1.0:
                    time.sleep(0.25)
                    retry_timeout = min(MAX_RETRY_TIMEOUT, deadline - time.time())
                    if retry_timeout > 0.2:
                        retry_resp = requests.get(endpoint, params=full_params, timeout=retry_timeout)
                        if retry_resp.status_code == 200:
                            data = retry_resp.json()
                            return data.get("results", []) if isinstance(data, dict) else []
            return []
        except (requests.Timeout, requests.ConnectionError):
            # Try once more if sufficient budget remains
            t_remaining = deadline - time.time()
            if t_remaining > 1.0:
                try:
                    time.sleep(0.25)
                    retry_timeout = min(MAX_RETRY_TIMEOUT, deadline - time.time())
                    if retry_timeout > 0.2:
                        retry_resp = requests.get(endpoint, params=full_params, timeout=retry_timeout)
                        if retry_resp.status_code == 200:
                            data = retry_resp.json()
                            return data.get("results", []) if isinstance(data, dict) else []
                except Exception:
                    pass
            return []
        except Exception:
            return []

    def retrieve_candidates(
        self,
        query_movie: UnifiedMovie,
        recommender: Any,
        collab_recommender: Optional[Any] = None,
        api_key: Optional[str] = None,
    ) -> List[UnifiedMovie]:
        """
        Retrieves candidates across all 5 channels within the 4.5s deadline budget,
        resolves identities, and runs the deterministic 100-candidate allocation algorithm.
        """
        start_time = time.time()
        deadline = start_time + NETWORK_BUDGET_SECONDS

        resolved_api_key = api_key if api_key is not None else get_tmdb_api_key()
        raw_channels: Dict[str, List[UnifiedMovie]] = {
            "ml_content": [],
            "ml_collab": [],
            "tmdb_recs": [],
            "tmdb_sim": [],
            "tmdb_discover": [],
        }

        # Ensure query identity is enriched
        query_movie = self.enrich_entity_identity(query_movie)
        query_cid = self.resolve_canonical_id(query_movie)

        # -------------------------------------------------------------
        # Channel 1: MovieLens Content Vector Search (Quota: 40)
        # -------------------------------------------------------------
        try:
            if query_movie.movielens_movie_id:
                # Query has a known MovieLens title
                title_query = query_movie.title
                ml_res = recommender.recommend(title_query, top_n=CHANNEL_QUOTAS["ml_content"])
                ml_recs = ml_res if isinstance(ml_res, list) else ml_res.get("recommendations", [])
            else:
                # Query is TMDb-only; project dynamically into MovieLens vector space
                ml_res = recommender.recommend_tmdb(query_movie, top_n=CHANNEL_QUOTAS["ml_content"])
                ml_recs = ml_res.get("recommendations", []) if isinstance(ml_res, dict) else ml_res

            df_preprocessed = self._get_ml_preprocessed()
            if df_preprocessed is not None and ml_recs:
                for rank, rec in enumerate(ml_recs):
                    m_id = rec.get("movieId")
                    if m_id:
                        m_rows = df_preprocessed[df_preprocessed["movieId"] == m_id]
                        if not m_rows.empty:
                            row_dict = m_rows.iloc[0].to_dict()
                            cand = UnifiedMovie.from_movielens_row(row_dict)
                            cand = self.enrich_entity_identity(cand)
                            # Set channel score for deterministic ranking
                            setattr(cand, "channel_score", float(rec.get("similarity_score", 0.0)))
                            if self.resolve_canonical_id(cand) != query_cid:
                                raw_channels["ml_content"].append(cand)
        except Exception:
            pass

        # -------------------------------------------------------------
        # Channel 2: MovieLens Collaborative Neighbors (Quota: 20)
        # -------------------------------------------------------------
        if collab_recommender and query_movie.movielens_movie_id:
            try:
                neighbors = collab_recommender.get_top_collaborative_neighbors(
                    query_movie.movielens_movie_id, top_n=CHANNEL_QUOTAS["ml_collab"]
                )
                df_preprocessed = self._get_ml_preprocessed()
                if df_preprocessed is not None:
                    for n_mid, sim in neighbors:
                        m_rows = df_preprocessed[df_preprocessed["movieId"] == n_mid]
                        if not m_rows.empty:
                            row_dict = m_rows.iloc[0].to_dict()
                            cand = UnifiedMovie.from_movielens_row(row_dict)
                            cand = self.enrich_entity_identity(cand)
                            setattr(cand, "channel_score", float(sim))
                            if self.resolve_canonical_id(cand) != query_cid:
                                raw_channels["ml_collab"].append(cand)
            except Exception:
                pass

        # -------------------------------------------------------------
        # Channels 3, 4, 5: TMDb Live Network Channels (If API Key Available)
        # -------------------------------------------------------------
        if resolved_api_key and query_movie.tmdb_id:
            tmdb_id = str(query_movie.tmdb_id)

            # Channel 3: TMDb Recommendations (Quota: 25)
            recs_endpoint = f"{TMDB_API_BASE}/movie/{tmdb_id}/recommendations"
            recs_raw = self._fetch_tmdb_channel(recs_endpoint, {}, deadline, resolved_api_key)
            for raw_item in recs_raw[: CHANNEL_QUOTAS["tmdb_recs"]]:
                if isinstance(raw_item, dict):
                    norm = normalize_tmdb_movie(raw_item)
                    cand = UnifiedMovie.from_tmdb_dict(norm)
                    cand = self.enrich_entity_identity(cand)
                    # Use normalized popularity / vote average as channel proxy score
                    vote_score = float(cand.tmdb_vote_average or 0.0) / 10.0
                    setattr(cand, "channel_score", vote_score)
                    if self.resolve_canonical_id(cand) != query_cid:
                        raw_channels["tmdb_recs"].append(cand)

            # Channel 4: TMDb Similar (Quota: 25)
            # Fetch similar only if deadline budget permits (> 1.0s remaining)
            if (deadline - time.time()) > 1.0:
                sim_endpoint = f"{TMDB_API_BASE}/movie/{tmdb_id}/similar"
                sim_raw = self._fetch_tmdb_channel(sim_endpoint, {}, deadline, resolved_api_key)
                for raw_item in sim_raw[: CHANNEL_QUOTAS["tmdb_sim"]]:
                    if isinstance(raw_item, dict):
                        norm = normalize_tmdb_movie(raw_item)
                        cand = UnifiedMovie.from_tmdb_dict(norm)
                        cand = self.enrich_entity_identity(cand)
                        vote_score = float(cand.tmdb_vote_average or 0.0) / 10.0
                        setattr(cand, "channel_score", vote_score)
                        if self.resolve_canonical_id(cand) != query_cid:
                            raw_channels["tmdb_sim"].append(cand)

            # Channel 5: TMDb Regional Discover (Quota: 20)
            if (deadline - time.time()) > 0.8:
                discover_endpoint = f"{TMDB_API_BASE}/discover/movie"
                disc_params: Dict[str, Any] = {
                    "sort_by": "popularity.desc",
                    "include_adult": False,
                }
                # Filter by origin country (e.g. IN) or language
                INDIAN_LANGS = {"hi", "te", "ta", "ml", "kn", "bn", "mr", "pa"}
                is_indian = (
                    "IN" in query_movie.origin_country
                    or (bool(query_movie.original_language) and query_movie.original_language.lower() in INDIAN_LANGS)
                )
                if is_indian:
                    disc_params["with_origin_country"] = "IN"
                    if query_movie.original_language:
                        disc_params["with_original_language"] = query_movie.original_language
                    disc_params["vote_count.gte"] = 25
                    disc_params["sort_by"] = "vote_average.desc"
                elif query_movie.origin_country:
                    disc_params["with_origin_country"] = query_movie.origin_country[0]
                    disc_params["vote_count.gte"] = 50
                    disc_params["sort_by"] = "vote_average.desc"
                elif query_movie.original_language:
                    disc_params["with_original_language"] = query_movie.original_language
                    disc_params["vote_count.gte"] = 50
                    disc_params["sort_by"] = "vote_average.desc"

                # Include primary genres if available
                if query_movie.genre_ids:
                    disc_params["with_genres"] = ",".join(str(g) for g in query_movie.genre_ids[:2])

                disc_raw = self._fetch_tmdb_channel(discover_endpoint, disc_params, deadline, resolved_api_key)
                for raw_item in disc_raw[: CHANNEL_QUOTAS["tmdb_discover"]]:
                    if isinstance(raw_item, dict):
                        norm = normalize_tmdb_movie(raw_item)
                        cand = UnifiedMovie.from_tmdb_dict(norm)
                        cand = self.enrich_entity_identity(cand)
                        vote_score = float(cand.tmdb_vote_average or 0.0) / 10.0
                        setattr(cand, "channel_score", vote_score)
                        if self.resolve_canonical_id(cand) != query_cid:
                            raw_channels["tmdb_discover"].append(cand)

        # Apply deterministic 100-candidate global cap allocation
        return self.select_top_100_candidates(raw_channels, query_canonical_id=query_cid)

    def select_top_100_candidates(
        self,
        raw_channel_results: Dict[str, List[UnifiedMovie]],
        query_canonical_id: Optional[str] = None,
    ) -> List[UnifiedMovie]:
        """
        Deterministic Global 100-Candidate Allocation Algorithm.
        Guarantees:
          1. len(final_pool) <= 100 always.
          2. Multi-source intersections prioritized within the 100 budget.
          3. Source-diversity floor reservations applied within remaining capacity.
          4. Residual slots filled deterministically by best channel score and canonical_id tie-breaker.
          5. Self-exclusion of query_canonical_id.
        """
        merged_candidates: Dict[str, UnifiedMovie] = {}
        source_attribution: Dict[str, Set[str]] = {}
        source_best_ranks: Dict[str, Dict[str, int]] = {}
        source_best_scores: Dict[str, float] = {}

        for channel_name, candidates in raw_channel_results.items():
            for rank, cand in enumerate(candidates):
                cid = self.resolve_canonical_id(cand)
                if query_canonical_id and cid == query_canonical_id:
                    continue  # Self-exclusion invariant

                cand_score = getattr(cand, "channel_score", 0.0)

                if cid not in merged_candidates:
                    merged_candidates[cid] = cand
                    source_attribution[cid] = set()
                    source_best_ranks[cid] = {}
                    source_best_scores[cid] = cand_score
                else:
                    merged_candidates[cid] = self.merge_entities(merged_candidates[cid], cand)
                    source_best_scores[cid] = max(source_best_scores[cid], cand_score)

                source_attribution[cid].add(channel_name)
                source_best_ranks[cid][channel_name] = rank

        total_unique = len(merged_candidates)
        if total_unique <= MAX_TOTAL_CANDIDATES:
            # All unique candidates fit within the 100-slot cap
            return list(merged_candidates.values())

        # Step 2: Budget Constrained Selection (Budget = 100)
        selected_cids: Set[str] = set()
        BUDGET = MAX_TOTAL_CANDIDATES

        # Phase A: Multi-Source Intersections (Retrieved by >= 2 independent channels)
        multi_source_cids = [cid for cid, sources in source_attribution.items() if len(sources) >= 2]
        # Sort deterministically: num_sources desc, best_score desc, canonical_id asc
        multi_source_cids.sort(
            key=lambda cid: (-len(source_attribution[cid]), -source_best_scores[cid], cid)
        )
        for cid in multi_source_cids:
            if len(selected_cids) < BUDGET:
                selected_cids.add(cid)

        # Phase B: Source-Diversity Floor Reservations (Applied within remaining capacity)
        for channel_name, floor in CHANNEL_FLOORS.items():
            if len(selected_cids) >= BUDGET:
                break
            already_selected = sum(1 for cid in selected_cids if channel_name in source_attribution[cid])
            needed = max(0, floor - already_selected)
            if needed > 0:
                channel_cands = [
                    cid
                    for cid, sources in source_attribution.items()
                    if channel_name in sources and cid not in selected_cids
                ]
                channel_cands.sort(
                    key=lambda cid: (
                        source_best_ranks[cid].get(channel_name, 999),
                        -source_best_scores[cid],
                        cid,
                    )
                )
                for cid in channel_cands[:needed]:
                    if len(selected_cids) < BUDGET:
                        selected_cids.add(cid)

        # Phase C: Residual Allocation (Fill remaining slots up to 100)
        if len(selected_cids) < BUDGET:
            remaining_cids = [cid for cid in merged_candidates if cid not in selected_cids]
            remaining_cids.sort(
                key=lambda cid: (-source_best_scores[cid], -len(source_attribution[cid]), cid)
            )
            for cid in remaining_cids:
                if len(selected_cids) < BUDGET:
                    selected_cids.add(cid)
                else:
                    break

        assert len(selected_cids) <= MAX_TOTAL_CANDIDATES, f"Candidate pool exceeded cap: {len(selected_cids)}"
        return [merged_candidates[cid] for cid in selected_cids]
