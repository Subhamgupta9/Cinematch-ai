"""
Hybrid Recommendation Engine for CineMatch AI
----------------------------------------------
Implements a production-grade, explainable hybrid recommendation engine uniting:
1. Dynamic Sparse TF-IDF Vector Projection across the 12,000-dimensional MovieLens feature space.
2. Item-Item Shrunk Pearson Collaborative Filtering (for MovieLens-linked items).
3. Source-Isolated Bayesian Quality Priors (MovieLens Bayesian mean vs TMDb damped vote score).
4. Relevance-First Gating & Minimum-Evidence Policy (IDF-based token discrimination).
5. Deterministic Multi-Tier Ranking Key with zero process-salted hash dependencies.
6. Unified Cross-Catalog Candidate Generation (MovieLens + TMDb Global Catalog).

Mathematical Fusion Specification (Stage 3 Architecture Section 6):
- Path A (Collaborative Available - both items in MovieLens):
    Sim_collab_eff = Sim_collab * min(1.0, Sim_content / 0.40)
    Score = (1 - beta) * [alpha * Sim_content + (1 - alpha) * Sim_collab_eff] + beta * Quality_ML
- Path B (Cold-Start - external or unrated candidates):
    Score = (1 - beta) * Sim_content + beta * Quality_source
- Default Weights: alpha = 0.65, beta = 0.10.
"""

import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Set
from scipy import sparse

from unified_data_model import UnifiedMovie
from recommender import MovieRecommender
from collaborative_recommender import CollaborativeRecommender
from candidate_generator import CandidateGenerator
from tmdb_client import get_tmdb_api_key
from tmdb_search import fetch_movie_details_global, search_movies_global


class HybridRecommender:
    """
    Hybrid Movie Recommendation Engine combining content similarity, collaborative filtering,
    and source-isolated quality scoring with relevance-first gating.
    """

    def __init__(
        self,
        data_dir: Optional[Path | str] = None,
        models_dir: Optional[Path | str] = None,
        alpha: float = 0.65,
        beta: float = 0.10,
        gamma_sim: float = 0.04,
    ):
        base_path = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir else (base_path / "data" / "ml-latest-small")
        self.models_dir = Path(models_dir) if models_dir else (base_path / "models")

        # Fusion parameters (initial experimental hypotheses)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma_sim = float(gamma_sim)

        # Core sub-modules
        self.recommender = MovieRecommender(data_dir=self.data_dir, models_dir=self.models_dir)
        self.collab_recommender = CollaborativeRecommender(data_dir=self.data_dir, models_dir=self.models_dir)
        self.candidate_generator = CandidateGenerator(data_dir=self.data_dir, models_dir=self.models_dir)

        # Token IDF cache for Minimum-Evidence Policy
        self.idf_map: Dict[str, float] = {}
        self.feature_names: List[str] = []
        self._is_initialized: bool = False

    def initialize(self) -> "HybridRecommender":
        """
        Loads precomputed models and prepares TF-IDF and collaborative spaces.
        """
        if self._is_initialized:
            return self

        # 1. Initialize content recommender
        if not self.recommender.load_artifacts():
            self.recommender.load_and_preprocess()

        # 2. Initialize collaborative recommender
        self.collab_recommender.load_or_fit()

        # 3. Cache feature names and IDF weights for Minimum-Evidence Policy
        if self.recommender.tfidf_vectorizer is not None:
            self.feature_names = list(self.recommender.tfidf_vectorizer.get_feature_names_out())
            idfs = self.recommender.tfidf_vectorizer.idf_
            self.idf_map = {term: float(idfs[idx]) for idx, term in enumerate(self.feature_names)}

        self._is_initialized = True
        return self

    def build_feature_soup(self, movie: UnifiedMovie) -> str:
        """
        Constructs metadata feature soup for any movie entity conforming strictly
        to the training feature soup schema in recommender.py:
        Clean Title + (Genres x 2) + IMAX + (Keywords/Tags x 2) + Decade
        """
        parts = []

        # 1. Clean title words
        clean_title = re.sub(r"\s*\(\d{4}\)\s*", "", movie.title).strip()
        for article in [", The", ", A", ", An"]:
            if clean_title.endswith(article):
                clean_title = article.replace(", ", "") + " " + clean_title[: -len(article)]
                break
        if clean_title:
            clean_title_words = re.findall(r"\b\w+\b", clean_title.lower())
            if clean_title_words:
                parts.append(" ".join(clean_title_words))

        # 2. Narrative genres (repeated 2x, excluding IMAX)
        narrative_genres = [
            g.replace("-", "").replace(" ", "").lower()
            for g in movie.genre_names
            if g.upper() != "IMAX"
        ]
        if narrative_genres:
            parts.append((" ".join(narrative_genres) + " ") * 2)

        # 3. Technical format IMAX (1x)
        if any(g.upper() == "IMAX" for g in movie.genre_names):
            parts.append("format_imax")

        # 4. Thematic tags / keywords (repeated 2x)
        tags_or_keywords = movie.movielens_tags or movie.keywords
        if tags_or_keywords:
            clean_tags = []
            for t in tags_or_keywords:
                clean_t = re.sub(r"[^\w\s-]", "", str(t).lower()).strip()
                if clean_t and clean_t not in ["in netflix queue", "none", "nan"]:
                    clean_tags.append(clean_t)
            if clean_tags:
                tokens = " ".join(clean_tags)
                parts.append((tokens + " ") * 2)

        # 5. Decade token
        year = movie.release_year
        if year and str(year).isdigit():
            decade = int(int(year) // 10 * 10)
            parts.append(f"decade_{decade}s")

        soup_str = " ".join(parts).strip().lower()
        if not soup_str:
            fallback_id = movie.movielens_movie_id or movie.tmdb_id or "unknown"
            soup_str = f"movie_id_{fallback_id}"
        return soup_str

    def project_to_vector(self, soup: str) -> sparse.csr_matrix:
        """
        Projects a metadata soup string into the fitted 12,000-dimensional TF-IDF space.
        Returns 1 x 12,000 sparse CSR matrix.
        """
        self.initialize()
        assert self.recommender.tfidf_vectorizer is not None
        return self.recommender.tfidf_vectorizer.transform([soup])

    def check_minimum_evidence(
        self,
        query_soup: str,
        cand_soup: str,
        similarity_score: float,
    ) -> Tuple[bool, str]:
        """
        Empirically Calibrated Minimum-Evidence Policy (Architecture Section 3.3):
        - Generic tokens: IDF < 4.0 (broad genres)
        - Discriminative tokens: IDF >= 4.0 (specific subgenres, themes, keywords, distinct title words)
        Requirement: Must share >= 1 discriminative token OR >= 2 generic tokens,
        AND similarity_score >= gamma_sim.
        """
        if similarity_score < self.gamma_sim:
            return False, f"Similarity {similarity_score:.4f} below threshold {self.gamma_sim:.4f}"

        q_tokens = set(re.findall(r"\b\w+\b", query_soup.lower()))
        c_tokens = set(re.findall(r"\b\w+\b", cand_soup.lower()))
        shared = q_tokens & c_tokens

        # Exclude synthetic tokens from structural evidence count
        filtered_shared = {
            t for t in shared
            if not t.startswith("decade_") and not t.startswith("movie_id_") and not t.startswith("format_")
        }

        if not filtered_shared:
            return False, "No non-synthetic tokens shared between query and candidate"

        discriminative_count = sum(1 for t in filtered_shared if self.idf_map.get(t, 0.0) >= 4.0)
        generic_count = sum(1 for t in filtered_shared if self.idf_map.get(t, 0.0) < 4.0)

        if discriminative_count >= 1 or generic_count >= 2:
            return True, "Passes minimum evidence policy"

        return False, f"Insufficient evidence: {discriminative_count} disc, {generic_count} gen tokens"

    def compute_quality_score(self, candidate: UnifiedMovie) -> float:
        """
        Computes source-isolated quality prior in [0.0, 1.0] (Architecture Section 6.3).
        MovieLens: Bayesian weighted mean (v / (v+5))*R + (5 / (v+5))*(3.44) normalized to /5.0.
        TMDb: (vote_average / 10.0) * (v / (v+500)) for v >= 20.
        """
        if candidate.movielens_movie_id and candidate.movielens_rating_count and candidate.movielens_rating:
            v = candidate.movielens_rating_count
            r = candidate.movielens_rating
            w = (v / (v + 5.0)) * r + (5.0 / (v + 5.0)) * 3.44
            return float(min(1.0, max(0.0, w / 5.0)))

        if candidate.tmdb_vote_count and candidate.tmdb_vote_average and candidate.tmdb_vote_count >= 20:
            v = candidate.tmdb_vote_count
            avg = candidate.tmdb_vote_average
            damping = v / (v + 500.0)
            return float(min(1.0, max(0.0, (avg / 10.0) * damping)))

        return 0.0

    def score_candidate(
        self,
        query_movie: UnifiedMovie,
        cand_movie: UnifiedMovie,
        query_vec: sparse.csr_matrix,
        cand_vec: sparse.csr_matrix,
        query_soup: str,
        cand_soup: str,
    ) -> Dict[str, Any]:
        """
        Scores a candidate against the query using parallel multi-signal fusion,
        adaptive routing, relevance-first gating, and dynamic collaborative shrinkage.
        """
        # 1. Content Cosine Similarity
        # Dot product of L2-normalized vectors
        sim_content = float(query_vec.dot(cand_vec.T).toarray()[0, 0])
        sim_content = max(0.0, min(1.0, sim_content))

        # Determine adaptive routing path
        is_collab_available = (
            query_movie.movielens_movie_id is not None
            and cand_movie.movielens_movie_id is not None
            and self.collab_recommender.is_collaborative_available(query_movie.movielens_movie_id)
            and self.collab_recommender.is_collaborative_available(cand_movie.movielens_movie_id)
        )
        routing_path = "path_a_collab" if is_collab_available else "path_b_cold_start"

        # 2. Content Relevance Gate
        passes_evidence, evidence_msg = self.check_minimum_evidence(query_soup, cand_soup, sim_content)
        if not passes_evidence:
            return {
                "eligible": False,
                "rejection_reason": evidence_msg,
                "ranking_score": 0.0,
                "similarity_score": sim_content,
                "collaborative_score": None,
                "quality_score": 0.0,
                "path": routing_path,
            }

        # 3. Source-Isolated Quality Score
        quality_score = self.compute_quality_score(cand_movie)

        # 4. Adaptive Routing Calculation

        collab_score: Optional[float] = None
        ranking_score: float = 0.0

        if is_collab_available:
            # Path A: Collaborative Available
            raw_collab = self.collab_recommender.get_pairwise_similarity(
                query_movie.movielens_movie_id, cand_movie.movielens_movie_id
            )
            collab_score = raw_collab

            # Dynamic shrinkage when content similarity is low (< 0.40)
            collab_attenuation = min(1.0, sim_content / 0.40) if sim_content < 0.40 else 1.0
            effective_collab = raw_collab * collab_attenuation

            ranking_score = (
                (1.0 - self.beta) * (self.alpha * sim_content + (1.0 - self.alpha) * effective_collab)
                + self.beta * quality_score
            )
        else:
            # Path B: Cold-Start / External Candidate
            ranking_score = (1.0 - self.beta) * sim_content + self.beta * quality_score

        # Sanitize NaN/Inf
        if math.isnan(ranking_score) or math.isinf(ranking_score):
            ranking_score = 0.0

        return {
            "eligible": True,
            "rejection_reason": "",
            "ranking_score": float(ranking_score),
            "similarity_score": sim_content,
            "collaborative_score": collab_score,
            "quality_score": quality_score,
            "path": "path_a_collab" if is_collab_available else "path_b_cold_start",
        }

    def _generate_explanation(
        self,
        query: UnifiedMovie,
        cand: UnifiedMovie,
        query_vec: sparse.csr_matrix,
        cand_vec: sparse.csr_matrix,
        scores: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Constructs verifiable feature intersections and domain explanation string.
        """
        # Shared genres
        q_genres = set(query.genre_names)
        c_genres = set(cand.genre_names)
        shared_genres = sorted(list(q_genres & c_genres))

        # Shared tags / keywords
        q_tags = set([t.lower() for t in (query.movielens_tags or query.keywords)])
        c_tags = set([t.lower() for t in (cand.movielens_tags or cand.keywords)])
        shared_tags = sorted(list(q_tags & c_tags))

        # Top contributing vector terms
        top_terms = []
        if self.feature_names:
            v_q = query_vec.toarray()[0]
            v_c = cand_vec.toarray()[0]
            prod = v_q * v_c
            top_indices = np.argsort(prod)[::-1][:6]
            for idx in top_indices:
                if prod[idx] > 0:
                    term = self.feature_names[idx]
                    if not term.startswith("decade_") and not term.startswith("movie_id_") and not term.startswith("format_"):
                        top_terms.append(term)

        # Build human-readable explanation
        reasons = []
        if shared_genres:
            reasons.append(f"Shares genre(s): {', '.join(shared_genres)}")
        if shared_tags:
            reasons.append(f"Matching themes: {', '.join(shared_tags)}")
        elif top_terms:
            reasons.append(f"Top matching terms: {', '.join(top_terms[:3])}")

        if scores.get("collaborative_score") is not None and scores["collaborative_score"] > 0.3:
            reasons.append(f"Behavioral affinity: Frequently co-rated by viewers of {query.title}")

        if query.release_year and cand.release_year:
            if abs(query.release_year - cand.release_year) <= 3:
                reasons.append(f"From the same era ({cand.release_year})")

        explanation_text = " | ".join(reasons) if reasons else "High overall thematic alignment"

        collab_display = (
            f"{scores['collaborative_score']:.4f}"
            if scores.get("collaborative_score") is not None
            else "N/A (Not Available - Content-Only Model)"
        )

        return {
            "shared_genres": shared_genres,
            "shared_tags": shared_tags,
            "top_terms": top_terms,
            "explanation": explanation_text,
            "collaborative_display": collab_display,
        }

    def recommend(
        self,
        query: Union[str, int, UnifiedMovie],
        top_n: int = 10,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates top-N hybrid movie recommendations for any query entity.
        Supports query by title string, MovieLens movieId, or UnifiedMovie instance.
        """
        self.initialize()
        resolved_api_key = api_key or get_tmdb_api_key()

        # 1. Resolve query into UnifiedMovie
        query_movie: Optional[UnifiedMovie] = None

        if isinstance(query, UnifiedMovie):
            query_movie = query
        elif isinstance(query, int):
            # MovieLens movieId lookup
            df = self.candidate_generator._get_ml_preprocessed()
            if df is not None:
                matches = df[df["movieId"] == query]
                if not matches.empty:
                    query_movie = UnifiedMovie.from_movielens_row(matches.iloc[0].to_dict())
        elif isinstance(query, str):
            clean_str = query.strip()
            stripped_title = re.sub(r"\s*\(\d{4}\)\s*", "", clean_str).strip()
            # Try finding in local MovieLens preprocessed catalog first
            df = self.candidate_generator._get_ml_preprocessed()
            if df is not None:
                # 1. Exact or case-insensitive match on title, clean_title, or stripped_title
                matches = df[
                    (df["title"].str.lower() == clean_str.lower())
                    | (df["clean_title"].str.lower() == clean_str.lower())
                    | (df["clean_title"].str.lower() == stripped_title.lower())
                ]
                if not matches.empty:
                    query_movie = UnifiedMovie.from_movielens_row(matches.iloc[0].to_dict())
                else:
                    # 2. Title containing search
                    part_matches = df[
                        df["title"].str.lower().str.contains(re.escape(stripped_title.lower()))
                        | df["clean_title"].str.lower().str.contains(re.escape(stripped_title.lower()))
                    ]
                    if not part_matches.empty:
                        query_movie = UnifiedMovie.from_movielens_row(part_matches.iloc[0].to_dict())

            # If not in local catalog, query TMDb global search if key available
            if query_movie is None and resolved_api_key:
                search_res = search_movies_global(clean_str, api_key=resolved_api_key, timeout=3.0)
                if search_res.get("success") and search_res.get("results"):
                    first_hit = search_res["results"][0]
                    # Enrich with full details & keywords
                    details = fetch_movie_details_global(first_hit["tmdb_id"], api_key=resolved_api_key, timeout=3.0)
                    if details:
                        query_movie = UnifiedMovie.from_tmdb_dict(details)
                    else:
                        query_movie = UnifiedMovie.from_tmdb_dict(first_hit)

        if query_movie is None:
            return {
                "success": False,
                "status": "not_found",
                "query": str(query),
                "query_movie": None,
                "recommendations": [],
                "candidate_pool_size": 0,
                "message": f"Could not find movie '{query}' in MovieLens or TMDb catalog.",
            }

        # Enrich query identity
        query_movie = self.candidate_generator.enrich_entity_identity(query_movie)
        query_cid = self.candidate_generator.resolve_canonical_id(query_movie)

        # 2. Build query feature soup and projection vector
        query_soup = self.build_feature_soup(query_movie)
        query_vec = self.project_to_vector(query_soup)

        # 3. Retrieve Candidates (capped at <= 100)
        candidates = self.candidate_generator.retrieve_candidates(
            query_movie=query_movie,
            recommender=self.recommender,
            collab_recommender=self.collab_recommender,
            api_key=resolved_api_key,
        )

        # If zero candidates retrieved (e.g. completely isolated), fallback to local catalog mode
        if not candidates and self.recommender.movies_df is not None:
            local_recs = self.recommender.recommend_tmdb(query_movie, top_n=top_n)
            query_dict = query_movie.to_dict()
            query_dict["canonical_id"] = query_cid
            query_dict["poster_url"] = query_movie.get_poster_url()
            return {
                "success": True,
                "status": "local_fallback",
                "query": query_movie.title,
                "query_movie": query_dict,
                "recommendations": local_recs.get("recommendations", []),
                "candidate_pool_size": len(local_recs.get("recommendations", [])),
                "message": "Generated recommendations via Local Catalog Mode.",
            }

        # 4. Score all candidates in pool
        scored_candidates = []
        for cand in candidates:
            cand_cid = self.candidate_generator.resolve_canonical_id(cand)
            if cand_cid == query_cid:
                continue  # Enforce self-exclusion invariant

            cand_soup = self.build_feature_soup(cand)
            cand_vec = self.project_to_vector(cand_soup)

            score_info = self.score_candidate(
                query_movie=query_movie,
                cand_movie=cand,
                query_vec=query_vec,
                cand_vec=cand_vec,
                query_soup=query_soup,
                cand_soup=cand_soup,
            )

            if score_info["eligible"]:
                explanation = self._generate_explanation(
                    query=query_movie,
                    cand=cand,
                    query_vec=query_vec,
                    cand_vec=cand_vec,
                    scores=score_info,
                )

                vote_count = cand.movielens_rating_count or cand.tmdb_vote_count or 0
                avg_rating = cand.movielens_rating or cand.tmdb_vote_average or 0.0

                scored_candidates.append({
                    "canonical_id": cand_cid,
                    "title": cand.title,
                    "original_title": cand.original_title,
                    "year": cand.release_year,
                    "movieId": cand.movielens_movie_id,
                    "tmdbId": cand.tmdb_id,
                    "imdbId": cand.imdb_id,
                    "language": cand.original_language or "en",
                    "origin_country": cand.origin_country,
                    "metadata_source": cand.metadata_source,
                    "ranking_score": score_info["ranking_score"],
                    "similarity_score": score_info["similarity_score"],
                    "collaborative_score": explanation["collaborative_display"],
                    "quality_score": score_info["quality_score"],
                    "vote_count": int(vote_count),
                    "avg_rating": float(avg_rating),
                    "shared_genres": explanation["shared_genres"],
                    "shared_tags": explanation["shared_tags"],
                    "top_terms": explanation["top_terms"],
                    "explanation": explanation["explanation"],
                    "poster_path": cand.poster_path,
                    "poster_url": cand.get_poster_url(),
                    "entity": cand,
                })

        # 5. Deterministic Descending Sort Key (Architecture Section 7.1)
        def ranking_sort_key(c: Dict[str, Any]) -> Tuple[float, float, float, int, str]:
            raw_score = c["ranking_score"]
            safe_score = -1.0 if (math.isnan(raw_score) or math.isinf(raw_score)) else float(raw_score)
            raw_sim = c["similarity_score"]
            safe_sim = -1.0 if math.isnan(raw_sim) else float(raw_sim)
            safe_qual = float(c.get("quality_score", 0.0))
            safe_votes = int(c.get("vote_count", 0))
            canonical_id = str(c["canonical_id"])
            return (safe_score, safe_sim, safe_qual, safe_votes, canonical_id)

        scored_candidates.sort(key=ranking_sort_key, reverse=True)

        # 6. Assign final 1-based ranks to top_n
        final_top = scored_candidates[:top_n]
        for rank_idx, item in enumerate(final_top, start=1):
            item["rank"] = rank_idx
            # Round scores for clean presentation
            item["recommendation_score"] = round(item["ranking_score"], 4)
            item["content_similarity_score"] = round(item["similarity_score"], 4)
            item.pop("entity", None)  # Clean entity object before serialization

        # Construct query movie dict
        query_dict = query_movie.to_dict()
        query_dict["canonical_id"] = query_cid
        query_dict["poster_url"] = query_movie.get_poster_url()

        return {
            "success": True,
            "status": "success",
            "query": query_movie.title,
            "query_movie": query_dict,
            "recommendations": final_top,
            "candidate_pool_size": len(candidates),
            "message": f"Successfully generated {len(final_top)} hybrid recommendations for '{query_movie.title}'.",
        }
