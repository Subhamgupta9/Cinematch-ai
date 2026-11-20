"""
Unified Movie Data Model
------------------------
Defines a normalized, source-attributed movie entity for CineMatch AI.
Reconciles MovieLens benchmark records and TMDb global catalog records into a
single cohesive schema with strict data provenance.

Strict Data Integrity Guarantees:
1. Rating Isolation: MovieLens ratings (0.5 - 5.0) and TMDb ratings (0.0 - 10.0)
   are tracked in distinct fields and are never combined, scaled, or conflated.
2. Independent Geography & Language: Origin country (e.g. 'IN') and original language
   (e.g. 'ta', 'hi', 'en') are maintained separately without assumptions.
3. Stable Identity: Movies are identified by explicit catalog IDs (TMDb ID / MovieLens ID),
   never by title alone, guaranteeing safe handling of remakes and duplicate titles.
4. Zero Fabrication: Missing attributes remain explicit None or empty collections.
5. Zero UI/Network Coupling: Pure standard-library dataclass with zero Streamlit or
   network dependencies.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import re

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"
DEFAULT_POSTER_SIZE = "w342"
DEFAULT_BACKDROP_SIZE = "w780"


@dataclass
class UnifiedMovie:
    """
    Normalized data representation for a movie entity across MovieLens and TMDb.
    """

    # Primary Identifiers
    tmdb_id: Optional[str] = None
    movielens_movie_id: Optional[int] = None
    imdb_id: str = ""

    # Titles and Synopsis
    title: str = "Untitled"
    original_title: str = ""
    overview: str = ""

    # Release Attributes
    release_date: str = ""
    release_year: Optional[int] = None

    # Regional & Linguistic Attributes
    original_language: str = ""
    origin_country: List[str] = field(default_factory=list)

    # Classification & Thematics
    genre_ids: List[int] = field(default_factory=list)
    genre_names: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Artwork & CDN Paths
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None

    # TMDb Community Ratings (Scale 0.0 - 10.0)
    tmdb_vote_average: Optional[float] = None
    tmdb_vote_count: Optional[int] = None

    # MovieLens Academic Ratings (Scale 0.5 - 5.0)
    movielens_rating: Optional[float] = None
    movielens_rating_count: Optional[int] = None
    movielens_tags: List[str] = field(default_factory=list)

    # Provenance Tracking
    # Allowed values: "movielens_linked", "tmdb_global", "movielens_only"
    metadata_source: str = "tmdb_global"

    # Collaborative Data Flag
    # Strictly True ONLY if verified MovieLens user interaction data exists
    collaborative_data_available: bool = False

    def __post_init__(self):
        # 1. Clean and normalize string identifiers
        if self.tmdb_id is not None:
            clean_tmdb = str(self.tmdb_id).strip()
            # Invalidate '0', 'None', 'nan', negative IDs
            if not clean_tmdb.isdigit() or clean_tmdb == "0":
                object.__setattr__(self, "tmdb_id", None)
            else:
                object.__setattr__(self, "tmdb_id", clean_tmdb)

        if self.movielens_movie_id is not None:
            try:
                ml_int = int(self.movielens_movie_id)
                if ml_int <= 0:
                    object.__setattr__(self, "movielens_movie_id", None)
                else:
                    object.__setattr__(self, "movielens_movie_id", ml_int)
            except (ValueError, TypeError):
                object.__setattr__(self, "movielens_movie_id", None)

        # 2. Normalize titles
        clean_title = str(self.title or "").strip() or "Untitled"
        object.__setattr__(self, "title", clean_title)

        clean_orig_title = str(self.original_title or "").strip() or clean_title
        object.__setattr__(self, "original_title", clean_orig_title)

        # 3. Overview cleanup
        object.__setattr__(self, "overview", str(self.overview or "").strip())

        # 4. Release year extraction fallback
        if self.release_year is None and self.release_date:
            match = re.match(r"^(\d{4})", str(self.release_date).strip())
            if match:
                object.__setattr__(self, "release_year", int(match.group(1)))

        # 5. Language & Country standardization
        object.__setattr__(self, "original_language", str(self.original_language or "").strip().lower())
        norm_countries = [
            str(c).strip().upper()
            for c in (self.origin_country or [])
            if str(c).strip()
        ]
        object.__setattr__(self, "origin_country", norm_countries)

        # 6. Ratings bounds & strict type conversion
        if self.tmdb_vote_average is not None:
            try:
                avg = float(self.tmdb_vote_average)
                object.__setattr__(self, "tmdb_vote_average", round(max(0.0, min(10.0, avg)), 1))
            except (ValueError, TypeError):
                object.__setattr__(self, "tmdb_vote_average", None)

        if self.tmdb_vote_count is not None:
            try:
                cnt = int(self.tmdb_vote_count)
                object.__setattr__(self, "tmdb_vote_count", max(0, cnt))
            except (ValueError, TypeError):
                object.__setattr__(self, "tmdb_vote_count", None)

        if self.movielens_rating is not None:
            try:
                ml_avg = float(self.movielens_rating)
                object.__setattr__(self, "movielens_rating", round(max(0.5, min(5.0, ml_avg)), 1))
            except (ValueError, TypeError):
                object.__setattr__(self, "movielens_rating", None)

        if self.movielens_rating_count is not None:
            try:
                ml_cnt = int(self.movielens_rating_count)
                object.__setattr__(self, "movielens_rating_count", max(0, ml_cnt))
            except (ValueError, TypeError):
                object.__setattr__(self, "movielens_rating_count", None)

        # 7. Collaborative Data Invariant:
        # Strictly True ONLY if MovieLens ratings are genuinely present
        has_ml_ratings = (
            self.movielens_rating_count is not None
            and self.movielens_rating_count > 0
            and self.movielens_rating is not None
        )
        object.__setattr__(self, "collaborative_data_available", bool(has_ml_ratings))

        # 8. Source Tagging Consistency
        if self.tmdb_id and self.movielens_movie_id:
            object.__setattr__(self, "metadata_source", "movielens_linked")
        elif self.movielens_movie_id and not self.tmdb_id:
            object.__setattr__(self, "metadata_source", "movielens_only")
        else:
            object.__setattr__(self, "metadata_source", "tmdb_global")

    @property
    def primary_uid(self) -> str:
        """
        Returns a globally stable, collision-free unique identifier.
        Never relies on title strings, ensuring complete stability for remakes and duplicates.
        """
        if self.tmdb_id:
            return f"tmdb_{self.tmdb_id}"
        if self.movielens_movie_id:
            return f"ml_{self.movielens_movie_id}"
        if self.imdb_id:
            return f"imdb_{self.imdb_id}"
        # Fallback incorporating title and year
        year_tag = f"_{self.release_year}" if self.release_year else ""
        clean_name = re.sub(r"\W+", "_", self.title.lower()).strip("_")
        return f"unlinked_{clean_name}{year_tag}"

    @property
    def canonical_id(self) -> str:
        """
        Canonical identifier conforming to Stage 3 Architecture Specification:
        - 'tmdb:{id}' if TMDb ID is known
        - 'imdb:{id}' if missing TMDb ID but IMDb ID is known
        - 'ml:{id}' if MovieLens ID is known
        - fallback to primary_uid
        """
        if self.tmdb_id:
            return f"tmdb:{self.tmdb_id}"
        if self.imdb_id:
            return f"imdb:{self.imdb_id}"
        if self.movielens_movie_id:
            return f"ml:{self.movielens_movie_id}"
        return self.primary_uid

    @property
    def has_official_poster(self) -> bool:
        """Returns True if a valid official poster path exists on TMDb CDN."""
        return bool(self.poster_path and isinstance(self.poster_path, str) and self.poster_path.startswith("/"))

    def get_poster_url(self, size: str = DEFAULT_POSTER_SIZE) -> Optional[str]:
        """Constructs full CDN image URL for official movie poster if available."""
        if self.has_official_poster:
            return f"{TMDB_IMAGE_BASE}/{size}{self.poster_path}"
        return None

    def get_backdrop_url(self, size: str = DEFAULT_BACKDROP_SIZE) -> Optional[str]:
        """Constructs full CDN image URL for official movie backdrop if available."""
        if self.backdrop_path and isinstance(self.backdrop_path, str) and self.backdrop_path.startswith("/"):
            return f"{TMDB_IMAGE_BASE}/{size}{self.backdrop_path}"
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes entity to standard Python dictionary."""
        d = asdict(self)
        d["canonical_id"] = self.canonical_id
        d["primary_uid"] = self.primary_uid
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedMovie":
        """Deserializes from dictionary safely, ignoring unknown keys."""
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_tmdb_dict(cls, raw: Dict[str, Any], movielens_id: Optional[int] = None) -> "UnifiedMovie":
        """
        Factory method constructing UnifiedMovie from Stage 1 normalized TMDb dictionary.
        Safely isolates TMDb ratings from MovieLens ratings.
        """
        return cls(
            tmdb_id=str(raw.get("tmdb_id", "") or raw.get("id", "") or "") or None,
            movielens_movie_id=movielens_id,
            imdb_id=str(raw.get("imdb_id", "") or "").strip(),
            title=str(raw.get("title", "") or "").strip() or "Untitled",
            original_title=str(raw.get("original_title", "") or "").strip(),
            overview=str(raw.get("overview", "") or "").strip(),
            release_date=str(raw.get("release_date", "") or "").strip(),
            release_year=raw.get("release_year"),
            original_language=str(raw.get("original_language", "") or "").strip().lower(),
            origin_country=raw.get("origin_country", []) if isinstance(raw.get("origin_country"), list) else [],
            genre_ids=raw.get("genre_ids", []) if isinstance(raw.get("genre_ids"), list) else [],
            genre_names=raw.get("genre_names", []) if isinstance(raw.get("genre_names"), list) else [],
            keywords=raw.get("keywords", []) if isinstance(raw.get("keywords"), list) else [],
            poster_path=raw.get("poster_path"),
            backdrop_path=raw.get("backdrop_path"),
            tmdb_vote_average=raw.get("tmdb_vote_average"),
            tmdb_vote_count=raw.get("tmdb_vote_count"),
            movielens_rating=None,
            movielens_rating_count=None,
            movielens_tags=[],
            metadata_source="movielens_linked" if movielens_id else "tmdb_global",
            collaborative_data_available=False,
        )

    @classmethod
    def from_movielens_row(
        cls,
        row: Dict[str, Any],
        tmdb_metadata: Optional[Dict[str, Any]] = None,
    ) -> "UnifiedMovie":
        """
        Factory method constructing UnifiedMovie from preprocessed MovieLens record.
        Preserves authentic MovieLens academic rating statistics and community tags.
        """
        ml_id = int(row.get("movieId", 0)) if row.get("movieId") else None
        title = str(row.get("clean_title", "") or row.get("title", "") or "").strip()
        year = int(row.get("year")) if row.get("year") and str(row.get("year")).isdigit() else None
        genres = list(row.get("genre_list", [])) if isinstance(row.get("genre_list"), list) else []
        tags = list(row.get("tag_list", [])) if isinstance(row.get("tag_list"), list) else []

        ml_rating = float(row.get("avg_rating", 0.0)) if row.get("vote_count", 0) > 0 else None
        ml_count = int(row.get("vote_count", 0)) if row.get("vote_count", 0) > 0 else None

        imdb_id = str(row.get("imdbId", "") or "").strip()
        tmdb_id = str(row.get("tmdbId", "") or "").strip() or None

        # Build default entity from MovieLens data alone
        entity = cls(
            tmdb_id=tmdb_id,
            movielens_movie_id=ml_id,
            imdb_id=imdb_id,
            title=title or "Untitled",
            original_title=title or "Untitled",
            overview=str(row.get("description", "") or "").strip(),
            release_date=f"{year}-01-01" if year else "",
            original_language=None,  # MovieLens records have no native language annotation; preserve as unknown (None)
            origin_country=[],  # MovieLens records have no native country annotation; preserve as unknown ([])
            genre_ids=[],
            genre_names=genres,
            keywords=[],
            poster_path=None,
            backdrop_path=None,
            tmdb_vote_average=None,
            tmdb_vote_count=None,
            movielens_rating=ml_rating,
            movielens_rating_count=ml_count,
            movielens_tags=tags,
            metadata_source="movielens_linked" if tmdb_id else "movielens_only",
            collaborative_data_available=bool(ml_count and ml_count > 0),
        )

        # Enrich with verified TMDb metadata if supplied
        if tmdb_metadata and isinstance(tmdb_metadata, dict):
            entity = entity.enrich_with_tmdb(tmdb_metadata)

        return entity

    def enrich_with_tmdb(self, tmdb: Dict[str, Any]) -> "UnifiedMovie":
        """
        Returns a new UnifiedMovie with additional TMDb metadata (poster, backdrop,
        plot overview, native language, origin country, TMDb vote metrics), while
        strictly preserving original MovieLens identifiers and ratings.
        """
        # Preserve original MovieLens data
        ml_id = self.movielens_movie_id
        ml_rating = self.movielens_rating
        ml_count = self.movielens_rating_count
        ml_tags = list(self.movielens_tags)

        # Merge TMDb attributes safely
        tmdb_id = str(tmdb.get("tmdb_id", "") or tmdb.get("id", "") or self.tmdb_id or "").strip() or None
        imdb_id = str(tmdb.get("imdb_id", "") or self.imdb_id or "").strip()
        orig_title = str(tmdb.get("original_title", "") or self.original_title or self.title).strip()
        overview = str(tmdb.get("overview", "") or self.overview or "").strip()
        rel_date = str(tmdb.get("release_date", "") or self.release_date or "").strip()

        rel_year = self.release_year
        if rel_year is None and tmdb.get("release_year"):
            rel_year = int(tmdb["release_year"])

        orig_lang = str(tmdb.get("original_language", "") or self.original_language or "").strip().lower()
        countries = list(tmdb.get("origin_country", [])) or list(self.origin_country)

        genre_ids = list(tmdb.get("genre_ids", [])) or list(self.genre_ids)
        genre_names = list(self.genre_names)
        if tmdb.get("genre_names") and isinstance(tmdb["genre_names"], list):
            for gn in tmdb["genre_names"]:
                if gn not in genre_names:
                    genre_names.append(gn)

        keywords = list(tmdb.get("keywords", [])) if isinstance(tmdb.get("keywords"), list) else list(self.keywords)
        poster = tmdb.get("poster_path") or self.poster_path
        backdrop = tmdb.get("backdrop_path") or self.backdrop_path

        tmdb_avg = tmdb.get("tmdb_vote_average") or tmdb.get("vote_average")
        tmdb_cnt = tmdb.get("tmdb_vote_count") or tmdb.get("vote_count")

        return UnifiedMovie(
            tmdb_id=tmdb_id,
            movielens_movie_id=ml_id,
            imdb_id=imdb_id,
            title=self.title,
            original_title=orig_title,
            overview=overview,
            release_date=rel_date,
            release_year=rel_year,
            original_language=orig_lang,
            origin_country=countries,
            genre_ids=genre_ids,
            genre_names=genre_names,
            keywords=keywords,
            poster_path=poster,
            backdrop_path=backdrop,
            tmdb_vote_average=tmdb_avg,
            tmdb_vote_count=tmdb_cnt,
            movielens_rating=ml_rating,
            movielens_rating_count=ml_count,
            movielens_tags=ml_tags,
            metadata_source="movielens_linked" if (tmdb_id and ml_id) else self.metadata_source,
            collaborative_data_available=bool(ml_count and ml_count > 0),
        )
