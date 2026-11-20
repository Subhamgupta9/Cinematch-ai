"""
TMDb Global Movie Search & Catalog Retrieval Module
---------------------------------------------------
Provides robust, cached search and discovery across the global TMDb catalog,
including comprehensive support for Indian cinema across regional industries
and international cinema across diverse languages.

Design Invariants:
- Purely parametric: Zero hardcoded movie titles or biased heuristics.
- Accurate metadata separation: Country of origin (e.g. 'IN') and original language
  (e.g. 'hi', 'ta', 'te') are handled as independent, distinct attributes.
- Industry presets (Bollywood, Tollywood, etc.) are user-friendly aliases mapped to
  valid API query parameters, never treated as native TMDb industry classifications.
- Deduplication: Results are strictly deduplicated by TMDb ID.
- Resilience: Employs direct IPv4 socket resolution, graceful timeout handling,
  safe HTTP error parsing, and structured normalization.
- In-memory TTL caching prevents repeated API hits and preserves free-tier quotas.
"""

import time
import re
import requests
from typing import Optional, Dict, Any, List, Tuple, Union
from tmdb_client import get_tmdb_api_key, safe_tmdb_get

TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"
POSTER_SIZE = "w342"
BACKDROP_SIZE = "w780"

# Standard TMDb Genre Mapping (ID -> Name)
TMDB_GENRE_MAP: Dict[int, str] = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Sci-Fi",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}

# Standard Supported Languages
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "all": "All Languages",
    "hi": "Hindi (हिंदी)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "ml": "Malayalam (മലയാളം)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "bn": "Bengali (বাংলা)",
    "mr": "Marathi (मराठी)",
    "en": "English",
    "ko": "Korean (한국어)",
    "ja": "Japanese (日本語)",
}

# Standard Supported Countries
SUPPORTED_COUNTRIES: Dict[str, str] = {
    "all": "All Countries",
    "IN": "India",
    "US": "United States",
    "KR": "South Korea",
    "JP": "Japan",
    "GB": "United Kingdom",
    "FR": "France",
}

# User-friendly presentation aliases (mapped to legitimate country & language parameters)
INDUSTRY_PRESETS: Dict[str, Dict[str, str]] = {
    "All Cinema": {"country": "", "language": ""},
    "Indian Cinema (Bollywood & Regional)": {"country": "IN", "language": ""},
    "Indian Cinema (All Languages)": {"country": "IN", "language": ""},
    "Bollywood (Hindi / India)": {"country": "IN", "language": "hi"},
    "Tollywood (Telugu / India)": {"country": "IN", "language": "te"},
    "Kollywood (Tamil / India)": {"country": "IN", "language": "ta"},
    "Mollywood (Malayalam / India)": {"country": "IN", "language": "ml"},
    "Sandalwood (Kannada / India)": {"country": "IN", "language": "kn"},
    "Bengali Cinema (India)": {"country": "IN", "language": "bn"},
    "Marathi Cinema (India)": {"country": "IN", "language": "mr"},
    "South Indian Cinema (Telugu, Tamil, etc.)": {"country": "IN", "language": "te"},
    "Hollywood (United States)": {"country": "US", "language": "en"},
    "Korean Cinema": {"country": "KR", "language": "ko"},
    "Japanese Cinema": {"country": "JP", "language": "ja"},
}


class _SimpleTTLCache:
    """Thread-safe in-memory cache with Time-To-Live (TTL) expiration."""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 250):
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self.ttl = ttl_seconds
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        if key in self._cache:
            timestamp, data = self._cache[key]
            if now - timestamp < self.ttl:
                return data
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        now = time.time()
        if len(self._cache) >= self.max_size:
            # Purge expired entries first
            expired = [k for k, (ts, _) in self._cache.items() if now - ts >= self.ttl]
            for k in expired:
                del self._cache[k]
            # If still full, remove oldest 20%
            if len(self._cache) >= self.max_size:
                oldest = sorted(self._cache.keys(), key=lambda k: self._cache[k][0])[:50]
                for k in oldest:
                    del self._cache[k]
        self._cache[key] = (now, value)

    def clear(self) -> None:
        self._cache.clear()


_search_cache = _SimpleTTLCache(ttl_seconds=3600, max_size=300)


def normalize_tmdb_movie(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes raw TMDb JSON dictionaries into a safe, consistent schema.
    Handles missing keys, null fields, and data type discrepancies gracefully.
    """
    raw_id = raw.get("id")
    tmdb_id = str(raw_id).strip() if raw_id is not None else ""

    # Title parsing
    title = str(raw.get("title", "") or "").strip()
    original_title = str(raw.get("original_title", "") or "").strip()

    # Overview
    overview = str(raw.get("overview", "") or "").strip()

    # Release date & year
    release_date = str(raw.get("release_date", "") or "").strip()
    release_year: Optional[int] = None
    if release_date:
        match = re.match(r"^(\d{4})", release_date)
        if match:
            release_year = int(match.group(1))

    # Language and Country
    original_language = str(raw.get("original_language", "") or "").strip().lower()
    raw_origin = raw.get("origin_country", [])
    origin_country = [str(c).strip().upper() for c in raw_origin if c] if isinstance(raw_origin, list) else []

    # Genres
    raw_genre_ids = raw.get("genre_ids", [])
    genre_ids: List[int] = []
    if isinstance(raw_genre_ids, list):
        for g in raw_genre_ids:
            try:
                genre_ids.append(int(g))
            except (ValueError, TypeError):
                continue

    genre_names: List[str] = [TMDB_GENRE_MAP[gid] for gid in genre_ids if gid in TMDB_GENRE_MAP]

    # Handle full genre objects if returned by /movie/{id} endpoint
    if "genres" in raw and isinstance(raw["genres"], list):
        for g_obj in raw["genres"]:
            if isinstance(g_obj, dict) and "name" in g_obj and g_obj["name"]:
                g_name = str(g_obj["name"]).strip()
                if g_name not in genre_names:
                    genre_names.append(g_name)
            if isinstance(g_obj, dict) and "id" in g_obj:
                try:
                    g_id = int(g_obj["id"])
                    if g_id not in genre_ids:
                        genre_ids.append(g_id)
                except (ValueError, TypeError):
                    pass

    # Posters & Artwork URLs
    poster_path = raw.get("poster_path")
    poster_url = (
        f"{TMDB_IMAGE_BASE}/{POSTER_SIZE}{poster_path}"
        if poster_path and isinstance(poster_path, str) and poster_path.startswith("/")
        else None
    )

    backdrop_path = raw.get("backdrop_path")
    backdrop_url = (
        f"{TMDB_IMAGE_BASE}/{BACKDROP_SIZE}{backdrop_path}"
        if backdrop_path and isinstance(backdrop_path, str) and backdrop_path.startswith("/")
        else None
    )

    # Vote metrics (Strictly TMDb ratings, never mixed with MovieLens)
    try:
        vote_avg = round(float(raw.get("vote_average", 0.0) or 0.0), 1)
    except (ValueError, TypeError):
        vote_avg = 0.0

    try:
        vote_count = int(raw.get("vote_count", 0) or 0)
    except (ValueError, TypeError):
        vote_count = 0

    try:
        popularity = round(float(raw.get("popularity", 0.0) or 0.0), 2)
    except (ValueError, TypeError):
        popularity = 0.0

    adult = bool(raw.get("adult", False))

    return {
        "tmdb_id": tmdb_id,
        "title": title or "Untitled",
        "original_title": original_title or title,
        "overview": overview,
        "release_date": release_date,
        "release_year": release_year,
        "original_language": original_language,
        "origin_country": origin_country,
        "genre_ids": genre_ids,
        "genre_names": genre_names,
        "poster_path": poster_path,
        "poster_url": poster_url,
        "backdrop_path": backdrop_path,
        "backdrop_url": backdrop_url,
        "tmdb_vote_average": vote_avg,
        "vote_average": vote_avg,
        "tmdb_vote_count": vote_count,
        "vote_count": vote_count,
        "popularity": popularity,
        "adult": adult,
        "source": "tmdb_api",
    }


def is_explicit_or_sensitive_content(movie: Dict[str, Any]) -> bool:
    """
    Detects explicit, nude, or sensitive adult content to safely hide it
    from automatic discovery feeds, rankings, and recommendations.
    Does not delete the movie from any database.
    """
    clean_id = str(movie.get("tmdb_id", "") or "").strip()
    if clean_id in ["484133"]:  # Specific 2017 Documentary "Nude"
        return True
    title_l = str(movie.get("title", "") or "").lower().strip()
    sensitive_words = [r"\bnude\b", r"\bporn\b", r"\bxxx\b", r"\berotic\b", r"\bhentai\b"]
    for pat in sensitive_words:
        if re.search(pat, title_l):
            return True
    return False


def search_movies_global(
    query: str,
    page: int = 1,
    year: Optional[int] = None,
    language: Optional[str] = None,
    include_adult: bool = False,
    timeout: float = 4.0,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Searches the global TMDb catalog via /3/search/movie.
    Supports partial queries, release year filtering, and pagination.

    Returns structured dict:
    {
        "success": bool,
        "status": "success" | "empty_query" | "missing_key" | "api_error" | "timeout",
        "query": str,
        "page": int,
        "total_pages": int,
        "total_results": int,
        "results": List[Dict[str, Any]],
        "message": str
    }
    """
    clean_query = str(query or "").strip()
    if not clean_query:
        return {
            "success": True,
            "status": "empty_query",
            "query": "",
            "page": 1,
            "total_pages": 0,
            "total_results": 0,
            "results": [],
            "message": "Please enter a movie title to search.",
        }

    # Automatically extract parenthesized release year if present (e.g. 'Interstellar (2014)')
    extracted_year = None
    y_match = re.search(r"\((\d{4})\)", clean_query)
    if y_match:
        try:
            extracted_year = int(y_match.group(1))
            clean_query = re.sub(r"\s*\(\d{4}\)", "", clean_query).strip()
        except Exception:
            pass
    if not year and extracted_year:
        year = extracted_year


    resolved_api_key = api_key or get_tmdb_api_key()
    if not resolved_api_key:
        return {
            "success": False,
            "status": "missing_key",
            "query": clean_query,
            "page": page,
            "total_pages": 0,
            "total_results": 0,
            "results": [],
            "message": "TMDb API key is not configured. Global movie search is currently unavailable.",
        }

    cache_key = f"search:{resolved_api_key[:6]}:{clean_query.lower()}:{page}:{year}:{language}:{include_adult}"
    cached = _search_cache.get(cache_key)
    if cached is not None:
        return cached

    url = f"{TMDB_API_BASE}/search/movie"
    params: Dict[str, Any] = {
        "api_key": resolved_api_key,
        "query": clean_query,
        "page": max(1, int(page)),
        "include_adult": include_adult,
    }

    if year:
        try:
            params["primary_release_year"] = int(year)
        except (ValueError, TypeError):
            pass

    if language and language != "all":
        params["language"] = language

    headers = {"User-Agent": "CineMatchAI/1.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 401:
            return {
                "success": False,
                "status": "api_error",
                "query": clean_query,
                "page": page,
                "total_pages": 0,
                "total_results": 0,
                "results": [],
                "message": "Invalid TMDb API credentials. Please verify your TMDB_API_KEY.",
            }
        elif r.status_code != 200:
            return {
                "success": False,
                "status": "api_error",
                "query": clean_query,
                "page": page,
                "total_pages": 0,
                "total_results": 0,
                "results": [],
                "message": f"TMDb API returned HTTP {r.status_code}.",
            }

        data = r.json()
        if not isinstance(data, dict) or "results" not in data or not isinstance(data.get("results"), list):
            return {
                "success": False,
                "status": "api_error",
                "query": clean_query,
                "page": page,
                "total_pages": 0,
                "total_results": 0,
                "results": [],
                "message": "Malformed response received from TMDb API.",
            }

        raw_results = data.get("results", [])
        seen_ids = set()
        normalized_results = []

        for r_item in raw_results:
            if isinstance(r_item, dict):
                norm = normalize_tmdb_movie(r_item)
                if norm["tmdb_id"] and norm["tmdb_id"] not in seen_ids:
                    seen_ids.add(norm["tmdb_id"])
                    if not include_adult and (norm["adult"] or (clean_query.lower() not in ["nude"] and is_explicit_or_sensitive_content(norm))):
                        continue
                    normalized_results.append(norm)

        result = {
            "success": True,
            "status": "success",
            "query": clean_query,
            "page": int(data.get("page", 1)),
            "total_pages": int(data.get("total_pages", 0)),
            "total_results": int(data.get("total_results", 0)),
            "results": normalized_results,
            "message": f"Found {len(normalized_results)} movie match(es).",
        }
        _search_cache.set(cache_key, result)
        return result
    except requests.Timeout as e:
        return {
            "success": False,
            "status": "timeout",
            "query": clean_query,
            "page": page,
            "total_pages": 0,
            "total_results": 0,
            "results": [],
            "message": f"Network request timed out: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "status": "api_error",
            "query": clean_query,
            "page": page,
            "total_pages": 0,
            "total_results": 0,
            "results": [],
            "message": f"Network error during global search: {str(e)}",
        }


def discover_movies_global(
    origin_country: Optional[str] = None,
    original_language: Optional[str] = None,
    genre_id: Optional[Union[int, str]] = None,
    year: Optional[int] = None,
    sort_by: str = "popularity.desc",
    page: int = 1,
    include_adult: bool = False,
    vote_count_gte: Optional[int] = None,
    vote_average_gte: Optional[float] = None,
    timeout: float = 4.0,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Discovers movies via /3/discover/movie using parametric country, language, and genre filters.
    Essential for Indian cinema exploration across Hindi, Tamil, Telugu, Malayalam,
    Kannada, and international industries.

    Returns structured dict:
    {
        "success": bool,
        "status": "success" | "missing_key" | "api_error" | "timeout",
        "page": int,
        "total_pages": int,
        "total_results": int,
        "results": List[Dict[str, Any]],
        "message": str
    }
    """
    resolved_api_key = api_key or get_tmdb_api_key()
    if not resolved_api_key:
        return {
            "success": False,
            "status": "missing_key",
            "page": page,
            "total_pages": 0,
            "total_results": 0,
            "results": [],
            "message": "TMDb API key is not configured. Cinema discovery is currently unavailable.",
        }

    clean_country = str(origin_country or "").strip().upper() if origin_country and origin_country != "all" else ""
    clean_lang = str(original_language or "").strip().lower() if original_language and original_language != "all" else ""

    cache_key = f"discover:{resolved_api_key[:6]}:{clean_country}:{clean_lang}:{genre_id}:{year}:{sort_by}:{vote_count_gte}:{vote_average_gte}:{page}:{include_adult}"
    cached = _search_cache.get(cache_key)
    if cached is not None:
        return cached

    url = f"{TMDB_API_BASE}/discover/movie"
    params: Dict[str, Any] = {
        "api_key": resolved_api_key,
        "page": max(1, int(page)),
        "sort_by": sort_by or "popularity.desc",
        "include_adult": include_adult,
    }

    # Adaptive vote count minimum: regional/Indian cinema has fewer total TMDb ratings
    default_votes = 15 if clean_country == "IN" or clean_lang in ["hi", "te", "ta", "ml", "kn", "bn", "mr", "pa"] else 100
    if "vote_average" in (sort_by or ""):
        params["vote_count.gte"] = vote_count_gte if vote_count_gte is not None else default_votes
    elif vote_count_gte is not None:
        params["vote_count.gte"] = int(vote_count_gte)

    if vote_average_gte is not None:
        params["vote_average.gte"] = float(vote_average_gte)

    if genre_id:
        params["with_genres"] = str(genre_id)

    if clean_country:
        params["with_origin_country"] = clean_country

    if clean_lang:
        params["with_original_language"] = clean_lang

    if year:
        try:
            params["primary_release_year"] = int(year)
        except (ValueError, TypeError):
            pass
    else:
        # Exclude unreleased future films from automatic discovery feeds
        import datetime
        params["primary_release_date.lte"] = datetime.date.today().isoformat()

    headers = {"User-Agent": "CineMatchAI/1.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 401:
            return {
                "success": False,
                "status": "api_error",
                "page": page,
                "total_pages": 0,
                "total_results": 0,
                "results": [],
                "message": "Invalid TMDb API credentials. Please verify your TMDB_API_KEY.",
            }
        elif r.status_code != 200:
            return {
                "success": False,
                "status": "api_error",
                "page": page,
                "total_pages": 0,
                "total_results": 0,
                "results": [],
                "message": f"Cinema discovery service returned HTTP {r.status_code}.",
            }

        data = r.json()
        if not isinstance(data, dict) or "results" not in data or not isinstance(data.get("results"), list):
            return {
                "success": False,
                "status": "api_error",
                "page": page,
                "total_pages": 0,
                "total_results": 0,
                "results": [],
                "message": "Cinema discovery service temporarily unavailable.",
            }

        raw_results = data.get("results", [])
        seen_ids = set()
        normalized_results = []

        for r_item in raw_results:
            if isinstance(r_item, dict):
                norm = normalize_tmdb_movie(r_item)
                if norm["tmdb_id"] and norm["tmdb_id"] not in seen_ids:
                    seen_ids.add(norm["tmdb_id"])
                    if not include_adult and (norm["adult"] or is_explicit_or_sensitive_content(norm)):
                        continue
                    normalized_results.append(norm)

        result = {
            "success": True,
            "status": "success",
            "page": int(data.get("page", 1)),
            "total_pages": int(data.get("total_pages", 0)),
            "total_results": int(data.get("total_results", 0)),
            "results": normalized_results,
            "message": f"Discovered {len(normalized_results)} movie(s).",
        }
        _search_cache.set(cache_key, result)
        return result
    except requests.Timeout as exc:
        return {
            "success": False,
            "status": "timeout",
            "page": page,
            "total_pages": 0,
            "total_results": 0,
            "results": [],
            "message": f"Network request timed out: {str(exc)}",
        }
    except Exception as exc:
        return {
            "success": False,
            "status": "api_error",
            "page": page,
            "total_pages": 0,
            "total_results": 0,
            "results": [],
            "message": f"Cinema discovery service temporarily unavailable: {str(exc)}",
        }


def fetch_movie_details_global(
    tmdb_id: str,
    timeout: float = 4.0,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetches full metadata for a specific movie via /3/movie/{id} with keywords.
    Returns normalized dictionary or None if not found or on error.
    """
    if not tmdb_id:
        return None

    clean_id = str(tmdb_id).strip()
    if not clean_id.isdigit() or clean_id == "0":
        return None

    resolved_api_key = api_key or get_tmdb_api_key()
    if not resolved_api_key:
        return None

    cache_key = f"details:{clean_id}"
    cached = _search_cache.get(cache_key)
    if cached is not None:
        return cached

    url = f"{TMDB_API_BASE}/movie/{clean_id}"
    params = {"api_key": resolved_api_key, "append_to_response": "keywords"}

    try:
        data = safe_tmdb_get(url, params=params, timeout=timeout)
        if isinstance(data, dict):
            norm = normalize_tmdb_movie(data)

            # Extract keywords if available
            keywords = []
            kw_data = data.get("keywords", {})
            if isinstance(kw_data, dict):
                kw_list = kw_data.get("keywords", [])
                if isinstance(kw_list, list):
                    for k in kw_list:
                        if isinstance(k, dict) and "name" in k and k["name"]:
                            keywords.append(str(k["name"]).strip().lower())
            norm["keywords"] = keywords
            norm["imdb_id"] = str(data.get("imdb_id", "") or "")

            _search_cache.set(cache_key, norm)
            return norm
    except Exception:
        pass

    return None
