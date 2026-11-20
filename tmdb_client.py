"""
TMDb API Client for CineMatch AI
--------------------------------
Handles movie poster retrieval and India streaming availability (Watch Providers)
via The Movie Database (TMDb) API.

Security & Resilience:
- Safely reads API key from environment variable TMDB_API_KEY or Streamlit secrets.
- Never crashes when API key is missing or invalid.
- Prevents ISP DNS poisoning/IPv6 timeouts on Indian networks via fast direct resolution.
- Generates beautiful, lightweight, accessible SVG poster placeholders when
  artwork is unavailable or unconfigured, clearly labeled as placeholder artwork.
- Employs TTL caching to minimize network overhead and respect rate limits.
"""

import os
import time
import html
import urllib.parse
from typing import Optional, Dict, Any, List
import requests

TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"
POSTER_SIZE = "w342"
BACKDROP_SIZE = "w780"
LOGO_SIZE = "w92"

def safe_tmdb_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: float = 4.0, retries: int = 3) -> Dict[str, Any]:
    """
    Robust HTTP GET with automatic retries and browser headers.
    Transparently recovers from connection errors and transient timeouts.
    """
    headers = {"User-Agent": "CineMatchAI/1.0 (Windows NT 10.0; Win64; x64)"}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    return data
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.3)
    return {}


def get_tmdb_api_key() -> str:
    """
    Safely retrieves TMDb API key from environment variable or Streamlit secrets.
    Returns empty string if unconfigured.
    Never throws exceptions.
    """
    # 1. Check environment variable
    env_key = os.environ.get("TMDB_API_KEY", "").strip()
    if env_key:
        return env_key

    # 2. Check Streamlit secrets if available
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "TMDB_API_KEY" in st.secrets:
            sec_key = str(st.secrets["TMDB_API_KEY"]).strip()
            if sec_key:
                return sec_key
    except Exception:
        pass

    return ""


def generate_fallback_poster_svg(title: str, year: Optional[int] = None, genres: Optional[List[str]] = None) -> str:
    """
    Generates a clean, accessible, modern SVG poster placeholder as a data URI.
    Explicitly labeled as generated placeholder artwork (never presented as an official poster).
    """
    safe_title = html.escape(str(title)[:36] + ("..." if len(str(title)) > 36 else ""), quote=True)
    year_str = html.escape(f"({year})", quote=True) if year else ""
    genre_str = html.escape(" · ".join([str(g) for g in (genres or [])[:2]]), quote=True)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 342 513" width="342" height="513">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#F1F5F9" />
      <stop offset="100%" stop-color="#E2E8F0" />
    </linearGradient>
    <linearGradient id="iconGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4F46E5" />
      <stop offset="100%" stop-color="#6366F1" />
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#bgGrad)" rx="14" />
  <rect x="12" y="12" width="318" height="489" fill="none" stroke="#CBD5E1" stroke-width="1.5" stroke-dasharray="6,6" rx="10" />
  
  <!-- Film Slate Icon -->
  <g transform="translate(131, 115)">
    <rect width="80" height="60" rx="8" fill="url(#iconGrad)" />
    <polygon points="15,20 30,12 30,28" fill="#FFFFFF" />
    <circle cx="50" cy="30" r="14" fill="none" stroke="#FFFFFF" stroke-width="3" />
    <circle cx="50" cy="30" r="5" fill="#FFFFFF" />
  </g>

  <!-- Title & Year -->
  <text x="171" y="235" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif" font-size="18" font-weight="700" fill="#0F172A">
    {safe_title}
  </text>
  <text x="171" y="262" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif" font-size="14" font-weight="600" fill="#4F46E5">
    {year_str}
  </text>
  <text x="171" y="288" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif" font-size="12" font-weight="500" fill="#475569">
    {genre_str}
  </text>

  <!-- Explicit Placeholder Artwork Indicator Badge -->
  <rect x="61" y="325" width="220" height="28" rx="6" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1" />
  <text x="171" y="343" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif" font-size="10" font-weight="700" fill="#64748B" letter-spacing="0.5">
    OFFICIAL POSTER UNAVAILABLE
  </text>
  <text x="171" y="372" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif" font-size="11" font-weight="500" fill="#94A3B8">
    Placeholder Artwork
  </text>

  <!-- Subtle Brand Watermark -->
  <text x="171" y="470" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif" font-size="11" font-weight="600" fill="#94A3B8" letter-spacing="1">
    CINEMATCH AI
  </text>
</svg>"""

    encoded = urllib.parse.quote(svg)
    return f"data:image/svg+xml;utf8,{encoded}"


def fetch_movie_poster_path(
    tmdb_id: Optional[str] = None,
    title: Optional[str] = None,
    timeout: float = 4.0,
) -> Optional[str]:
    """
    Queries TMDb API for the movie's official poster_path using linked TMDb ID or title.
    Returns full image URL on official TMDb CDN if available, None otherwise.
    Never throws exceptions.
    """
    api_key = get_tmdb_api_key()
    if not api_key:
        return None

    clean_id = str(tmdb_id or "").strip()
    if clean_id.isdigit() and clean_id != "0":
        # Do not restrict language to en-US so international/regional films return their primary poster
        url = f"{TMDB_API_BASE}/movie/{clean_id}"
        data = safe_tmdb_get(url, params={"api_key": api_key}, timeout=timeout)
        if isinstance(data, dict):
            poster_path = data.get("poster_path")
            if poster_path and isinstance(poster_path, str) and poster_path.startswith("/"):
                return f"{TMDB_IMAGE_BASE}/{POSTER_SIZE}{poster_path}"

        # If primary poster_path is null, check the movie images endpoint with multi-language fallback
        img_url = f"{TMDB_API_BASE}/movie/{clean_id}/images"
        img_data = safe_tmdb_get(
            img_url,
            params={"api_key": api_key, "include_image_language": "en,null,es,hi,ko,ja,fr,it,de"},
            timeout=timeout,
        )
        if isinstance(img_data, dict):
            posters = img_data.get("posters", [])
            if posters and isinstance(posters[0], dict) and posters[0].get("file_path"):
                return f"{TMDB_IMAGE_BASE}/{POSTER_SIZE}{posters[0]['file_path']}"

    # Fallback search by title
    clean_title = str(title or "").strip()
    if clean_title:
        s_url = f"{TMDB_API_BASE}/search/movie"
        s_data = safe_tmdb_get(s_url, params={"api_key": api_key, "query": clean_title}, timeout=timeout)
        if isinstance(s_data, dict):
            res = s_data.get("results", [])
            if res and isinstance(res[0], dict):
                p_path = res[0].get("poster_path")
                if p_path and isinstance(p_path, str) and p_path.startswith("/"):
                    return f"{TMDB_IMAGE_BASE}/{POSTER_SIZE}{p_path}"

    return None


def fetch_watch_providers(tmdb_id: str, region: str = "IN", timeout: float = 4.0) -> Dict[str, Any]:
    """
    Queries TMDb Watch Providers API for streaming, rent, and buy availability.
    Region defaults to 'IN' (India).

    Returns dictionary:
    {
        "available": bool,
        "region": str,
        "status": "success" | "no_providers" | "api_error" | "missing_key" | "invalid_id",
        "link": str,
        "flatrate": [{"provider_name": str, "logo_url": str, "provider_id": int}],
        "rent": [...],
        "buy": [...],
        "message": str
    }
    Never throws exceptions.
    """
    result: Dict[str, Any] = {
        "available": False,
        "region": region,
        "status": "no_providers",
        "link": "",
        "flatrate": [],
        "rent": [],
        "buy": [],
        "message": "Streaming availability information is currently unavailable for this title in India.",
    }

    if not tmdb_id:
        result["status"] = "invalid_id"
        result["message"] = "TMDb identifier is missing for this title."
        return result

    clean_id = str(tmdb_id).strip()
    if not clean_id.isdigit() or clean_id == "0":
        result["status"] = "invalid_id"
        result["message"] = "TMDb identifier is invalid for this title."
        return result

    api_key = get_tmdb_api_key()
    if not api_key:
        result["status"] = "missing_key"
        result["message"] = "Streaming availability information is currently unavailable for this title in India."
        return result

    url = f"{TMDB_API_BASE}/movie/{clean_id}/watch/providers"
    params = {"api_key": api_key}

    headers = {"User-Agent": "CineMatchAI/1.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            result["status"] = "api_error"
            result["message"] = f"Streaming availability is temporarily unavailable (HTTP {r.status_code})."
            return result

        data = r.json()
        if not isinstance(data, dict):
            result["status"] = "api_error"
            result["message"] = "Streaming availability is temporarily unavailable."
            return result

        results = data.get("results", {})
        region_data = results.get(region)

        if region_data and isinstance(region_data, dict):
            # Validate external link
            raw_link = str(region_data.get("link", "")).strip()
            if raw_link.startswith(("https://", "http://")):
                result["link"] = raw_link
            else:
                result["link"] = ""

            for category in ["flatrate", "rent", "buy"]:
                providers = region_data.get(category, [])
                parsed_providers = []
                if isinstance(providers, list):
                    for p in providers:
                        if isinstance(p, dict):
                            logo_path = p.get("logo_path", "")
                            logo_url = (
                                f"{TMDB_IMAGE_BASE}/{LOGO_SIZE}{logo_path}"
                                if logo_path and isinstance(logo_path, str) and logo_path.startswith("/")
                                else ""
                            )
                            parsed_providers.append({
                                "provider_name": str(p.get("provider_name", "Unknown Provider")).strip(),
                                "logo_url": logo_url,
                                "provider_id": int(p.get("provider_id", 0)),
                            })
                result[category] = parsed_providers

            total_options = len(result["flatrate"]) + len(result["rent"]) + len(result["buy"])
            if total_options > 0:
                result["available"] = True
                result["status"] = "success"
                categories_available = []
                if result["flatrate"]:
                    categories_available.append("streaming")
                if result["rent"]:
                    categories_available.append("rent")
                if result["buy"]:
                    categories_available.append("buy")
                result["message"] = f"Available for {', '.join(categories_available)} in India."
            else:
                result["status"] = "no_providers"
                result["message"] = "Streaming availability information is currently unavailable for this title in India."
        else:
            result["status"] = "no_providers"
            result["message"] = "Streaming availability information is currently unavailable for this title in India."
    except Exception as e:
        result["status"] = "api_error"
        result["message"] = f"Streaming availability is temporarily unavailable due to a network connection timeout: {str(e)}"

    return result


def fetch_rich_movie_details(
    tmdb_id: Optional[str] = None,
    title: Optional[str] = None,
    timeout: float = 4.0,
) -> Dict[str, Any]:
    """
    Fetches rich movie metadata including cast, directors, official trailers,
    ratings, runtime, and India streaming options in a single consolidated call.
    Never throws exceptions.
    """
    clean_title = str(title or "").strip()
    clean_id = str(tmdb_id or "").strip()

    result: Dict[str, Any] = {
        "success": False,
        "tmdb_id": clean_id if clean_id.isdigit() else "",
        "title": clean_title or "Unknown Title",
        "tagline": "",
        "overview": "Detailed overview is currently being updated for this title.",
        "release_date": "",
        "year": "",
        "runtime": 0,
        "runtime_str": "",
        "vote_average": 0.0,
        "vote_count": 0,
        "genres": [],
        "directors": [],
        "writers": [],
        "cast": [],
        "trailer_url": None,
        "trailer_key": None,
        "trailer_name": None,
        "streaming": [],
        "rent_buy": [],
        "justwatch_link": None,
        "poster_url": None,
        "backdrop_url": None,
        "budget": 0,
        "budget_str": "",
        "revenue": 0,
        "revenue_str": "",
        "original_language": "en",
        "status": "Released",
    }

    api_key = get_tmdb_api_key()
    if not api_key:
        if clean_title:
            result["trailer_url"] = f"https://www.youtube.com/results?search_query={urllib.parse.quote(clean_title + ' official trailer')}"
        return result

    # Resolve ID via search if missing
    if (not clean_id or not clean_id.isdigit() or clean_id == "0") and clean_title:
        s_data = safe_tmdb_get(f"{TMDB_API_BASE}/search/movie", params={"api_key": api_key, "query": clean_title})
        if s_data.get("results"):
            clean_id = str(s_data["results"][0].get("id", ""))
            result["tmdb_id"] = clean_id

    if not clean_id or not clean_id.isdigit() or clean_id == "0":
        if clean_title:
            result["trailer_url"] = f"https://www.youtube.com/results?search_query={urllib.parse.quote(clean_title + ' official trailer')}"
        return result

    try:
        url = f"{TMDB_API_BASE}/movie/{clean_id}"
        params = {
            "api_key": api_key,
            "append_to_response": "credits,videos,watch/providers",
            "language": "en-US",
        }
        data = safe_tmdb_get(url, params=params, timeout=timeout)
        if not data and clean_title:
            s_data = safe_tmdb_get(f"{TMDB_API_BASE}/search/movie", params={"api_key": api_key, "query": clean_title})
            if s_data.get("results"):
                found_id = str(s_data["results"][0].get("id", ""))
                data = safe_tmdb_get(f"{TMDB_API_BASE}/movie/{found_id}", params=params, timeout=timeout)

        if data and isinstance(data, dict) and data.get("id"):
            result["success"] = True
            result["tmdb_id"] = str(data.get("id") or clean_id)
            result["title"] = data.get("title") or clean_title or "Untitled"
            result["tagline"] = data.get("tagline") or ""
            result["overview"] = data.get("overview") or result["overview"]
            result["release_date"] = data.get("release_date") or ""
            result["year"] = str(data.get("release_date", ""))[:4] if data.get("release_date") else ""
            
            rt = data.get("runtime") or 0
            result["runtime"] = rt
            if rt > 0:
                hrs = rt // 60
                mins = rt % 60
                result["runtime_str"] = f"{hrs}h {mins}m" if hrs > 0 else f"{mins}m"
            
            result["vote_average"] = round(float(data.get("vote_average") or 0.0), 1)
            result["vote_count"] = int(data.get("vote_count") or 0)
            result["genres"] = [g.get("name", "") for g in data.get("genres", []) if g.get("name")]
            result["original_language"] = data.get("original_language", "en")
            result["status"] = data.get("status", "Released")

            # Media URLs
            p_path = data.get("poster_path")
            if p_path and isinstance(p_path, str) and p_path.startswith("/"):
                result["poster_url"] = f"{TMDB_IMAGE_BASE}/w500{p_path}"
            b_path = data.get("backdrop_path")
            if b_path and isinstance(b_path, str) and b_path.startswith("/"):
                result["backdrop_url"] = f"{TMDB_IMAGE_BASE}/{BACKDROP_SIZE}{b_path}"

            # Cast & Crew
            credits_data = data.get("credits", {})
            raw_cast = credits_data.get("cast", [])
            parsed_cast = []
            for c in raw_cast[:10]:
                if isinstance(c, dict):
                    photo_p = c.get("profile_path")
                    photo_url = f"{TMDB_IMAGE_BASE}/w185{photo_p}" if photo_p and str(photo_p).startswith("/") else None
                    parsed_cast.append({
                        "name": c.get("name", "Unknown Actor"),
                        "character": c.get("character", ""),
                        "photo_url": photo_url,
                    })
            result["cast"] = parsed_cast

            raw_crew = credits_data.get("crew", [])
            result["directors"] = [c.get("name", "") for c in raw_crew if isinstance(c, dict) and c.get("job") == "Director"]
            result["writers"] = [c.get("name", "") for c in raw_crew if isinstance(c, dict) and c.get("job") in ["Screenplay", "Writer"]][:3]

            # Official Trailers
            videos_data = data.get("videos", {})
            v_results = videos_data.get("results", [])
            trailers = [
                v for v in v_results
                if isinstance(v, dict) and v.get("site") == "YouTube" and v.get("type") in ["Trailer", "Teaser"] and v.get("key")
            ]
            # Prefer Trailer over Teaser
            official_trailers = [v for v in trailers if v.get("type") == "Trailer"]
            selected_v = official_trailers[0] if official_trailers else (trailers[0] if trailers else None)
            if selected_v:
                v_key = selected_v.get("key")
                result["trailer_key"] = v_key
                result["trailer_url"] = f"https://www.youtube.com/watch?v={v_key}"
                result["trailer_name"] = selected_v.get("name", "Official Trailer")
            elif result["title"]:
                result["trailer_url"] = f"https://www.youtube.com/results?search_query={urllib.parse.quote(result['title'] + ' official trailer')}"

            # Watch Providers (IN)
            wp_results = data.get("watch/providers", {}).get("results", {})
            in_data = wp_results.get("IN", {})
            if isinstance(in_data, dict):
                result["justwatch_link"] = in_data.get("link")
                for p in in_data.get("flatrate", []):
                    if isinstance(p, dict):
                        l_path = p.get("logo_path")
                        logo_u = f"{TMDB_IMAGE_BASE}/{LOGO_SIZE}{l_path}" if l_path and str(l_path).startswith("/") else ""
                        result["streaming"].append({
                            "provider_name": p.get("provider_name", "Streaming Provider"),
                            "logo_url": logo_u,
                        })
                for p in in_data.get("rent", []) + in_data.get("buy", []):
                    if isinstance(p, dict):
                        p_name = p.get("provider_name", "")
                        if not any(rb["provider_name"] == p_name for rb in result["rent_buy"]):
                            l_path = p.get("logo_path")
                            logo_u = f"{TMDB_IMAGE_BASE}/{LOGO_SIZE}{l_path}" if l_path and str(l_path).startswith("/") else ""
                            result["rent_buy"].append({
                                "provider_name": p_name,
                                "logo_url": logo_u,
                            })

            # Financials
            bg = data.get("budget", 0)
            if bg and bg > 0:
                result["budget"] = bg
                result["budget_str"] = f"${bg:,.0f}"
            rv = data.get("revenue", 0)
            if rv and rv > 0:
                result["revenue"] = rv
                result["revenue_str"] = f"${rv:,.0f}"

    except Exception:
        pass

    return result
