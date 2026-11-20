"""
CineMatch AI - Movie Recommendation & Discovery Engine
======================================================
A modern, minimal movie discovery platform built on Streamlit and Python.
Powered by Content-Based Filtering, TF-IDF Vectorization, Sparse Cosine Similarity,
TMDb Global Search, TMDb Posters, and India Streaming Availability.

Design Direction: Minimal Design (Content-focused, calm, high-contrast)
Accessibility Standard: WCAG 2.2 AA (Full contrast, keyboard navigable, 0px overflow)
"""

import sys
import json
import html
import random
import re
from typing import Optional, List, Dict, Any
from pathlib import Path
import streamlit as st

# Configure project path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommender import MovieRecommender
from tmdb_client import (
    get_tmdb_api_key,
    generate_fallback_poster_svg,
    fetch_movie_poster_path,
    fetch_watch_providers,
)
from tmdb_search import (
    search_movies_global,
    discover_movies_global,
    fetch_movie_details_global,
    SUPPORTED_LANGUAGES,
    INDUSTRY_PRESETS,
)
from ui.styles import MINIMAL_THEME_CSS
from ui.components import render_navbar, render_footer, init_session_state, interesting_loading
from ui.page_discover import (
    render_discover_page,
    get_cached_poster,
    fetch_smart_recommendations,
    get_cached_providers,
)
import urllib.parse
from ui.modal_movie_details import render_movie_details_modal

# -----------------------------------------------------------------------------
# Page Configuration (Full-Width Minimal Architecture)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch AI | Find Your Next Favourite Movie",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply Minimal Design CSS Tokens & Global Resets
st.markdown(MINIMAL_THEME_CSS, unsafe_allow_html=True)

# Initialize Session State & Preferences
init_session_state()


# -----------------------------------------------------------------------------
# Model Initialization & Caching (Protected Backend)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading recommendation engine...")
def get_recommender() -> MovieRecommender:
    """
    Initializes and caches the MovieRecommender instance.
    Loads precomputed models/ artifacts if present, otherwise fits on raw data.
    """
    recommender = MovieRecommender()
    if not recommender.load_artifacts():
        recommender.fit()
        recommender.save_artifacts()
    return recommender


recommender = get_recommender()
mem_stats = recommender.get_memory_stats()
all_movie_titles = recommender.get_all_titles()


# -----------------------------------------------------------------------------
# Cached TMDb Poster & Streaming Lookups
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_cached_poster_url(tmdb_id: str, title: str, year: Optional[int], genres: Optional[List[str]]) -> str:
    """Returns official poster image from TMDb CDN if available, else SVG data URI."""
    api_url = fetch_movie_poster_path(tmdb_id)
    if api_url:
        return api_url
    return generate_fallback_poster_svg(title, year, genres)


@st.cache_data(ttl=86400, show_spinner=False)
def get_cached_watch_providers(tmdb_id: str, region: str = "IN") -> Dict[str, Any]:
    """Returns India watch provider availability data cached for 24 hours."""
    return fetch_watch_providers(tmdb_id, region=region)


# -----------------------------------------------------------------------------
# Robust State Synchronization
# -----------------------------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Discover"
if "discover_mode" not in st.session_state:
    st.session_state["discover_mode"] = "✨ By AI Taste"
if "active_query" not in st.session_state:
    st.session_state["active_query"] = "Interstellar (2014)"
if "selected_entity" not in st.session_state:
    st.session_state["selected_entity"] = None

# Apply staged target_query before rendering
if "target_query" in st.session_state and st.session_state.target_query:
    target = st.session_state.pop("target_query")
    if isinstance(target, dict):
        st.session_state["selected_entity"] = target
        t_title = re.sub(r"\s*\(\d{4}\)$", "", str(target.get("title", ""))).strip()
        t_year = f" ({target['release_year']})" if target.get("release_year") else ""
        st.session_state["active_query"] = f"{t_title}{t_year}".strip()
        # Track in history
        history_entry = {
            "title": t_title,
            "year": target.get("release_year"),
            "tmdb_id": target.get("tmdb_id"),
            "language": target.get("original_language", "en"),
            "poster_path": target.get("poster_path"),
        }
        if not any(h.get("title") == t_title for h in st.session_state["history"]):
            st.session_state["history"].insert(0, history_entry)
            st.session_state["history"] = st.session_state["history"][:25]
    else:
        st.session_state["active_query"] = target
        resolved_idx, _ = recommender.find_movie_idx(target)
        if resolved_idx is not None:
            catalog_full_title = recommender.movies_df.iloc[resolved_idx]["title"]
            st.session_state["dropdown_search"] = catalog_full_title
            st.session_state["selected_entity"] = None
            if not any(h.get("title") == catalog_full_title for h in st.session_state["history"]):
                st.session_state["history"].insert(0, {"title": catalog_full_title, "year": None})
                st.session_state["history"] = st.session_state["history"][:25]
        else:
            api_key = get_tmdb_api_key()
            if api_key:
                search_res = search_movies_global(target)
                if search_res.get("success") and search_res.get("results"):
                    top_m = search_res["results"][0]
                    details = fetch_movie_details_global(str(top_m["tmdb_id"])) or top_m
                    st.session_state["selected_entity"] = details
                else:
                    st.session_state["selected_entity"] = None
            else:
                st.session_state["selected_entity"] = None

if "dropdown_search" not in st.session_state:
    st.session_state["dropdown_search"] = st.session_state["active_query"]


def request_movie_switch(query_payload: str | dict):
    """Safely stages a new movie selection to take effect on immediate rerun."""
    st.session_state["target_query"] = query_payload
    st.rerun()


# -----------------------------------------------------------------------------
# Global Movie Details Modal Trigger (Works on all pages & queries)
# -----------------------------------------------------------------------------
if "movie_modal" in st.query_params:
    g_modal_id = st.query_params.pop("movie_modal")
    g_modal_title = st.query_params.pop("movie_title", None)
    try:
        g_modal_score = int(st.query_params.pop("match_score", 0)) or None
    except Exception:
        g_modal_score = None
    st.session_state["active_modal_req"] = {
        "tmdb_id": g_modal_id,
        "title": g_modal_title,
        "match_score": g_modal_score,
    }

if st.session_state.get("active_modal_req"):
    req = st.session_state.pop("active_modal_req")
    render_movie_details_modal(
        tmdb_id=req.get("tmdb_id"),
        title=req.get("title"),
        match_score=req.get("match_score"),
        on_similar_click=request_movie_switch,
    )


# =============================================================================
# TOP NAVIGATION BAR (Replaces Technical Sidebar)
# =============================================================================
active_page = render_navbar()


# =============================================================================
# DESTINATION ROUTER
# =============================================================================

# -----------------------------------------------------------------------------
# PAGE 1: DISCOVER (Primary Homepage)
# -----------------------------------------------------------------------------
if active_page == "Discover":
    render_discover_page(recommender, request_movie_switch)



# -----------------------------------------------------------------------------
# PAGE 2: FIND SIMILAR (Targeted Recommendation Workflow)
# -----------------------------------------------------------------------------
elif active_page == "Find Similar":
    st.markdown(
        """
        <div class="cm-hero">
            <div class="cm-hero-kicker">Targeted Recommendation Flow</div>
            <h1 class="cm-hero-title">Find Similar Movies</h1>
            <div class="cm-hero-subtitle">
                Search any film to discover movies with matching story themes, genres, emotional depth, and cinematic style.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fs_form_col, fs_rand_col = st.columns([3.8, 1.2], vertical_alignment="center")
    with fs_form_col:
        with st.form(key="find_similar_form", clear_on_submit=False):
            fc1, fc2 = st.columns([3.0, 1.2], vertical_alignment="center")
            with fc1:
                search_kw = st.text_input(
                    "Search movie to find similar titles",
                    placeholder="Type any movie (e.g. 3 Idiots, RRR, Inception, Dangal, Interstellar)...",
                    key="fs_search_input",
                    label_visibility="collapsed",
                )
            with fc2:
                submit_fs = st.form_submit_button("🔍 Find Similar", type="primary", use_container_width=True)

            if submit_fs and search_kw and search_kw.strip():
                request_movie_switch(search_kw.strip())

    with fs_rand_col:
        if st.button("🎲 Surprise Pick", use_container_width=True):
            popular_picks = recommender.get_popular_movies(top_n=30)
            rand_movie = random.choice(popular_picks)["title"]
            request_movie_switch(rand_movie)

    current_target = st.session_state.get("active_query", "Interstellar (2014)")
    st.markdown(f"### 🎯 Movies Similar to *{current_target}*")

    with interesting_loading(f"Discovering movies resonant with '{current_target}'..."):
        sim_recs = fetch_smart_recommendations(
            query_title=current_target,
            top_n=8,
        )

    if sim_recs:
        r_cols = st.columns(4)
        for idx, rec in enumerate(sim_recs[:8]):
            col = r_cols[idx % 4]
            with col:
                r_title = rec.get("clean_title") or rec.get("title", "")
                r_year = rec.get("release_year") or rec.get("year") or "N/A"
                r_score = rec.get("explanation", {}).get("match_score_pct") or 85
                r_genres = rec.get("genre_names") or []
                r_tmdb_id = str(rec.get("tmdbId") or rec.get("tmdb_id") or "")
                poster_url = get_cached_poster(r_tmdb_id, r_title, rec.get("year") or rec.get("release_year"), r_genres)
                first_g = r_genres[0] if r_genres else "Cinema"

                encoded_title = urllib.parse.quote(str(r_title))
                modal_url = f"?movie_modal={r_tmdb_id}&movie_title={encoded_title}&match_score={r_score}"

                st.markdown(
                    f"""
                    <a href="{modal_url}" target="_self" class="aura-card-anchor" title="Click to view Cast, Ratings, Trailer & Details for {html.escape(r_title)}">
                        <div class="aura-movie-card">
                            <img src="{poster_url}" class="aura-poster-bg" alt="{html.escape(r_title)}" />
                            <div class="aura-card-vignette"></div>
                            <div class="aura-card-click-banner">
                                <span>▶ Info & Trailer</span>
                            </div>
                            <div class="aura-card-content">
                                <div class="aura-card-title">{html.escape(r_title)}</div>
                                <div class="aura-card-meta">
                                    <span class="aura-score-pill">{r_score}% Match</span>
                                    <span class="aura-genre-pill">{html.escape(str(r_year))}</span>
                                    <span class="aura-genre-pill">{html.escape(first_g)}</span>
                                </div>
                            </div>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
                b_c1, b_c2, b_c3 = st.columns([1.1, 1.0, 1.0])
                with b_c1:
                    if st.button("🎬 Info", key=f"fs_info_{idx}_{r_tmdb_id}", help="View Cast, Ratings, Trailer & Where to watch", use_container_width=True):
                        st.session_state["active_modal_req"] = {
                            "tmdb_id": r_tmdb_id,
                            "title": r_title,
                            "match_score": r_score,
                        }
                        st.rerun()
                with b_c2:
                    if st.button("Similar ↗", key=f"fs_sim_{idx}_{r_tmdb_id}", use_container_width=True):
                        request_movie_switch(r_title)
                with b_c3:
                    is_in_wl = any(w.get("title") == r_title for w in st.session_state.get("watchlist", []))
                    wl_label = "Saved ✓" if is_in_wl else "+ List"
                    if st.button(wl_label, key=f"fs_wl_{idx}_{r_tmdb_id}", use_container_width=True):
                        if is_in_wl:
                            st.session_state["watchlist"] = [w for w in st.session_state["watchlist"] if w.get("title") != r_title]
                            st.toast(f"Removed {r_title} from Watchlist")
                        else:
                            st.session_state["watchlist"].append({
                                "title": r_title,
                                "year": r_year,
                                "tmdb_id": r_tmdb_id,
                                "genres": r_genres,
                                "poster_url": poster_url,
                            })
                            st.toast(f"Added {r_title} to Watchlist!")
                        st.rerun()


# -----------------------------------------------------------------------------
# PAGE 3: EXPLORE (Catalog Browsing)
# -----------------------------------------------------------------------------
elif active_page == "Explore":
    st.markdown(
        """
        <div class="cm-hero">
            <div class="cm-hero-kicker">Catalog Discovery</div>
            <h1 class="cm-hero-title">Explore Cinema</h1>
            <div class="cm-hero-subtitle">
                Browse acclaimed films across Bollywood, regional Indian cinema (Telugu, Tamil, Malayalam, Kannada), and international cinema.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ind_col1, ind_col2 = st.columns([3.0, 1.2], vertical_alignment="center")
    with ind_col1:
        chosen_preset = st.selectbox(
            "Filter by Indian & Regional Film Industry:",
            options=list(INDUSTRY_PRESETS.keys()),
            index=0,
            key="explore_industry_selector",
        )
    with ind_col2:
        sort_mode = st.selectbox(
            "Sort Order:",
            options=["Highest Rated (Acclaimed)", "Most Popular (Trending)"],
            index=0,
            key="explore_sort_selector",
        )

    preset_info = INDUSTRY_PRESETS.get(chosen_preset, {})
    sort_query = "vote_average.desc" if "Highest" in sort_mode else "popularity.desc"

    if get_tmdb_api_key():
        with interesting_loading(f"Loading acclaimed films for {chosen_preset}..."):
            disc_res = discover_movies_global(
                origin_country=preset_info.get("country", ""),
                original_language=preset_info.get("language", ""),
                page=1,
                sort_by=sort_query,
            )
            raw_movies = disc_res.get("results", []) if disc_res.get("success") else []
    else:
        raw_movies = []

    if raw_movies:
        st.markdown(f"**Acclaimed Titles in {chosen_preset}:**")
        e_cols = st.columns(4)
        for e_idx, e_movie in enumerate(raw_movies[:8]):
            col = e_cols[e_idx % 4]
            with col:
                e_title = e_movie.get("title", "")
                e_year = e_movie.get("release_year") or "N/A"
                e_tmdb_id = str(e_movie.get("tmdb_id", ""))
                e_genres = e_movie.get("genre_names", [])
                e_poster = get_cached_poster(e_tmdb_id, e_title, e_year if isinstance(e_year, int) else None, e_genres)
                first_g = e_genres[0] if e_genres else "Cinema"

                e_encoded = urllib.parse.quote(str(e_title))
                e_modal_url = f"?movie_modal={e_tmdb_id}&movie_title={e_encoded}"

                st.markdown(
                    f"""
                    <a href="{e_modal_url}" target="_self" class="aura-card-anchor" title="Click to view Cast, Ratings, Trailer & Details for {html.escape(e_title)}">
                        <div class="aura-movie-card">
                            <img src="{e_poster}" class="aura-poster-bg" alt="{html.escape(e_title)}" />
                            <div class="aura-card-vignette"></div>
                            <div class="aura-card-click-banner">
                                <span>▶ Info & Trailer</span>
                            </div>
                            <div class="aura-card-content">
                                <div class="aura-card-title">{html.escape(e_title)}</div>
                                <div class="aura-card-meta">
                                    <span class="aura-score-pill">★ {e_movie.get('vote_average', 0.0)}</span>
                                    <span class="aura-genre-pill">{html.escape(str(e_year))}</span>
                                    <span class="aura-genre-pill">{html.escape(first_g)}</span>
                                </div>
                            </div>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
                eb_c1, eb_c2, eb_c3 = st.columns([1.1, 1.0, 1.0])
                with eb_c1:
                    if st.button("🎬 Info", key=f"exp_info_{e_idx}_{e_tmdb_id}", help="View Cast, Ratings, Trailer & Where to watch", use_container_width=True):
                        st.session_state["active_modal_req"] = {
                            "tmdb_id": e_tmdb_id,
                            "title": e_title,
                        }
                        st.rerun()
                with eb_c2:
                    if st.button("Similar ↗", key=f"exp_sim_{e_idx}_{e_tmdb_id}", use_container_width=True):
                        request_movie_switch(e_title)
                with eb_c3:
                    is_in_wl = any(w.get("title") == e_title for w in st.session_state.get("watchlist", []))
                    wl_label = "Saved ✓" if is_in_wl else "+ List"
                    if st.button(wl_label, key=f"exp_wl_{e_idx}_{e_tmdb_id}", use_container_width=True):
                        if is_in_wl:
                            st.session_state["watchlist"] = [w for w in st.session_state["watchlist"] if w.get("title") != e_title]
                            st.toast(f"Removed {e_title} from Watchlist")
                        else:
                            st.session_state["watchlist"].append({
                                "title": e_title,
                                "year": e_year,
                                "tmdb_id": e_tmdb_id,
                                "genres": e_genres,
                                "poster_url": e_poster,
                            })
                            st.toast(f"Added {e_title} to Watchlist!")
                        st.rerun()


# -----------------------------------------------------------------------------
# PAGE 4: WATCHLIST
# -----------------------------------------------------------------------------
elif active_page == "Watchlist":
    st.markdown(
        """
        <div class="cm-hero">
            <div class="cm-hero-kicker">Personal Collection (Session-Only)</div>
            <h1 class="cm-hero-title">Your Watchlist</h1>
            <div class="cm-hero-subtitle">
                Films you've saved to watch next during this browser session. Notice: To persist your selections across visits, use the Export JSON option below.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    watchlist = st.session_state.get("watchlist", [])
    if not watchlist:
        st.markdown(
            """
            <div style="background: #11161D; border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 16px; padding: 3rem 1.5rem; text-align: center; color: #8B98A5;">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🔖</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.35rem;">Your Watchlist is empty</div>
                <div style="font-size: 0.88rem; max-width: 440px; margin: 0 auto 1.25rem auto; color: #8B98A5;">
                    Add movies to your watchlist on Discover or Find Similar to keep track of films you want to watch. (Stored for the duration of this browser session).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        wl_top1, wl_top2, wl_top3 = st.columns([2.0, 1.2, 1.0], vertical_alignment="center")
        with wl_top1:
            st.markdown(f"**Saved Titles ({len(watchlist)} films — Current Session):**")
        with wl_top2:
            st.download_button(
                "📥 Export Watchlist (JSON)",
                data=json.dumps(watchlist, indent=2),
                file_name="cinematch_watchlist.json",
                mime="application/json",
                use_container_width=True,
            )
        with wl_top3:
            if st.button("Clear All ✕", key="clear_watchlist_btn", use_container_width=True):
                st.session_state["watchlist"] = []
                st.toast("Watchlist cleared")
                st.rerun()

        w_cols = st.columns(3)
        for w_idx, w_item in enumerate(watchlist):
            col = w_cols[w_idx % 3]
            with col:
                w_title = w_item.get("title", "")
                w_year = w_item.get("year", "N/A")
                w_tmdb_id = str(w_item.get("tmdb_id", ""))
                w_genres = w_item.get("genres", [])
                w_poster = w_item.get("poster_url") or get_cached_poster(w_tmdb_id, w_title, None, w_genres)
                first_g = w_genres[0] if w_genres else "Cinema"

                w_encoded = urllib.parse.quote(str(w_title))
                w_modal_url = f"?movie_modal={w_tmdb_id}&movie_title={w_encoded}"

                st.markdown(
                    f"""
                    <a href="{w_modal_url}" target="_self" class="aura-card-anchor" title="Click to view Cast, Ratings, Trailer & Details for {html.escape(w_title)}">
                        <div class="aura-movie-card">
                            <img src="{w_poster}" class="aura-poster-bg" alt="{html.escape(w_title)}" />
                            <div class="aura-card-vignette"></div>
                            <div class="aura-card-click-banner">
                                <span>▶ Info & Trailer</span>
                            </div>
                            <div class="aura-card-content">
                                <div class="aura-card-title">{html.escape(w_title)}</div>
                                <div class="aura-card-meta">
                                    <span class="aura-score-pill">Saved 🔖</span>
                                    <span class="aura-genre-pill">{html.escape(str(w_year))}</span>
                                    <span class="aura-genre-pill">{html.escape(first_g)}</span>
                                </div>
                            </div>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
                b1, b2, b3 = st.columns([1.1, 1.0, 1.0])
                with b1:
                    if st.button("🎬 Info", key=f"wl_info_{w_idx}_{w_tmdb_id}", help="View Cast, Ratings, Trailer & Where to watch", use_container_width=True):
                        st.session_state["active_modal_req"] = {
                            "tmdb_id": w_tmdb_id,
                            "title": w_title,
                        }
                        st.rerun()
                with b2:
                    if st.button("Similar ↗", key=f"wl_sim_{w_idx}", use_container_width=True):
                        request_movie_switch(w_title)
                with b3:
                    if st.button("Remove ✕", key=f"wl_del_{w_idx}", use_container_width=True):
                        st.session_state["watchlist"] = [w for w in watchlist if w.get("title") != w_title]
                        st.rerun()


# -----------------------------------------------------------------------------
# PAGE 5: HISTORY
# -----------------------------------------------------------------------------
elif active_page == "History":
    st.markdown(
        """
        <div class="cm-hero">
            <div class="cm-hero-kicker">Exploration Log (Session-Only)</div>
            <h1 class="cm-hero-title">Recently Explored Movies</h1>
            <div class="cm-hero-subtitle">
                A timeline of films explored during this session. Note: Session history is cleared when the browser tab is closed.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state.get("history", [])
    if not history:
        st.info("No exploration history recorded yet in this session. Explore movies on Discover or Find Similar to see them here.")
    else:
        h_top1, h_top2, h_top3 = st.columns([2.0, 1.2, 1.0], vertical_alignment="center")
        with h_top1:
            st.markdown(f"**Recently Viewed ({len(history)} titles — Current Session):**")
        with h_top2:
            st.download_button(
                "📥 Export History (JSON)",
                data=json.dumps(history, indent=2),
                file_name="cinematch_history.json",
                mime="application/json",
                use_container_width=True,
            )
        with h_top3:
            if st.button("Clear Log ✕", key="clear_history_btn", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()

        h_cols = st.columns(4)
        for h_idx, h_item in enumerate(history[:8]):
            col = h_cols[h_idx % 4]
            with col:
                h_title = h_item.get("title", "")
                h_year = h_item.get("year") or "N/A"
                h_tmdb_id = str(h_item.get("tmdb_id", ""))
                h_poster = get_cached_poster(h_tmdb_id, h_title, h_year if isinstance(h_year, int) else None, None)

                h_encoded = urllib.parse.quote(str(h_title))
                h_modal_url = f"?movie_modal={h_tmdb_id}&movie_title={h_encoded}"

                st.markdown(
                    f"""
                    <a href="{h_modal_url}" target="_self" class="aura-card-anchor" title="Click to view Cast, Ratings, Trailer & Details for {html.escape(h_title)}">
                        <div class="aura-movie-card">
                            <img src="{h_poster}" class="aura-poster-bg" alt="{html.escape(h_title)}" />
                            <div class="aura-card-vignette"></div>
                            <div class="aura-card-click-banner">
                                <span>▶ Info & Trailer</span>
                            </div>
                            <div class="aura-card-content">
                                <div class="aura-card-title">{html.escape(h_title)}</div>
                                <div class="aura-card-meta">
                                    <span class="aura-score-pill">Explored</span>
                                    <span class="aura-genre-pill">{html.escape(str(h_year))}</span>
                                </div>
                            </div>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
                hb1, hb2 = st.columns([1.1, 1.0])
                with hb1:
                    if st.button("🎬 Info", key=f"hist_info_{h_idx}_{h_tmdb_id}", help="View Cast, Ratings, Trailer & Where to watch", use_container_width=True):
                        st.session_state["active_modal_req"] = {
                            "tmdb_id": h_tmdb_id,
                            "title": h_title,
                        }
                        st.rerun()
                with hb2:
                    if st.button("Revisit ↺", key=f"hist_revisit_{h_idx}", use_container_width=True):
                        request_movie_switch(h_title)




# =============================================================================
# FOOTER
# =============================================================================
render_footer()
