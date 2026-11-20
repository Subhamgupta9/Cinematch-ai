"""
CineMatch AI - Movie Details & Media Spotlight Modal
=====================================================
Renders a rich, production-grade overlay modal with complete movie metadata:
- Ratings (TMDb score, review counts, personalized match score)
- Cast & Crew (Actors with character names & photos, Directors, Screenwriters)
- Where to Watch in India (Streaming platforms: Netflix, Prime Video, Hotstar, JioCinema, Zee5)
- Official Trailer (Embedded high-res YouTube player + direct link)
- Storyline, Runtime, Genres, Budget, Box Office, and Quick Actions
"""

import html
import urllib.parse
from typing import Optional, Dict, Any, Callable
import streamlit as st

from tmdb_client import (
    fetch_rich_movie_details,
    generate_fallback_poster_svg,
)
from ui.components import interesting_loading


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_details_memo(tmdb_id: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
    """Internal memoized details fetcher."""
    return fetch_rich_movie_details(tmdb_id=tmdb_id, title=title)


def get_cached_rich_details(tmdb_id: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
    """Cached wrapper that auto-heals stale fallback cache entries with live fetch."""
    cached = _fetch_details_memo(tmdb_id=tmdb_id, title=title)
    if not cached.get("success") or not cached.get("poster_url") or "currently being updated" in cached.get("overview", ""):
        live = fetch_rich_movie_details(tmdb_id=tmdb_id, title=title)
        if live.get("poster_url") or live.get("overview") != "Detailed overview is currently being updated for this title.":
            return live
    return cached


@st.dialog("🎬 CineMatch Movie Spotlight", width="large")
def render_movie_details_modal(
    tmdb_id: Optional[str] = None,
    title: Optional[str] = None,
    match_score: Optional[int] = None,
    on_similar_click: Optional[Callable[[str], None]] = None,
):
    """
    Renders an elegant, comprehensive movie spotlight modal.
    Accessible, dark-cinematic themed, with full media and streaming info.
    """
    st.markdown(
        """
        <style>
        div[data-testid="stDialog"] > div, div[role="dialog"] {
            background-color: #0A0E14 !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(0, 242, 152, 0.4) !important;
            border-radius: 18px !important;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.9) !important;
        }
        div[data-testid="stDialog"] h2, div[data-testid="stDialog"] h3, div[data-testid="stDialog"] p {
            color: #FFFFFF !important;
        }
        div[data-testid="stDialog"] button[aria-label="Close"] {
            color: #CBD5E1 !important;
        }
        div[data-testid="stDialog"] button[aria-label="Close"]:hover {
            color: #00F298 !important;
        }
        div[data-testid="stDialog"] [data-baseweb="tab-list"] {
            background-color: transparent !important;
        }
        div[data-testid="stDialog"] button[data-baseweb="tab"] {
            color: #94A3B8 !important;
        }
        div[data-testid="stDialog"] button[data-baseweb="tab"][aria-selected="true"] {
            color: #00F298 !important;
            border-bottom-color: #00F298 !important;
        }
        div[data-testid="stDialog"] a[data-testid="stLinkButton"],
        div[data-testid="stDialog"] [data-testid="stLinkButton"] > a {
            background-color: #111827 !important;
            color: #00F298 !important;
            border: 1px solid rgba(0, 242, 152, 0.4) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            text-decoration: none !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stDialog"] a[data-testid="stLinkButton"]:hover,
        div[data-testid="stDialog"] [data-testid="stLinkButton"] > a:hover {
            background-color: rgba(0, 242, 152, 0.12) !important;
            border-color: #00F298 !important;
            color: #FFFFFF !important;
        }
        div[data-testid="stDialog"] div[data-testid="stAlert"] {
            background-color: #111827 !important;
            color: #CBD5E1 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    disp_title = title or "movie spotlight"
    with interesting_loading(f"Loading official cast, trailer, ratings & streaming for '{disp_title}'..."):
        details = get_cached_rich_details(tmdb_id=tmdb_id, title=title)

    m_title = details.get("title") or title or "Movie Details"
    m_year = details.get("year") or ""
    m_tagline = details.get("tagline") or ""
    m_overview = details.get("overview") or "Overview is currently unavailable."
    m_runtime = details.get("runtime_str") or ""
    m_rating = details.get("vote_average") or 0.0
    m_votes = details.get("vote_count") or 0
    m_genres = details.get("genres") or []
    m_directors = details.get("directors") or []
    m_writers = details.get("writers") or []
    m_cast = details.get("cast") or []
    m_trailer_url = details.get("trailer_url")
    m_trailer_name = details.get("trailer_name") or "Official Trailer"
    m_streaming = details.get("streaming") or []
    m_rent_buy = details.get("rent_buy") or []
    m_justwatch = details.get("justwatch_link")
    m_poster = details.get("poster_url")
    m_backdrop = details.get("backdrop_url")
    m_budget = details.get("budget_str")
    m_revenue = details.get("revenue_str")
    resolved_id = details.get("tmdb_id") or tmdb_id

    # Fallback poster if official poster missing
    if not m_poster:
        m_poster = generate_fallback_poster_svg(m_title, int(m_year) if m_year.isdigit() else None, m_genres)

    # -------------------------------------------------------------------------
    # HEADER BANNER & HERO
    # -------------------------------------------------------------------------
    badge_score = f'<span style="background: rgba(0, 242, 152, 0.15); color: #00F298; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; border: 1px solid rgba(0, 242, 152, 0.3);">{match_score}% Match</span>' if match_score else ''
    badge_rating = f'<span style="background: rgba(251, 191, 36, 0.15); color: #FBBF24; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; border: 1px solid rgba(251, 191, 36, 0.3);">⭐ {m_rating:.1f}/10 ({m_votes:,} reviews)</span>' if m_votes > 0 else '<span style="background: rgba(255, 255, 255, 0.08); color: #94A3B8; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem;">⭐ Acclaimed</span>'
    badge_runtime = f'<span style="background: rgba(255, 255, 255, 0.08); color: #E2E8F0; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600;">⏱ {html.escape(m_runtime)}</span>' if m_runtime else ''
    genre_badges = "".join([f'<span style="background: #1E293B; color: #CBD5E1; padding: 3px 9px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; margin-right: 4px;">{html.escape(g)}</span>' for g in m_genres])

    col_poster, col_info = st.columns([1.0, 2.2], gap="medium")
    with col_poster:
        st.image(m_poster, use_container_width=True)
    with col_info:
        st.markdown(
            f'<div style="font-size: 0.75rem; font-weight: 700; color: #00F298; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.25rem;">✦ CINEMATCH SPOTLIGHT</div>',
            unsafe_allow_html=True,
        )
        year_span = f'<span style="font-size: 1.2rem; color: #94A3B8; font-weight: 500;">({html.escape(str(m_year))})</span>' if m_year else ''
        st.markdown(
            f'<h2 style="font-size: 1.85rem; font-weight: 800; color: #FFFFFF; margin: 0 0 0.25rem 0; line-height: 1.2;">{html.escape(m_title)} {year_span}</h2>',
            unsafe_allow_html=True,
        )
        if m_tagline:
            st.markdown(
                f'<div style="font-style: italic; color: #38BDF8; font-size: 0.9rem; margin-bottom: 0.65rem;">"{html.escape(m_tagline)}"</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.75rem; align-items: center;">{badge_score} {badge_rating} {badge_runtime}</div>',
            unsafe_allow_html=True,
        )
        if genre_badges:
            st.markdown(
                f'<div style="display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.85rem;">{genre_badges}</div>',
                unsafe_allow_html=True,
            )

        # Overview & Storyline placed directly in the top spotlight space
        st.markdown(
            f"""
            <div style="margin-top: 0.5rem;">
                <div style="font-size: 0.74rem; font-weight: 700; color: #00F298; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;">
                    📖 OVERVIEW & STORYLINE
                </div>
                <div style="font-size: 0.92rem; color: #E2E8F0; line-height: 1.65; max-height: 190px; overflow-y: auto; background: rgba(15, 23, 42, 0.7); padding: 0.85rem 1rem; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.08);">
                    {html.escape(m_overview)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Key Production & Crew Credits in Spotlight
        credits_items = []
        if m_directors:
            credits_items.append(f'<span style="color: #94A3B8;">Director:</span> <b style="color: #FFFFFF;">{html.escape(", ".join(m_directors[:2]))}</b>')
        if m_writers:
            credits_items.append(f'<span style="color: #94A3B8;">Writer:</span> <b style="color: #FFFFFF;">{html.escape(", ".join(m_writers[:2]))}</b>')
        if m_budget:
            credits_items.append(f'<span style="color: #94A3B8;">Budget:</span> <b style="color: #00F298;">{html.escape(m_budget)}</b>')
        if m_revenue:
            credits_items.append(f'<span style="color: #94A3B8;">Box Office:</span> <b style="color: #00F298;">{html.escape(m_revenue)}</b>')

        if credits_items:
            st.markdown(
                f'<div style="font-size: 0.82rem; margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 1rem; color: #CBD5E1;">{" · ".join(credits_items)}</div>',
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # TABBED / SECTIONAL CONTENT (Trailer is now Tab 1)
    # -------------------------------------------------------------------------
    tab_trailer, tab_watch, tab_cast = st.tabs([
        "▶ Official Trailer",
        "📺 Where to Watch (India)",
        "🎭 Cast & Crew",
    ])

    # TAB 2: OFFICIAL TRAILER
    with tab_trailer:
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if m_trailer_url and "youtube.com/watch?v=" in m_trailer_url:
            st.markdown(f"#### 🎬 {html.escape(m_trailer_name)}")
            st.video(m_trailer_url)
            st.link_button("▶ Watch on YouTube ↗", m_trailer_url, use_container_width=True)
        else:
            yt_search = f"https://www.youtube.com/results?search_query={urllib.parse.quote(m_title + ' ' + str(m_year) + ' official trailer')}"
            st.info(f"Official embedded player preview is not directly linked in TMDb for this title. You can watch verified trailers directly on YouTube:")
            st.link_button("🔍 Search & Watch Trailer on YouTube ↗", yt_search, type="primary", use_container_width=True)

    # TAB 3: WHERE TO WATCH IN INDIA (JUSTWATCH)
    with tab_watch:
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if m_streaming or m_rent_buy:
            if m_streaming:
                st.markdown("##### 🟢 Subscription Streaming (India)")
                p_cols = st.columns(min(4, max(1, len(m_streaming))))
                for p_idx, prov in enumerate(m_streaming[:4]):
                    with p_cols[p_idx % len(p_cols)]:
                        st.markdown(
                            f'<div style="background: #111827; border: 1px solid rgba(0, 242, 152, 0.3); border-radius: 8px; padding: 0.6rem; text-align: center;"><div style="font-size: 0.85rem; font-weight: 700; color: #FFFFFF;">▶ {html.escape(prov.get("provider_name", ""))}</div><div style="font-size: 0.72rem; color: #00F298;">Included in Sub</div></div>',
                            unsafe_allow_html=True,
                        )

            if m_rent_buy:
                st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                st.markdown("##### 💳 Digital Rent / Buy Options")
                rb_badges = " ".join([
                    f'<span style="background: #1F2937; color: #E5E7EB; padding: 5px 12px; border-radius: 6px; font-size: 0.8rem; display: inline-block; margin: 3px;">💳 {html.escape(rb.get("provider_name", ""))}</span>'
                    for rb in m_rent_buy[:6]
                ])
                st.markdown(rb_badges, unsafe_allow_html=True)

            if m_justwatch:
                st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
                st.link_button("Check Real-Time Availability on JustWatch India ↗", m_justwatch, use_container_width=True)

            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: flex-end; gap: 0.4rem; margin-top: 0.6rem; padding-right: 0.25rem;">
                    <span style="font-size: 0.72rem; color: #94A3B8;">Streaming data powered by</span>
                    <span style="font-weight: 800; font-size: 0.78rem; color: #F5C518; letter-spacing: 0.5px; background: rgba(245, 197, 24, 0.1); padding: 2px 6px; border-radius: 4px;">JustWatch</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Streaming availability data is currently not reported by JustWatch for this title in India. Availability may vary across local theatrical, digital rental, and streaming platforms.")
            st.link_button(
                "🔍 Search Where to Stream on Google ↗",
                f"https://www.google.com/search?q={urllib.parse.quote('where to watch ' + m_title + ' ' + str(m_year) + ' in India')}",
                use_container_width=True,
            )

    # TAB 4: CAST & CREW
    with tab_cast:
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if m_cast:
            st.markdown("##### 🌟 Starring Actors & Characters")
            c_cols = st.columns(2)
            for c_idx, actor in enumerate(m_cast[:8]):
                with c_cols[c_idx % 2]:
                    a_name = actor.get("name", "Unknown")
                    a_char = actor.get("character", "")
                    char_html = f'<div style="color: #94A3B8; font-size: 0.78rem;">as {html.escape(a_char)}</div>' if a_char else ''
                    st.markdown(
                        f'<div style="background: #0F172A; border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.5rem; border: 1px solid rgba(255, 255, 255, 0.05);"><div style="font-weight: 700; color: #FFFFFF; font-size: 0.9rem;">{html.escape(a_name)}</div>{char_html}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Cast metadata is currently indexing for this title.")

    # -------------------------------------------------------------------------
    # MODAL FOOTER ACTIONS
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 1.25rem; border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 1rem;'></div>", unsafe_allow_html=True)
    act_col1, act_col2 = st.columns(2)

    with act_col1:
        if st.button("🎯 Find Movies Similar to This ↗", key="modal_find_sim_btn", type="primary", use_container_width=True):
            if on_similar_click:
                on_similar_click(m_title)
            else:
                st.session_state["target_query"] = m_title
                st.rerun()

    with act_col2:
        is_in_wl = any(w.get("title") == m_title for w in st.session_state.get("watchlist", []))
        wl_btn_label = "Saved in Watchlist ✓" if is_in_wl else "🔖 Add to Watchlist"
        if st.button(wl_btn_label, key="modal_toggle_watchlist_btn", use_container_width=True):
            if is_in_wl:
                st.session_state["watchlist"] = [w for w in st.session_state["watchlist"] if w.get("title") != m_title]
                st.toast(f"Removed {m_title} from Watchlist")
            else:
                st.session_state["watchlist"].append({
                    "title": m_title,
                    "year": m_year,
                    "tmdb_id": resolved_id,
                    "genres": m_genres,
                    "poster_url": m_poster,
                })
                st.toast(f"Added {m_title} to Watchlist!")
            st.rerun()
