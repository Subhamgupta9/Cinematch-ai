"""
CineMatch AI - Reusable UI Components
=====================================
Modular components including AuraScope Top Navigation, Footer, and Page State Managers.
Adheres strictly to AuraScope Dark Cinematic tokens and WCAG 2.2 AA.
"""

from typing import Optional, List, Dict, Any
import html
import streamlit as st

import random
from contextlib import contextmanager

DESTINATIONS = [
    "Discover",
    "Find Similar",
    "Explore",
    "Watchlist",
    "History",
]

PAGE_ICONS = {
    "Discover": "✦",
    "Find Similar": "🔍",
    "Explore": "🧭",
    "Watchlist": "🔖",
    "History": "📜",
}

CINEMATIC_LORE_AND_QUOTES = [
    ("Scanning 14,204 cinematic archives across global & Indian cinema...", "✦ CineMatch Neural Core"),
    ("Consulting Christopher Nolan's dream architecture...", "Inception (2010)"),
    ("All Izz Well! Calibrating comedy and life philosophy...", "3 Idiots (2009)"),
    ("Why so serious? Extracting masterwork storytelling frequency...", "The Dark Knight (2008)"),
    ("Do not go gentle into that good night... calculating cosmic affinity...", "Interstellar (2014)"),
    ("Finding the golden ratio of direction, screenplay, and performance...", "CineMatch AI"),
    ("Curating Bollywood, South Indian cinema & international masterpieces...", "Indian & World Discovery"),
    ("Brewing fresh cinematic popcorn while neural vectors align...", "Cinema Lounge"),
    ("Extracting emotional resonance and narrative pacing...", "AuraScope AI"),
    ("Whatever it takes... assembling the ultimate recommendations...", "Avengers: Endgame (2019)"),
]


def get_random_cinematic_lore() -> tuple[str, str]:
    """Returns a random engaging movie thought and film source."""
    return random.choice(CINEMATIC_LORE_AND_QUOTES)


@contextmanager
def interesting_loading(custom_msg: Optional[str] = None):
    """
    Renders an interesting, animated cinematic loading experience with
    pulsing neon glows and rotating cinema lore and quotes.
    """
    quote_text, quote_source = get_random_cinematic_lore()
    display_title = custom_msg or quote_text

    loader_placeholder = st.empty()
    loader_placeholder.markdown(
        f"""
        <div class="cm-interesting-loader">
            <div class="cm-loader-radar">
                <div class="cm-loader-ring"></div>
                <div class="cm-loader-core">✦</div>
            </div>
            <div class="cm-loader-text-box">
                <div class="cm-loader-msg">{html.escape(display_title)}</div>
                <div class="cm-loader-quote">"{html.escape(quote_text)}" — <span style="color: #00F298;">{html.escape(quote_source)}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        loader_placeholder.empty()



def init_session_state():
    """Initializes default session state variables for navigation, watchlist, and preferences."""
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Discover"
    if "discover_mode" not in st.session_state:
        st.session_state["discover_mode"] = "✨ By AI Taste"
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = []
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "preferences" not in st.session_state:
        st.session_state["preferences"] = {
            "preferred_languages": ["Hindi", "English", "Telugu", "Tamil"],
            "preferred_industries": ["Bollywood", "Tollywood", "Kollywood", "All Cinema"],
            "exclude_watched": False,
            "default_top_n": 8,
            "min_rating": 0.0,
        }
    if "active_modifier" not in st.session_state:
        st.session_state["active_modifier"] = "Ethereal Sci-Fi"
    if "dev_mode" not in st.session_state:
        dev_param = st.query_params.get("dev", "false")
        st.session_state["dev_mode"] = dev_param.lower() in ("true", "1")


def get_active_page() -> str:
    """Returns the current active destination."""
    init_session_state()
    return st.session_state.get("current_page", "Discover")


def set_active_page(page_name: str):
    """Sets the active destination and triggers rerun."""
    if page_name in DESTINATIONS:
        st.session_state["current_page"] = page_name
        st.rerun()


def render_navbar() -> str:
    """
    Renders the clean, cinematic top navigation bar across all screens.
    Matches the AuraScope brand identity with glowing mint accent.
    """
    init_session_state()
    current = st.session_state.get("current_page", "Discover")

    nav_col1, nav_col2 = st.columns([1.3, 2.7], vertical_alignment="center")

    with nav_col1:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.35rem 0;">
                <div style="display: flex; align-items: center; justify-content: center; width: 34px; height: 34px; background: linear-gradient(135deg, #00F298 0%, #00B371 100%); color: #070A0E; border-radius: 50%; font-size: 1rem; font-weight: 800; box-shadow: 0 0 16px rgba(0, 242, 152, 0.45);">
                    ✦
                </div>
                <div>
                    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.03em; line-height: 1.1;">
                        CineMatch <span style="color: #00F298; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; background: rgba(0, 242, 152, 0.12); padding: 0.15rem 0.45rem; border-radius: 9999px; border: 1px solid rgba(0, 242, 152, 0.3);">AI</span>
                    </div>
                    <div style="font-size: 0.74rem; color: #8B98A5; font-weight: 500; letter-spacing: 0.01em;">
                        Intelligent Movie Discovery & Frequency Engine
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav_col2:
        options_with_icons = [f"{PAGE_ICONS[d]} {d}" for d in DESTINATIONS]
        default_index = DESTINATIONS.index(current) if current in DESTINATIONS else 0

        selected = st.segmented_control(
            "Navigation Menu",
            options=options_with_icons,
            default=options_with_icons[default_index],
            key="top_navbar_segmented_control",
            label_visibility="collapsed",
        )

        if selected:
            chosen_dest = selected.split(" ", 1)[-1]
            if chosen_dest != current and chosen_dest in DESTINATIONS:
                st.session_state["current_page"] = chosen_dest
                st.rerun()

    st.markdown(
        "<div style='border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-top: 0.6rem; margin-bottom: 1.8rem;'></div>",
        unsafe_allow_html=True,
    )
    return st.session_state.get("current_page", "Discover")


def render_footer():
    """Renders the dark cinematic site footer."""
    st.markdown(
        """
        <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 2.5rem; margin-top: 4rem; text-align: center; color: #586674; font-size: 0.82rem; line-height: 1.7;">
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.92rem; margin-bottom: 0.35rem;">
                CineMatch AI — Intelligent Movie Discovery & Recommendation
            </div>
            <div>
                Data sourced via <a href="https://grouplens.org/datasets/movielens/" target="_blank" rel="noopener noreferrer" style="color: #00F298; text-decoration: none;">MovieLens</a> 
                and <a href="https://www.themoviedb.org/" target="_blank" rel="noopener noreferrer" style="color: #00F298; text-decoration: none;">The Movie Database (TMDb) API</a>.
                Streaming availability supplied via JustWatch.
            </div>
            <div style="font-size: 0.75rem; color: #586674; margin-top: 0.4rem;">
                🟢 14,204 Titles Cataloged · Hybrid Content & Collaborative Matrix Online
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
