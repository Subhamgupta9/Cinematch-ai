"""
CineMatch AI - UI Presentation Layer
====================================
Modular UI architecture adhering to Minimal Design tokens and WCAG 2.2 AA.
"""

from .styles import MINIMAL_THEME_CSS, DESIGN_TOKENS
from .components import render_navbar, render_footer, get_active_page, set_active_page

__all__ = [
    "MINIMAL_THEME_CSS",
    "DESIGN_TOKENS",
    "render_navbar",
    "render_footer",
    "get_active_page",
    "set_active_page",
]
