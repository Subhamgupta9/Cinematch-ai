"""
CineMatch AI - AuraScope Dark Cinematic Theme & CSS Architecture
================================================================
Inspired by the AuraScope cinematic frequency aesthetic.
Deep space dark canvas (#070A0E / #0B0F15), glassmorphic surfaces,
neon mint accents (#00F298), crisp display typography, and 2:3 vertical poster cards.
Replaces the developer sidebar with a full-width (1280px) responsive layout.
"""

DESIGN_TOKENS = {
    "bg_main": "#070A0E",
    "bg_surface": "#11161D",
    "bg_surface_glass": "rgba(17, 22, 29, 0.82)",
    "bg_subtle": "#171E28",
    "bg_card": "#121820",
    "text_primary": "#FFFFFF",
    "text_secondary": "#8B98A5",
    "text_muted": "#586674",
    "border_color": "rgba(255, 255, 255, 0.08)",
    "border_strong": "rgba(255, 255, 255, 0.16)",
    "accent_mint": "#00F298",
    "accent_mint_glow": "rgba(0, 242, 152, 0.25)",
    "accent_cyan": "#00D2FF",
    "accent_violet": "#8B5CF6",
    "accent_hover": "#00DC8A",
    "success": "#00F298",
    "error": "#FF4D4D",
    "max_width": "1360px",
    "radius_sm": "8px",
    "radius_md": "14px",
    "radius_lg": "22px",
    "radius_pill": "9999px",
    "shadow_sm": "0 2px 8px rgba(0, 0, 0, 0.35)",
    "shadow_card": "0 8px 24px rgba(0, 0, 0, 0.55)",
    "shadow_hover": "0 14px 36px rgba(0, 242, 152, 0.18)",
}

MINIMAL_THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {{
    --cm-bg: {DESIGN_TOKENS['bg_main']};
    --cm-surface: {DESIGN_TOKENS['bg_surface']};
    --cm-surface-glass: {DESIGN_TOKENS['bg_surface_glass']};
    --cm-subtle: {DESIGN_TOKENS['bg_subtle']};
    --cm-card: {DESIGN_TOKENS['bg_card']};
    --cm-text: {DESIGN_TOKENS['text_primary']};
    --cm-text-sec: {DESIGN_TOKENS['text_secondary']};
    --cm-text-muted: {DESIGN_TOKENS['text_muted']};
    --cm-border: {DESIGN_TOKENS['border_color']};
    --cm-border-strong: {DESIGN_TOKENS['border_strong']};
    --cm-accent: {DESIGN_TOKENS['accent_mint']};
    --cm-accent-glow: {DESIGN_TOKENS['accent_mint_glow']};
    --cm-cyan: {DESIGN_TOKENS['accent_cyan']};
    --cm-violet: {DESIGN_TOKENS['accent_violet']};
    --cm-success: {DESIGN_TOKENS['success']};
    --cm-error: {DESIGN_TOKENS['error']};
    --cm-radius-sm: {DESIGN_TOKENS['radius_sm']};
    --cm-radius-md: {DESIGN_TOKENS['radius_md']};
    --cm-radius-lg: {DESIGN_TOKENS['radius_lg']};
    --cm-radius-pill: {DESIGN_TOKENS['radius_pill']};
    --cm-shadow-sm: {DESIGN_TOKENS['shadow_sm']};
    --cm-shadow-card: {DESIGN_TOKENS['shadow_card']};
    --cm-shadow-hover: {DESIGN_TOKENS['shadow_hover']};
    --cm-max-width: {DESIGN_TOKENS['max_width']};
}}

/* Global Reset: Deep cinematic dark space */
html, body, [class*="css"], .stApp {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: var(--cm-bg) !important;
    background: radial-gradient(circle at 10% 20%, rgba(0, 242, 152, 0.03) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.04) 0%, transparent 45%),
                #070A0E !important;
    color: var(--cm-text) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

/* Completely remove the technical sidebar */
section[data-testid="stSidebar"],
div[data-testid="stSidebarNav"],
button[data-testid="stSidebarCollapseButton"],
header [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{
    display: none !important;
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
    visibility: hidden !important;
    pointer-events: none !important;
}}

/* Clean full-width container centered with comfortable margins */
.block-container,
div[data-testid="stMainBlockContainer"],
.stMainBlockContainer {{
    max-width: var(--cm-max-width) !important;
    padding-top: 5.5rem !important;
    padding-bottom: 4rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    margin: 0 auto !important;
}}

/* Minimal dark header */
header[data-testid="stHeader"] {{
    background-color: rgba(7, 10, 14, 0.75) !important;
    backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid var(--cm-border) !important;
    height: 3.5rem !important;
    z-index: 99 !important;
}}

/* Accessible focus visible outlines */
*:focus-visible {{
    outline: 2px solid var(--cm-accent) !important;
    outline-offset: 2px !important;
}}

/* Typography Scale */
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif !important;
    color: var(--cm-text) !important;
    letter-spacing: -0.025em;
    font-weight: 700;
}}

/* Top Navigation Bar */
.cm-navbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 0 1.25rem 0;
    border-bottom: 1px solid var(--cm-border);
    margin-bottom: 2rem;
}}

.cm-brand {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--cm-text);
    text-decoration: none;
    letter-spacing: -0.03em;
}}

.cm-brand-mark {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #00F298 0%, #00B371 100%);
    color: #070A0E;
    border-radius: 50%;
    font-size: 1.1rem;
    font-weight: 800;
    box-shadow: 0 0 16px rgba(0, 242, 152, 0.45);
}}

.cm-brand-accent {{
    color: var(--cm-accent);
    font-weight: 600;
    font-size: 0.8rem;
    margin-left: 0.35rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: rgba(0, 242, 152, 0.1);
    padding: 0.2rem 0.5rem;
    border-radius: var(--cm-radius-pill);
    border: 1px solid rgba(0, 242, 152, 0.25);
}}

/* Navigation Links / Tabs */
.cm-nav-items {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}

.cm-nav-btn {{
    font-size: 0.86rem;
    font-weight: 600;
    color: var(--cm-text-sec);
    padding: 0.45rem 0.95rem;
    border-radius: var(--cm-radius-pill);
    border: 1px solid transparent;
    background: transparent;
    text-decoration: none;
    transition: all 180ms ease;
}}

.cm-nav-btn:hover {{
    color: var(--cm-text);
    background-color: var(--cm-subtle);
    border-color: var(--cm-border);
}}

.cm-nav-btn-active {{
    color: #070A0E !important;
    background-color: var(--cm-accent) !important;
    border-color: var(--cm-accent) !important;
    box-shadow: 0 0 14px var(--cm-accent-glow) !important;
}}

/* AuraScope Left Panel Styling */
.aura-display-title {{
    font-size: 3.4rem !important;
    font-weight: 800 !important;
    line-height: 1.04 !important;
    letter-spacing: -0.04em !important;
    margin: 0 0 1.5rem 0 !important;
}}

.aura-title-white {{
    color: #FFFFFF !important;
    display: block;
}}

.aura-title-muted {{
    color: #586674 !important;
    display: block;
}}

/* Glassmorphic AI Prompt Box */
.aura-prompt-box {{
    background: rgba(18, 24, 32, 0.75);
    backdrop-filter: blur(16px);
    border: 1px solid var(--cm-border-strong);
    border-radius: 20px;
    padding: 1.25rem;
    margin-bottom: 1.75rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    transition: border-color 200ms ease, box-shadow 200ms ease;
}}

.aura-prompt-box:focus-within {{
    border-color: var(--cm-accent);
    box-shadow: 0 0 24px var(--cm-accent-glow);
}}

.aura-sense-label {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--cm-accent);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}

.aura-section-kicker {{
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--cm-text-muted);
    margin-bottom: 0.85rem;
}}

/* Curated Modifier Pills */
.aura-pill-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 2rem;
}}

.aura-modifier-pill {{
    display: inline-flex;
    align-items: center;
    background: #121820;
    border: 1px solid var(--cm-border);
    color: var(--cm-text-sec);
    padding: 0.4rem 0.85rem;
    border-radius: var(--cm-radius-pill);
    font-size: 0.82rem;
    font-weight: 600;
    transition: all 180ms ease;
    cursor: pointer;
    text-decoration: none;
}}

.aura-modifier-pill:hover {{
    color: var(--cm-text);
    background: #171E28;
    border-color: var(--cm-border-strong);
    transform: translateY(-1px);
}}

.aura-modifier-pill.active {{
    color: #070A0E;
    background: var(--cm-accent);
    border-color: var(--cm-accent);
    box-shadow: 0 0 12px var(--cm-accent-glow);
}}

/* Engine Online Status Pill */
.aura-status-bar {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--cm-border);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--cm-text-muted);
}}

.aura-status-dot {{
    width: 8px;
    height: 8px;
    background-color: var(--cm-accent);
    border-radius: 50%;
    box-shadow: 0 0 10px var(--cm-accent);
    display: inline-block;
}}

/* Right Panel: Aligned Frequencies Stage */
.aura-stage-header {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--cm-border);
}}

.aura-stage-title {{
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}}

.aura-stage-subtitle {{
    font-size: 0.88rem;
    color: var(--cm-text-sec);
    margin-top: 0.25rem;
}}

/* Streamlit Native Inputs & Textareas in Dark Mode - High Contrast Text Visibility */
div[class*="elm3lkv1"],
div[data-testid="stTextArea"] div,
div[data-testid="stTextArea"] > div,
.stTextArea > div,
div[class*="stTextArea"] > div,
div[data-baseweb="textarea"],
div[data-baseweb="textarea"] > div,
div[data-baseweb="base-input"],
div[data-baseweb="input"] > div,
div[data-baseweb="input"],
div[data-testid="stTextInput"] > div > div,
div[data-baseweb="select"] > div {{
    background-color: #0E141C !important;
    background: #0E141C !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: var(--cm-radius-md) !important;
    color: #FFFFFF !important;
    box-shadow: none !important;
    transition: all 180ms ease !important;
}}

/* Universal text visibility inside input fields and typing bars */
div[data-testid="stTextInput"] input,
div[data-testid="stTextInput"] input[type="text"],
div[data-baseweb="input"] input,
.stTextInput input,
input[type="text"],
input,
.stTextArea textarea,
div[class*="elm3lkv2"],
textarea {{
    padding: 0.65rem 1rem !important;
    font-size: 0.98rem !important;
    font-weight: 500 !important;
    line-height: 1.5 !important;
    background-color: transparent !important;
    background: transparent !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    caret-color: #00F298 !important;
    opacity: 1 !important;
}}

/* Crisp, easily readable placeholder text */
div[data-testid="stTextInput"] input::placeholder,
div[data-baseweb="input"] input::placeholder,
.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
textarea::placeholder,
input::placeholder {{
    color: #94A3B8 !important;
    -webkit-text-fill-color: #94A3B8 !important;
    font-style: normal !important;
    opacity: 0.85 !important;
}}

/* Active focus glow */
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="input"]:focus-within,
div[data-testid="stTextInput"] > div > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within,
div[class*="elm3lkv1"]:focus-within {{
    border-color: #00F298 !important;
    box-shadow: 0 0 16px rgba(0, 242, 152, 0.35) !important;
}}

/* Interesting Cinematic Loader Component */
.cm-interesting-loader {{
    display: flex;
    align-items: center;
    gap: 1.25rem;
    background: linear-gradient(135deg, rgba(14, 20, 28, 0.95) 0%, rgba(10, 14, 20, 0.95) 100%);
    border: 1px solid rgba(0, 242, 152, 0.35);
    border-radius: 16px;
    padding: 1.25rem 1.75rem;
    margin: 1.5rem 0;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 242, 152, 0.15);
    animation: cm-loader-pulse 2s infinite ease-in-out;
}}

@keyframes cm-loader-pulse {{
    0%, 100% {{ border-color: rgba(0, 242, 152, 0.3); box-shadow: 0 0 15px rgba(0, 242, 152, 0.1); }}
    50% {{ border-color: rgba(0, 242, 152, 0.7); box-shadow: 0 0 25px rgba(0, 242, 152, 0.25); }}
}}

.cm-loader-radar {{
    position: relative;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}}

.cm-loader-ring {{
    position: absolute;
    width: 44px;
    height: 44px;
    border: 2px dashed #00F298;
    border-radius: 50%;
    animation: cm-spin 3s linear infinite;
}}

@keyframes cm-spin {{
    100% {{ transform: rotate(360deg); }}
}}

.cm-loader-core {{
    font-size: 1.1rem;
    font-weight: 800;
    color: #00F298;
    text-shadow: 0 0 10px #00F298;
    animation: cm-core-pulse 1.5s ease-in-out infinite alternate;
}}

@keyframes cm-core-pulse {{
    0% {{ transform: scale(0.85); opacity: 0.7; }}
    100% {{ transform: scale(1.15); opacity: 1; }}
}}

.cm-loader-text-box {{
    flex: 1;
}}

.cm-loader-msg {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.25rem;
    letter-spacing: -0.01em;
}}

.cm-loader-quote {{
    font-size: 0.84rem;
    color: #94A3B8;
    font-style: italic;
    line-height: 1.4;
}}

/* Dark Selectbox & ComboBox Styling (React-Aria & BaseWeb) */
.stSelectbox,
.react-aria-ComboBox,
div[class*="stSelectbox"] {{
    background-color: transparent !important;
}}

.stSelectbox > div,
.react-aria-ComboBox > div,
div[class*="equpb2c0"],
div[data-baseweb="select"] > div {{
    background-color: #11161D !important;
    background: #11161D !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: var(--cm-radius-pill) !important;
    color: #FFFFFF !important;
    transition: all 180ms ease !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4) !important;
}}

.stSelectbox > div:focus-within,
.react-aria-ComboBox > div:focus-within,
div[class*="equpb2c0"]:focus-within,
div[data-baseweb="select"] > div:focus-within {{
    border-color: var(--cm-accent) !important;
    box-shadow: 0 0 14px var(--cm-accent-glow) !important;
}}

.stSelectbox input,
.react-aria-ComboBox input,
input[class*="equpb2c4"],
div[data-baseweb="select"] input {{
    background-color: transparent !important;
    background: transparent !important;
    color: #FFFFFF !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    border: none !important;
}}

.stSelectbox input::placeholder,
.react-aria-ComboBox input::placeholder {{
    color: #8B98A5 !important;
}}

.stSelectbox button,
.react-aria-ComboBox button,
button[class*="equpb2c5"],
div[data-baseweb="select"] svg {{
    background-color: transparent !important;
    background: transparent !important;
    color: #8B98A5 !important;
    border: none !important;
}}

.stSelectbox button svg,
.react-aria-ComboBox button svg,
button[class*="equpb2c5"] svg {{
    fill: #8B98A5 !important;
    color: #8B98A5 !important;
}}

.stSelectbox button:hover svg,
.react-aria-ComboBox button:hover svg {{
    fill: #00F298 !important;
    color: #00F298 !important;
}}

/* Dropdown Menu Popover / Listbox Container */
div[class*="equpb2c8"],
div[role="listbox"],
[data-testid="stSelectboxVirtualDropdown"],
div[class*="react-aria-Popover"],
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[data-baseweb="menu"] {{
    background-color: #11161D !important;
    background: #11161D !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: var(--cm-radius-md) !important;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.7) !important;
    padding: 6px !important;
    z-index: 999999 !important;
}}

/* Dropdown Options */
div[class*="equpb2c10"],
div[role="option"],
li[role="option"],
[data-baseweb="menu"] li {{
    background-color: transparent !important;
    background: transparent !important;
    color: #E2E8F0 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 0.85rem !important;
    border-radius: var(--cm-radius-sm) !important;
    cursor: pointer !important;
    transition: all 120ms ease !important;
}}

div[class*="equpb2c10"]:hover,
div[role="option"]:hover,
li[role="option"]:hover,
div[class*="equpb2c10"][aria-selected="true"],
div[role="option"][aria-selected="true"],
li[role="option"][aria-selected="true"] {{
    background-color: rgba(0, 242, 152, 0.15) !important;
    background: rgba(0, 242, 152, 0.15) !important;
    color: #00F298 !important;
    font-weight: 700 !important;
}}

/* Dark Alert & Notification Banners (st.info, st.warning, st.success) */
div[data-testid="stAlert"],
.stAlert {{
    background-color: transparent !important;
    background: transparent !important;
}}

div[data-testid="stAlertContainer"],
.stAlert > div {{
    background-color: rgba(18, 24, 32, 0.95) !important;
    background: rgba(18, 24, 32, 0.95) !important;
    border: 1px solid rgba(0, 242, 152, 0.3) !important;
    border-radius: var(--cm-radius-md) !important;
    color: #E2E8F0 !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5) !important;
}}

div[data-testid="stAlertContainer"] p,
div[data-testid="stAlertContainer"] span {{
    color: #E2E8F0 !important;
    font-size: 0.9rem !important;
}}

div[data-testid="stAlertContainer"] svg {{
    fill: #00F298 !important;
    color: #00F298 !important;
}}

/* Dark Segmented Control Navigation (.stButtonGroup & .e1aouyeq) */
div.stButtonGroup,
div[class*="stButtonGroup"],
div[class*="e1aouyeq21"],
div[data-testid="stSegmentedControl"],
div[data-testid="stSegmentedControl"] > div {{
    background-color: #11161D !important;
    background: #11161D !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: var(--cm-radius-pill) !important;
    padding: 3px !important;
}}

div.stButtonGroup button,
div[class*="stButtonGroup"] button,
button[class*="e1aouyeq29"],
button[class*="e1aouyeq"],
div[data-testid="stSegmentedControl"] button {{
    background-color: #11161D !important;
    background: #11161D !important;
    color: #8B98A5 !important;
    border: 1px solid transparent !important;
    border-radius: var(--cm-radius-pill) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    transition: all 150ms ease !important;
}}

div.stButtonGroup button:hover,
div[class*="stButtonGroup"] button:hover,
button[class*="e1aouyeq"]:hover,
div[data-testid="stSegmentedControl"] button:hover {{
    color: #FFFFFF !important;
    background-color: #1A222D !important;
    background: #1A222D !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
}}

div.stButtonGroup button[aria-checked="true"],
div.stButtonGroup button[data-checked="true"],
div[class*="stButtonGroup"] button[aria-checked="true"],
button[class*="e1aouyeq"][aria-checked="true"],
div[data-testid="stSegmentedControl"] button[aria-checked="true"] {{
    background: rgba(0, 242, 152, 0.16) !important;
    background-color: rgba(0, 242, 152, 0.16) !important;
    color: #00F298 !important;
    border: 1px solid rgba(0, 242, 152, 0.4) !important;
    font-weight: 700 !important;
    box-shadow: 0 0 12px var(--cm-accent-glow) !important;
}}




/* Streamlit Buttons: Clean Pill Style */
div.stButton > button {{
    background-color: #161D26 !important;
    color: #FFFFFF !important;
    border: 1px solid var(--cm-border-strong) !important;
    border-radius: var(--cm-radius-pill) !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 180ms ease !important;
}}

div.stButton > button:hover {{
    background-color: #1F2834 !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
    color: #FFFFFF !important;
    transform: translateY(-1px);
}}

div.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #00F298 0%, #00C87A 100%) !important;
    color: #070A0E !important;
    border-color: #00F298 !important;
    box-shadow: 0 0 16px var(--cm-accent-glow) !important;
    font-weight: 700 !important;
}}

div.stButton > button[kind="primary"]:hover {{
    background: #00FF9F !important;
    border-color: #00FF9F !important;
    box-shadow: 0 0 24px rgba(0, 242, 152, 0.4) !important;
    color: #070A0E !important;
}}

/* Vertical 2:3 Movie Poster Cards */
.aura-movie-card {{
    background-color: #11161D;
    border: 1px solid var(--cm-border);
    border-radius: 20px;
    overflow: hidden;
    position: relative;
    aspect-ratio: 2 / 3;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    box-shadow: var(--cm-shadow-card);
    transition: all 220ms cubic-bezier(0.16, 1, 0.3, 1);
    text-decoration: none;
    margin-bottom: 1.25rem;
}}

.aura-movie-card:hover {{
    transform: translateY(-6px) scale(1.02);
    border-color: var(--cm-accent);
    box-shadow: 0 16px 36px rgba(0, 242, 152, 0.22);
    cursor: pointer;
}}

.aura-card-anchor {{
    text-decoration: none !important;
    display: block;
    cursor: pointer;
    color: inherit !important;
}}

.aura-card-anchor:hover, .aura-card-anchor:focus {{
    text-decoration: none !important;
    color: inherit !important;
}}

.aura-card-click-banner {{
    position: absolute;
    top: 12px;
    right: 12px;
    z-index: 4;
    background: rgba(7, 10, 14, 0.85);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(0, 242, 152, 0.4);
    color: #00F298;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    gap: 4px;
    opacity: 0.85;
    transition: all 180ms ease;
}}

.aura-movie-card:hover .aura-card-click-banner {{
    opacity: 1;
    background: rgba(0, 242, 152, 0.95);
    color: #070A0E;
    box-shadow: 0 0 16px rgba(0, 242, 152, 0.6);
    transform: scale(1.04);
}}

.aura-poster-bg {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    z-index: 1;
}}

.aura-card-vignette {{
    position: absolute;
    inset: 0;
    background: linear-gradient(to top, rgba(7, 10, 14, 0.95) 0%, rgba(7, 10, 14, 0.6) 40%, transparent 80%);
    z-index: 2;
    transition: opacity 200ms ease;
}}

.aura-movie-card:hover .aura-card-vignette {{
    background: linear-gradient(to top, rgba(7, 10, 14, 0.98) 0%, rgba(7, 10, 14, 0.75) 50%, rgba(0, 242, 152, 0.08) 100%);
}}

.aura-card-content {{
    position: relative;
    z-index: 3;
    padding: 1.1rem;
    color: #FFFFFF;
}}

.aura-card-title {{
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.25;
    margin: 0 0 0.4rem 0;
    color: #FFFFFF;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.8);
}}

.aura-card-meta {{
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.78rem;
    color: var(--cm-text-sec);
    flex-wrap: wrap;
}}

.aura-score-pill {{
    background: rgba(0, 242, 152, 0.15);
    border: 1px solid rgba(0, 242, 152, 0.35);
    color: var(--cm-accent);
    padding: 0.15rem 0.5rem;
    border-radius: var(--cm-radius-pill);
    font-weight: 700;
    font-size: 0.72rem;
}}

.aura-genre-pill {{
    background: rgba(255, 255, 255, 0.1);
    color: #E2E8F0;
    padding: 0.15rem 0.45rem;
    border-radius: var(--cm-radius-pill);
    font-size: 0.72rem;
}}

/* Detailed Movie Spotlight Panel */
.aura-spotlight {{
    background: rgba(18, 24, 32, 0.9);
    border: 1px solid var(--cm-border-strong);
    border-radius: 24px;
    padding: 1.75rem;
    margin: 2rem 0;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6);
}}

/* Streaming Availability Badges */
.aura-watch-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: #171E28;
    border: 1px solid var(--cm-border-strong);
    padding: 0.35rem 0.75rem;
    border-radius: var(--cm-radius-pill);
    font-size: 0.8rem;
    font-weight: 600;
    color: #FFFFFF;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}}

/* Minimal Dark Footer */
.cm-footer {{
    border-top: 1px solid var(--cm-border);
    padding-top: 2.5rem;
    margin-top: 4rem;
    color: var(--cm-text-muted);
    font-size: 0.82rem;
    line-height: 1.7;
    text-align: center;
}}

.cm-footer a {{
    color: var(--cm-accent);
    text-decoration: none;
}}

.cm-footer a:hover {{
    text-decoration: underline;
}}

/* Streamlit Modal Dialog Dark Styling */
[data-testid="stDialog"] {{
    background-color: rgba(3, 5, 8, 0.82) !important;
    backdrop-filter: blur(12px) !important;
}}

[data-testid="stDialog"] > div,
div[role="dialog"] {{
    background: #0B0F15 !important;
    background-color: #0B0F15 !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(0, 242, 152, 0.35) !important;
    border-radius: 20px !important;
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.95) !important;
}}

[data-testid="stDialog"] h1,
[data-testid="stDialog"] h2,
[data-testid="stDialog"] h3,
[data-testid="stDialog"] h4,
[data-testid="stDialog"] p {{
    color: #F8FAFC !important;
}}

[data-testid="stDialog"] button[aria-label="Close"],
div[role="dialog"] button[aria-label="Close"] {{
    color: #94A3B8 !important;
}}

[data-testid="stDialog"] button[aria-label="Close"]:hover,
div[role="dialog"] button[aria-label="Close"]:hover {{
    color: #00F298 !important;
}}

[data-testid="stDialog"] [data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
}}

[data-testid="stDialog"] button[data-baseweb="tab"] {{
    color: #94A3B8 !important;
    font-weight: 600 !important;
}}

[data-testid="stDialog"] button[data-baseweb="tab"][aria-selected="true"] {{
    color: #00F298 !important;
    border-bottom-color: #00F298 !important;
}}

/* Responsive Breakpoints */
@media (max-width: 992px) {{
    .aura-display-title {{
        font-size: 2.6rem !important;
    }}
    .block-container {{
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }}
}}

@media (max-width: 768px) {{
    .aura-display-title {{
        font-size: 2.2rem !important;
    }}
    .cm-navbar {{
        flex-direction: column;
        align-items: flex-start;
        gap: 0.75rem;
    }}
    .cm-nav-items {{
        width: 100%;
        overflow-x: auto;
    }}
}}
</style>
"""
