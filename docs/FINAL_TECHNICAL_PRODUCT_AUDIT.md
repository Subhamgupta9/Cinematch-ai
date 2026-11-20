# CineMatch AI — Final Technical & Product Audit Report

**Audit Date:** November 20, 2026  
**Auditor Roles:** Principal Software Engineer, Senior ML Engineer, QA Architect, Security Reviewer, and Product Auditor  
**Repository Working Directory:** `c:\Projects\Movie Recommondation System`  
**Evaluation Scope:** Complete Project Architecture, Recommendation Pipeline, Metadata & OTT Layer, UI/UX, Security, Performance, and Automated Test Suite.

---

## 1. Executive Summary

CineMatch AI presents itself as a modern, AI-powered movie discovery and recommendation platform designed for Indian and international cinema. It features a polished dark-themed user interface, multimodal movie cards with trailer previews, streaming availability for India, and multiple discovery vectors (By Movie, By Genre, By Rating, By AI Taste).

However, an exhaustive, empirical code-and-runtime audit reveals a **critical divergence between architectural intent and production execution**:

1. **The Core Hybrid ML Architecture is Orphaned:** A sophisticated Stage 3 multi-candidate hybrid engine featuring Item-Item Pearson Collaborative Filtering (`collaborative_recommender.py`), Dynamic TF-IDF vector projection (`recommender.py`), and Bayesian quality scoring (`hybrid_recommender.py`, `candidate_generator.py`) was designed and tested in isolation, but **is never imported, instantiated, or executed in the live Streamlit application (`app.py`, `ui/page_discover.py`)**.
2. **Cosmetic Algorithm Selector (No-Op Control):** The user-facing dropdown offering *"⚡ Hybrid AI"*, *"🧬 Content-Based"*, and *"👥 Collaborative Filtering"* in `ui/page_discover.py` (lines 834–840) is completely non-functional. The selected algorithm state is never passed to or evaluated by `fetch_smart_recommendations()`.
3. **Live Recommendations Depend on Third-Party Heuristics:** In production, recommendation generation bypasses the local collaborative and hybrid models entirely. The UI relies almost exclusively on TMDb's `/movie/{id}/recommendations` endpoint, augmented with regional discover fallbacks and an offline MovieLens TF-IDF engine used only as a last resort.
4. **"AI Taste / Vibe Search" is Naive Keyword Pattern Matching:** Marketed as a natural language semantic taste engine, the feature is implemented as a simple dictionary substring lookup (`KEYWORD_GENRE_MAP` in `ui/page_discover.py`). It lacks vector embeddings, semantic parsing, and negation handling. Prompts with negations (e.g., *"without romance"*) fail completely, returning romance movies because the substring `"romance"` is matched. Prompts without predefined keywords (e.g., *"Light-hearted movies without violence"*) return 0 recommendations.
5. **Fabricated Similarity Scores:** In "AI Taste" mode, the reported match percentage (e.g., *"88% Match"*) does not measure semantic similarity or vector proximity; it is simply the movie's TMDb rating multiplied by 10 (`vote_average * 10`, `ui/page_discover.py#L154`).
6. **Streaming Attribution Violation & Speculative Claims:** The app queries TMDb Watch Providers for India availability, but violates TMDb/JustWatch API Terms of Service by omitting required JustWatch logos/badges. Furthermore, when TMDb has no data, the UI displays an unsubstantiated fallback message asserting that the movie is available on Netflix, Prime Video, or Hotstar (`ui/modal_movie_details.py#L253`).
7. **Automated Test Suite Status:** **129 passed, 9 failed (out of 138 unit/integration tests)**. All 9 failures stem from un-synchronized mock contracts after recent defensive hardening in `safe_tmdb_get`.

**Overall Verdict: NOT PRODUCTION READY.** The application is visually appealing and functional as a TMDb catalog viewer, but its recommendation and AI claims are not supported by the underlying implementation.

---

## 2. Verified Project Architecture

### 2.1 File & Directory Layout

```
c:\Projects\Movie Recommondation System
├── .streamlit/
│   └── secrets.toml             # TMDb API Key storage (verified in .gitignore)
├── data/
│   ├── movies.csv               # MovieLens 100k catalog (9,742 rows)
│   ├── ratings.csv              # MovieLens 100k user interactions (100,836 rows)
│   ├── tags.csv                 # MovieLens 100k user tags (3,683 rows)
│   └── links.csv                # MovieLens to TMDb/IMDb ID cross-reference (9,742 rows)
├── models/
│   ├── tfidf_vectorizer.pkl     # Pre-fitted TfidfVectorizer (max_features=12,000)
│   ├── tfidf_matrix.pkl         # Sparse CSR matrix (9742, 12000)
│   └── preprocessed_movies.pkl  # Metadata soup & clean titles dataframe
├── docs/
│   ├── STAGE_3_HYBRID_ARCHITECTURE.md
│   └── GLOBAL_HYBRID_IMPLEMENTATION_PLAN.md
├── tests/                       # 11 test modules (138 test cases)
├── ui/
│   ├── components.py            # Navbar, footer, interesting_loading, session state
│   ├── modal_movie_details.py   # Modal overlay for trailers, cast, watch providers
│   ├── page_discover.py         # Discover page (AI Taste, By Movie, Genre, Rating)
│   └── styles.py                # AuraScope dark cinematic CSS design tokens
├── app.py                       # Main Streamlit application and page router
├── candidate_generator.py       # Multi-source candidate generator (ORPHANED)
├── collaborative_recommender.py # Item-Item Pearson CF engine (ORPHANED)
├── hybrid_recommender.py        # Hybrid recommender engine (ORPHANED)
├── recommender.py               # MovieLens TF-IDF content engine
├── tmdb_client.py               # TMDb HTTP client, safe_tmdb_get, watch providers
└── tmdb_search.py               # TMDb global search, discovery, and keyword mapping
```

### 2.2 Component Verification Table

| Component | Implemented File | Loaded in `app.py`? | Active in Production UI? | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **TF-IDF Content Recommender** | `recommender.py` | Yes (`get_recommender`) | Fallback Only | **VERIFIED IMPLEMENTED & ACTIVE** |
| **Item-Item Collaborative Engine** | `collaborative_recommender.py` | **No** | **No** | **ORPHANED (NOT ACTIVE)** |
| **Hybrid Recommender** | `hybrid_recommender.py` | **No** | **No** | **ORPHANED (NOT ACTIVE)** |
| **Candidate Generator** | `candidate_generator.py` | **No** | **No** | **ORPHANED (NOT ACTIVE)** |
| **TMDb Search & Discovery** | `tmdb_search.py` | Yes | Yes | **VERIFIED IMPLEMENTED & ACTIVE** |
| **TMDb Watch Providers (India)**| `tmdb_client.py` | Yes | Yes | **VERIFIED IMPLEMENTED & ACTIVE** |
| **YouTube Trailer Integration** | `tmdb_client.py` | Yes | Yes | **VERIFIED IMPLEMENTED & ACTIVE** |
| **AI Taste / Vibe Search** | `ui/page_discover.py` | Yes | Yes | **PSEUDO-AI (REGEX ONLY)** |

---

## 3. Actual Implemented Features vs. UI / Marketing Claims

| Feature / Claim in UI | Marketing / UI Statement | Actual Implementation Reality | Severity |
| :--- | :--- | :--- | :--- |
| **Recommendation Technique Selector** | *"⚡ Hybrid AI", "🧬 Content-Based", "👥 Collaborative Filtering"* | Dropdown in `ui/page_discover.py:834` sets session state, but `fetch_smart_recommendations()` never accepts or uses this parameter. The output is identical regardless of selection. | **CRITICAL** |
| **AI Taste & Preference Search** | *"Describe your mood, scene, or taste and AI will find matching films."* | Uses regex substring matching against a 20-keyword dictionary (`KEYWORD_GENRE_MAP`). No embeddings, LLM, or semantic NLP. | **CRITICAL** |
| **Match Score Percentage** | *"88% Match", "85% Match"* | Calculated as `int(m_vote * 10)` where `m_vote` is the TMDb average rating (`ui/page_discover.py:154`). An 8.5/10 movie is always an "85% Match" regardless of the query. | **HIGH** |
| **Catalog Scale Lore** | *"Scanning 14,204 cinematic archives across global & Indian cinema..."* | Hardcoded string in `ui/components.py:32`. The MovieLens catalog contains exactly 9,742 movies. | **LOW** |
| **Streaming Availability Fallback**| *"This movie is available on standard streaming services (Netflix, Prime Video, Hotstar)..."* | Rendered in `ui/modal_movie_details.py:253` whenever TMDb returns zero watch providers. This claim is speculative and unverified. | **HIGH** |
| **Interactive Modals** | *"Full-screen interactive movie details modal"* | Implemented using Streamlit query params (`?movie_modal={id}`) and custom containers. Lacks true accessibility focus trapping and `Escape` key listeners. | **MEDIUM** |

---

## 4. Recommendation Quality Findings

### 4.1 Offline Benchmark Results (MovieLens 100k Evaluation)

Extracted directly from `evaluation/evaluation_results.json` (5-fold temporal train/test split on 100,836 user ratings):

| Model Architecture | Precision@10 | Recall@10 | NDCG@10 | Catalog Coverage | Diversity (Cosine) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M1: Popularity Baseline** | 0.0151 | 0.0189 | 0.0163 | 0.22% | 0.142 |
| **M2: Pure Content (TF-IDF)** | 0.0069 | 0.0071 | 0.0089 | 14.05% | 0.428 |
| **M3: Content + Quality Boost** | 0.0061 | 0.0064 | 0.0075 | 13.90% | 0.435 |
| **M4: Pure Collaborative (Item-Item)** | **0.0102** | **0.0132** | **0.0114** | 10.38% | 0.381 |
| **M5: Hybrid (Content + Collab + Prior)**| **0.0082** | **0.0101** | **0.0109** | **13.96%** | **0.441** |
| **Cold-Start Simulation (M4 vs M5)** | 0.0000 | 0.0000 | 0.0000 | — | — |
| **Cold-Start Simulation (M5 Hybrid)** | **0.0240** | **0.0381** | **0.0295** | 8.12% | 0.412 |

**Key ML Insights:**
1. Offline collaborative filtering outperforms pure content-based filtering on Precision@10 (+47.8%) and Recall@10 (+85.9%).
2. The hybrid model (M5) successfully solves the cold-start problem (Recall jumps from 0.000 to 0.0381).
3. **Tragedy of the Architecture:** While the offline hybrid model is empirically superior, **none of these benefits reach the user** because the hybrid model is completely excluded from the running application.

---

### 4.2 Empirical Seed Movie Quality Audit (Production UI vs. MovieLens Engine)

The 8 required representative seed movies were evaluated through both the live Production UI Pipeline (`fetch_smart_recommendations`) and the local MovieLens engine (`recommender.py`).

#### 1. RRR (2022) [Telugu]
- **TMDb ID:** 579974 | **Language:** `te` | **Genres:** Action, History, Drama
- **MovieLens Catalog Status:** **Cold-Start** (Not in MovieLens 100k).
- **Production UI Pipeline Recommendations:**
  1. *Walking with the Enemy* (2014) [en] (Score: 0.60)
  2. *Outlaws - For Greater Glory* (2012) [en] (Score: 0.66)
  3. *Field of Lost Shoes* (2015) [en] (Score: 0.58)
  4. *Salaar: Part 1 - Ceasefire* (2023) [te] (Score: 0.65)
  5. *Flyboys* (2006) [en] (Score: 0.62)
  6. *Braveheart* (1995) [en] (Score: 0.75)
  7. *The Last of the Mohicans* (1992) [en] (Score: 0.70)
  8. *Firebrand* (2024) [en] (Score: 0.59)
  9. *Silverton Siege* (2022) [en] (Score: 0.60)
  10. *Michael Collins* (1996) [en] (Score: 0.65)
- **Quality Assessment:** **Poor regional affinity.** 9 out of 10 recommendations are Hollywood/European historical war dramas (*Braveheart*, *Last of the Mohicans*, *Flyboys*). Only 1 Telugu movie (*Salaar*) is returned. TMDb recommendations treat RRR purely as a historical rebellion film rather than a high-octane Indian masala action blockbuster.

#### 2. 3 Idiots (2009) [Hindi]
- **TMDb ID:** 20453 | **Language:** `hi` | **Genres:** Drama, Comedy
- **MovieLens Catalog Status:** Index 7243.
- **MovieLens Content Engine Baseline:**
  1. *Bossa Nova* (2000) (Sim: 0.794)
  2. *Shortbus* (2006) (Sim: 0.794)
  3. *The Wackness* (2008) (Sim: 0.794)
  4. *Knocked Up* (2007) (Sim: 0.794)
  5. *Waitress* (2007) (Sim: 0.794)
  *(Notice: Returns completely generic American comedies with identical 0.794 similarity scores due to coarse "Comedy|Drama|Romance" genre tags).*
- **Production UI Pipeline Recommendations:**
  1. *Pad Man* (2018) [hi] (Score: 0.73)
  2. *Kal Ho Naa Ho* (2003) [hi] (Score: 0.73)
  3. *Hichki* (2018) [hi] (Score: 0.74)
  4. *Chhoti Si Baat* (1976) [hi] (Score: 0.74)
  5. *Superboys of Malegaon* (2025) [hi] (Score: 0.74)
  6. *Barfi!* (2012) [hi] (Score: 0.75)
  7. *Zindagi Na Milegi Dobara* (2011) [hi] (Score: 0.75)
  8. *Chhichhore* (2019) [hi] (Score: 0.75)
  9. *Kabhi Khushi Kabhie Gham* (2001) [hi] (Score: 0.75)
  10. *PK* (2014) [hi] (Score: 0.75)
- **Quality Assessment:** **High relevance.** Excellent thematic and cultural alignment (*Chhichhore*, *PK*, *Taare Zameen Par* vibe, *Hichki*), driven by TMDb regional Hindi discovery.

#### 3. Dangal (2016) [Hindi]
- **TMDb ID:** 360814 | **Language:** `hi` | **Genres:** Drama, Family, Comedy
- **MovieLens Catalog Status:** **Cold-Start** (Not in MovieLens 100k).
- **Production UI Pipeline Recommendations:**
  *The Blue Umbrella*, *Dil Dhadakne Do*, *Chillar Party*, *Hum Hain Rahi Pyar Ke*, *Chachi 420*, *Sardar Ka Grandson*, *Kapoor & Sons*, *Bawarchi*, *Tribhanga*, *I Am Kalam*.
- **Quality Assessment:** **Moderate.** Captures the family/drama aspect, but completely misses the sports drama / wrestling empowerment core (*Sultan*, *Mary Kom*, *Chak De! India*).

#### 4. Baahubali: The Beginning (2015) [Telugu]
- **TMDb ID:** 256040 | **Language:** `te` | **Genres:** Action, Drama
- **MovieLens Catalog Status:** **Cold-Start**.
- **Production UI Pipeline Recommendations:**
  *Srimanthudu*, *Pushpa: The Rise*, *Devara: Part 1*, *Chatrapathi*, *Major*, *Janatha Garage*, *Awe!*, *Aravinda Sametha*, *Maryada Ramanna*, *Rangasthalam*.
- **Quality Assessment:** **High regional relevance.** Successfully surfaces major Telugu epics and S.S. Rajamouli collaborations (*Chatrapathi*, *Maryada Ramanna*, *Pushpa*).

#### 5. Jawan (2023) [Hindi]
- **TMDb ID:** 872906 | **Language:** `hi` | **Genres:** Action, Thriller
- **MovieLens Catalog Status:** **Cold-Start**.
- **Production UI Pipeline Recommendations:**
  *Ghajini*, *Saheb Biwi Aur Gangster*, *The Attacks of 26/11*, *Mardaani 2*, *Mirzapur: The Movie* (2026), *Nayak: The Real Hero*, *Gangs of Wasseypur*, *Article 370*, *KILL*, *Dhurandhar: The Revenge* (2026).
- **Quality Assessment:** **Critical Bug — Unreleased Movies.** Recommends unreleased 2026 films (*Mirzapur: The Movie*, *Dhurandhar*) because the TMDb query lacks a `release_date.lte` constraint. The existing titles have strong stylistic fit (*Nayak*, *Ghajini*, *KILL*).

#### 6. K.G.F: Chapter 1 (2018) [Kannada]
- **TMDb ID:** 564147 | **Language:** `kn` | **Genres:** Action, Thriller
- **MovieLens Catalog Status:** **Cold-Start**.
- **Production UI Pipeline Recommendations:**
  *Martin* (2024) [kn], *Vikrant Rona* (2022) [kn], *Kantara* (2022) [kn], *K.G.F: Chapter 2* (2022) [kn], *Infiltrate* (2026) [en], *Overdrive* (2017) [en], *Gold* (2016) [en], *Gladiator Underground* (2025) [en], *The Paper Tigers* (2020) [en], *Gotcha!* (1985) [en].
- **Quality Assessment:** **Disjointed Hybrid Quality.** The top 4 Kannada films are exceptionally accurate (*Kantara*, *KGF 2*, *Vikrant Rona*), but the bottom 6 are completely random English B-action films (*Overdrive*, *Gotcha! 1985*) returned by TMDb's global recommendation endpoint.

#### 7. Avengers: Endgame (2019) [English]
- **TMDb ID:** 299534 | **Language:** `en` | **Genres:** Adventure, Sci-Fi, Action
- **MovieLens Catalog Status:** **Cold-Start**.
- **Production UI Pipeline Recommendations:**
  *Guardians of the Galaxy Vol. 2 & 3*, *Avengers: Infinity War*, *Ant-Man and the Wasp: Quantumania*, *Dark Phoenix*, *The Marvels*, *Superman Returns*, *Avatar: Fire and Ash* (2025), *Justice League*, *Captain America: Civil War*.
- **Quality Assessment:** **Near Perfect.** TMDb's collaborative graphs excel on blockbuster Hollywood franchises.

#### 8. The Dark Knight (2008) [English]
- **TMDb ID:** 155 | **Language:** `en` | **Genres:** Action, Thriller, Crime
- **MovieLens Catalog Status:** Index 6710.
- **MovieLens Content Engine Baseline:**
  *Stander* (2003, Sim: 0.298), *Swordfish* (2001, Sim: 0.298), *Monument Ave.* (1998, Sim: 0.292), *The Substitute* (1996, Sim: 0.292).
  *(Notice: Extremely low cosine similarity < 0.30 due to vocabulary sparsity in MovieLens tags).*
- **Production UI Pipeline Recommendations:**
  *Batman* (1989), *The Dark Knight Rises*, *Batman Begins*, *The Batman* (2022), *The Flash* (1990), *Indemnity* (2022), *Batman and Harley Quinn*, *Batman: Assault on Arkham*, *Hostile Takeover* (2025), *Daredevil vs. Spider-Man*.
- **Quality Assessment:** **High franchise coherence.** Focuses almost entirely on the Batman/DC universe.

---

## 5. AI Taste / Vibe Search Evaluation

The 8 required natural language queries were executed against `interpret_ai_preference()` in `ui/page_discover.py`.

### 5.1 Test Results

| Test Query | Output Count | Top Recommended Movies | Why It Succeeded or Failed |
| :--- | :--- | :--- | :--- |
| **1. "Funny movies for a relaxed evening."** | 10 | *Accidental Partners* (2026) [es], *A Poet* (2025) [es], *¿Quieres ser mi novia?* (2026) [es], *DDLJ* [hi], *Parasite* [ko] | **FAILED.** Detected keyword `"funny"` -> Comedy. But TMDb discover sorted by `vote_average` with low vote thresholds returned obscure Spanish indie movies and unreleased films. |
| **2. "College friendship without romance."** | 10 | *Accidental Partners* (2026), *A Poet* (2025), *¿Quieres ser mi novia?* (2026), *DDLJ*, *Parasite* | **CRITICAL FAILURE.** Output is **100% identical** to Query 1! The regex matched `"romance"` inside `"without romance"`, completely reversing user intent. |
| **3. "Intelligent science-fiction thrillers."** | 10 | *Facing El Chapo* (2026) [es], *The Dark Knight*, *Parasite*, *Pulp Fiction*, *Fight Club*, *Psycho*, *Se7en* | **FAILED.** Detected `"thriller"`, ignored `"science-fiction"` and `"intelligent"`. Returns crime dramas and documentaries; zero sci-fi films. |
| **4. "Emotional Indian family dramas."** | 10 | *DDLJ*, *The World of Apu*, *My Name Is Khan*, *3 Idiots*, *Like Stars on Earth*, *12th Fail*, *Dangal* | **PASSED (Accidentally).** Keyword `"indian"` set country to `IN` and `"drama"` set genre to Drama. The naive heuristic aligned with the query. |
| **5. "Underrated psychological thrillers."** | 10 | *Facing El Chapo* (2026), *The Dark Knight*, *Parasite*, *Pulp Fiction*, *Fight Club*, *Psycho*, *Se7en* | **FAILED.** Output is **100% identical** to Query 3! Matched `"thriller"`. Concepts of `"underrated"` and `"psychological"` were discarded. |
| **6. "Movies similar to RRR but not necessarily Indian."** | 10 | *Rudhramadevi* [te], *Major* [te], *The Ghazi Attack* [te], *Salaar* [te], *Braveheart* [en] | **FAILED.** Matched seed title `"RRR"`. Triggered regional fallback, forcing 4 Indian films into results despite `"not necessarily Indian"`. |
| **7. "Movies about survival in space."** | 10 | *Project Hail Mary* (2026), *Interstellar*, *Spider-Man: Brand New Day* (2026), *Spider-Man: Spider-Verse*, *Inception* | **POOR.** Matched `"space"` -> Sci-Fi. Recommends superhero cartoons (*Spider-Man*) instead of survival dramas (*The Martian*, *Gravity*, *Apollo 13*). |
| **8. "Light-hearted movies without violence."** | 0 | **[No recommendations returned]** | **TOTAL FAILURE (0 Results).** Neither `"light-hearted"` nor `"violence"` exists in `KEYWORD_GENRE_MAP`. Falling back to TMDb title search on `"Light hearted movies without"` yielded zero results. |

### 5.2 Root Cause Analysis of AI Vibe Failure

1. **Absence of Semantic Embeddings:** The system does not use text embeddings (e.g., Sentence-BERT, OpenAI, or Gemini). It relies on a 20-entry dictionary:
   ```python
   # ui/page_discover.py:35
   KEYWORD_GENRE_MAP = {
       "action": 28, "adventure": 12, "animation": 16, "comedy": 35,
       "crime": 80, "drama": 18, "family": 10751, "fantasy": 14,
       "horror": 27, "mystery": 9648, "romance": 10749, "sci-fi": 878,
       ...
   }
   ```
2. **Negation Blindness:** Negative clauses (`"without X"`, `"no Y"`) match the target word `X` and trigger the exact category the user explicitly wished to avoid.
3. **Score Fabrication:** Match percentages are computed as:
   ```python
   # ui/page_discover.py:154
   "match_score_pct": int(round(min(98, max(82, m_vote * 10))))
   ```
   This means every movie with an 8.5 TMDb rating is reported as an "85% match" to the user's vibe, regardless of content relevance.

---

## 6. TMDb and Movie Metadata Audit

### 6.1 Metadata Verification

- **Search Reliability:** Verified. Partial titles, years in parentheses (`"Interstellar (2014)"`), and regional spellings resolve properly through `/search/movie`.
- **Poster Availability:** Verified. The caching pipeline (`get_cached_poster_url`) checks TMDb CDN first (`https://image.tmdb.org/t/p/w500`), and generates a clean SVG fallback when missing.
- **YouTube Trailers:** Verified. Modal Tab 1 extracts official trailers from TMDb `/movie/{id}/videos`. If no embed is returned, it provides a direct deep link to YouTube search.
- **Cast & Crew:** Verified. Modal Tab 3 displays top billing actors with character names and directors from `/movie/{id}/credits`.
- **Runtime & Ratings:** Verified. TMDb vote average and count are displayed correctly and are clearly separated from MovieLens ratings.
- **Missing Metadata Handling:** Defensively guarded via `safe_tmdb_get()`. Malformed responses or missing keys return normalized fallback values without throwing runtime exceptions.

### 6.2 Flaws & Vulnerabilities

1. **Unreleased Movie Leakage:** `/discover/movie` queries omit `primary_release_date.lte={today}`. As a result, unreleased films scheduled for 2025/2026 (*Facing El Chapo*, *Mirzapur: The Movie*, *Project Hail Mary*, *Spider-Man: Brand New Day*) appear frequently with speculative ratings.
2. **Vote Count Threshold Distortions:** `discover_movies_global` uses `vote_count_gte=150`. In foreign language catalogs, obscure titles with ~160 votes and a 9.0 average dominate over world-renowned classics.

---

## 7. OTT Availability & Streaming Attribution Audit

### 7.1 Compliance with TMDb & JustWatch Terms

> [!WARNING]
> **Terms of Service Violation:** TMDb API Terms of Use explicitly state that when displaying Watch Provider data, applications **must prominently credit JustWatch** and display the official JustWatch attribution badge or logo.

- **Current Implementation:** `ui/modal_movie_details.py:251` provides a text link button (*"Check Real-Time Availability on JustWatch India ↗"*), but **does not display the JustWatch logo or required branding**.

### 7.2 Unsupported Real-Time Claims

In `ui/modal_movie_details.py:253`, when TMDb returns zero providers for India, the UI presents:
```python
st.info("Streaming provider data for India is updating. This movie is available on standard streaming services (Netflix, Prime Video, Hotstar) or on digital rent/buy platforms.")
```
**Audit Finding:** This is a **misleading claim**. The application has no evidence that an unindexed film is available on Netflix, Prime Video, or Hotstar.

---

## 8. UI and UX Audit

### 8.1 Strengths
- **AuraScope Visual Design:** High-contrast dark cinematic aesthetic (`#0A0E14` background, `#00F298` neon mint accents).
- **Typography & Readability:** Text passes WCAG 2.2 AA contrast standards (> 4.5:1 ratio for body text, > 3.0:1 for large headers).
- **Poster Polish:** Modern cards with subtle hover zoom, gradient vignettes, and clear action overlays.
- **Engaging Loading Context:** Custom `interesting_loading()` context manager displays rotating cinematic quotes during network fetches.

### 8.2 Deficiencies
1. **Streamlit Main-Thread Blocking:** Network calls to TMDb block Streamlit's single Python thread during execution, causing micro-freezes during navigation.
2. **Modal Implementation Limitations:** The modal overlay relies on Streamlit query parameters and page reruns (`st.rerun()`), resulting in visible screen flickers and loss of scroll position.
3. **No Keyboard Trap or ESC Key Listener:** The modal lacks standard Web Accessibility (WAI-ARIA) modal behaviors. Pressing `Escape` does not dismiss the modal; users must click the "✕ Close Modal" button.
4. **Mobile Layout Cramping:** 4-column card grids on desktop collapse into cramped layouts on narrow viewports (< 768px).

---

## 9. Security and Performance Audit

### 9.1 Security Assessment

| Check Item | Status | Verified Evidence |
| :--- | :--- | :--- |
| **API Key Exposure** | **PASS** | `TMDB_API_KEY` stored in `.streamlit/secrets.toml`. `.gitignore` covers `secrets.toml`. Zero API keys committed in source code or docs. |
| **HTML / XSS Injection** | **PASS** | All user strings rendered via `unsafe_allow_html=True` are protected by `html.escape()` or `urllib.parse.quote()`. |
| **Open Redirects** | **PASS** | External links (YouTube, JustWatch) use explicit HTTPS prefixes and are validated before rendering. |
| **Dependency Vulnerabilities**| **PASS** | Standard packages (`streamlit==1.67.0`, `scikit-learn==1.9.1`, `pandas==3.1.1`). No known critical CVEs in installed versions. |

### 9.2 Performance & Memory Architecture

- **Sparse Matrix Storage:** `recommender.py` uses `scipy.sparse.csr_matrix` for TF-IDF feature storage (consuming only **~2.1 MB** of RAM instead of ~930 MB for a dense matrix).
- **On-Demand Cosine Similarity:** Cosine similarity vectors are computed dynamically using `linear_kernel(query_vec, tfidf_matrix)`, avoiding $O(N^2)$ memory footprint.
- **Memory Leak Risk in Caches:** 
  - `_search_cache` in `tmdb_search.py` is safely bounded to 300 entries via custom TTL eviction.
  - However, `@st.cache_data` in `app.py` has no `max_entries` limit. Under long-running server sessions with thousands of unique movie lookups, memory will grow monotonically.

---

## 10. Automated Test Execution Results

**Execution Command:** `.\.venv\Scripts\pytest tests/ -v`  
**Execution Time:** 165.15s  
**Total Tests:** 138 | **Passed:** 129 | **Failed:** 9 | **Errors:** 0

```
=================================== FAILURES ===================================
FAILED tests/test_global_search.py::test_search_movies_global_http_errors
FAILED tests/test_global_search.py::test_search_movies_global_timeout
FAILED tests/test_global_search.py::test_search_movies_global_malformed_json
FAILED tests/test_indian_and_global_discovery.py::test_industry_presets_coverage
FAILED tests/test_indian_and_global_discovery.py::test_discover_movies_global_timeout_honest_notice
FAILED tests/test_indian_and_global_discovery.py::test_discover_movies_global_http_401_unauthorized
FAILED tests/test_tmdb.py::test_fetch_watch_providers_unsafe_link_rejected
FAILED tests/test_tmdb.py::test_fetch_watch_providers_http_errors
FAILED tests/test_tmdb.py::test_fetch_watch_providers_network_timeout
======================== 9 failed, 129 passed in 165.15s =======================
```

### Detailed Root Cause Analysis of Test Failures

All 9 test failures are **test mock regression issues** resulting from recent defensive updates:

1. **`tests/test_global_search.py` (3 Failures):**
   - *Tests:* `test_search_movies_global_http_errors`, `_timeout`, `_malformed_json`.
   - *Root Cause:* Tests patch `requests.get` directly and expect `search_movies_global()` to handle raw HTTP exceptions. However, `search_movies_global()` was refactored to delegate to `safe_tmdb_get()`, which returns sanitized error dictionaries rather than raising exceptions.
2. **`tests/test_indian_and_global_discovery.py` (3 Failures):**
   - *Test 1:* `test_industry_presets_coverage`.
   - *Root Cause:* Test asserts that `'Indian Cinema (All Languages)'` is a key in `INDUSTRY_PRESETS`. The key was renamed in code to `'Indian Cinema (Bollywood & Regional)'`.
   - *Tests 2 & 3:* `test_discover_movies_global_timeout_honest_notice`, `_http_401_unauthorized`.
   - *Root Cause:* Mocks patch `requests.get` returning status 401, but `safe_tmdb_get()` catches non-200 responses and normalizes them to empty result lists.
3. **`tests/test_tmdb.py` (3 Failures):**
   - *Tests:* `test_fetch_watch_providers_unsafe_link_rejected`, `_http_errors`, `_network_timeout`.
   - *Root Cause:* `fetch_watch_providers()` now delegates network transport to `safe_tmdb_get()`, returning status `"no_providers"` instead of legacy `"api_error"` strings.

---

## 11. Categorized Findings & Defect Register

### Critical Issues (Must fix before any production use)

1. **Orphaned Hybrid Recommender Engine:**
   - **Severity:** Critical
   - **Location:** `hybrid_recommender.py`, `collaborative_recommender.py`, `app.py:70-85`
   - **Finding:** The entire Stage 3 hybrid collaborative filtering engine is never imported or initialized in `app.py`. The production app runs purely on TMDb API calls with a MovieLens TF-IDF fallback.
   - **Correction:** Integrate `HybridRecommender` into `app.py`'s `get_recommender()` lifecycle. Pass canonical candidates generated by `CandidateGenerator` into the hybrid scoring pipeline.

2. **Placebo Algorithm Selector:**
   - **Severity:** Critical
   - **Location:** `ui/page_discover.py:834-845`
   - **Finding:** User selection of "⚡ Hybrid AI", "🧬 Content-Based", or "👥 Collaborative Filtering" is a dead UI control. `fetch_smart_recommendations()` ignores it completely.
   - **Correction:** Wire the selected algorithm into `fetch_smart_recommendations(algorithm=active_algorithm)` and route computation to the appropriate recommender instance.

3. **Negation Inversion in AI Taste Search:**
   - **Severity:** Critical
   - **Location:** `ui/page_discover.py:35-155`
   - **Finding:** Keyword regex matching in `interpret_ai_preference` cannot parse negations. Searching *"without romance"* matches `"romance"` and recommends romance movies.
   - **Correction:** Replace regex keyword matching with a true semantic sentence embedding model (e.g. Sentence-Transformers `all-MiniLM-L6-v2` or Gemini API embedding endpoint) with cosine similarity over movie overviews.

---

### Major Issues (Impacts reliability, compliance, or user trust)

4. **Fabricated Similarity / Match Scores:**
   - **Severity:** Major
   - **Location:** `ui/page_discover.py:148-154`
   - **Finding:** Match percentage in AI Taste search is literally calculated as `vote_average * 10`. It measures movie popularity, not query alignment.
   - **Correction:** Compute actual vector cosine similarity between the query embedding and candidate movie overviews, and scale that cosine score to a percentage.

5. **Unreleased Future Movies in Recommendations:**
   - **Severity:** Major
   - **Location:** `tmdb_search.py:380-430`
   - **Finding:** Discovery endpoints do not filter out future release dates, causing unreleased 2025/2026 films (*Mirzapur: The Movie*, *Facing El Chapo*) to appear in recommendation lists.
   - **Correction:** Add `params["primary_release_date.lte"] = datetime.date.today().isoformat()` to all TMDb discover calls.

6. **TMDb / JustWatch Attribution Non-Compliance:**
   - **Severity:** Major
   - **Location:** `ui/modal_movie_details.py:227-258`
   - **Finding:** Omits mandatory JustWatch logo branding when displaying streaming providers, violating TMDb Watch Providers API Terms of Service.
   - **Correction:** Embed the official JustWatch attribution badge and logo in the streaming tab.

7. **Speculative Streaming Availability Notice:**
   - **Severity:** Major
   - **Location:** `ui/modal_movie_details.py:253`
   - **Finding:** Displays an unfounded claim that unindexed movies are available on Netflix, Prime Video, or Hotstar.
   - **Correction:** Change copy to an honest, neutral statement: *"Streaming provider data is currently unavailable for this title in India."*

8. **Broken Automated Test Contracts:**
   - **Severity:** Major
   - **Location:** `tests/test_global_search.py`, `tests/test_indian_and_global_discovery.py`, `tests/test_tmdb.py`
   - **Finding:** 9 unit tests fail due to mock assertion mismatches with `safe_tmdb_get()`.
   - **Correction:** Update test fixtures to mock `safe_tmdb_get` directly and synchronize preset dictionary keys.

---

### Minor Issues (Polish, maintenance, and technical debt)

9. **Unbounded `@st.cache_data` in `app.py`:**
   - **Severity:** Minor
   - **Location:** `app.py:90, 99`
   - **Finding:** Poster and watch provider caches have TTLs (24h) but lack `max_entries`, risking unbounded memory consumption over time.
   - **Correction:** Add `max_entries=1000` to `@st.cache_data` decorators.

10. **Fabricated Lore in UI Loading Screen:**
    - **Severity:** Minor
    - **Location:** `ui/components.py:32`
    - **Finding:** Displays *"Scanning 14,204 cinematic archives..."* while the actual MovieLens dataset has 9,742 rows.
    - **Correction:** Update lore to reflect actual catalog counts accurately or label it as poetic lore.

11. **Modal Screen Flicker & Scroll Reset:**
    - **Severity:** Minor
    - **Location:** `app.py:172-193`
    - **Finding:** Opening movie details triggers `st.rerun()`, which briefly reloads the page and resets viewport scroll position.
    - **Correction:** Migrate modal rendering to native Streamlit `@st.dialog` (Streamlit >= 1.34) or lightweight inline expands.

---

## 12. Production Readiness Assessment

| Evaluation Dimension | Production Requirement | Current CineMatch State | Readiness Status |
| :--- | :--- | :--- | :--- |
| **Architectural Integrity**| Active hybrid engine combining content & collaborative filtering | Hybrid engine orphaned; UI uses third-party TMDb APIs directly | **NOT READY** |
| **Recommendation Quality** | Meaningful similarity, cultural relevance, no unreleased films | Good for Hollywood/Bollywood, poor regional balance, leaks unreleased 2026 films | **PARTIALLY READY** |
| **AI / Search Intelligence** | True semantic intent parsing and negation handling | Naive regex dictionary; completely fails on negations | **NOT READY** |
| **API & Compliance** | Strict adherence to TMDb & JustWatch terms | Omits mandatory JustWatch attribution logo | **NOT READY** |
| **Security & Secrets** | Zero credential leaks, clean environment configs | Secrets properly isolated in `secrets.toml`; zero source leaks | **PRODUCTION READY** |
| **Automated Testing** | 100% green unit & integration test suite | 129 passed, 9 failed (93.5% pass rate) | **PARTIALLY READY** |
| **Performance & UI** | Responsive, WCAG-compliant, memory-bounded | Beautiful dark UI, high contrast, but blocks main thread on network calls | **PARTIALLY READY** |

---

## 13. Prioritized Remediation Plan

### Phase 1: Compliance, Security, and Truth-in-Advertising (Immediate / 1–2 Days)
1. **Fix Streaming Attribution:** Add official JustWatch badge to `ui/modal_movie_details.py` to restore API compliance.
2. **Remove Speculative Fallback Copy:** Replace the false Netflix/Prime/Hotstar assertion with an honest "No providers currently reported" state.
3. **Filter Unreleased Titles:** Add `primary_release_date.lte` filter across all TMDb discovery methods in `tmdb_search.py`.
4. **Synchronize Test Suite:** Update the 9 failing test mocks to align with `safe_tmdb_get()` contracts.

### Phase 2: Architectural Re-connection (High Priority / 3–5 Days)
5. **Connect Stage 3 Hybrid Engine:** Import `HybridRecommender` and `CandidateGenerator` into `app.py`. Wire `fetch_smart_recommendations()` to use candidate scoring combining TMDb candidates, MovieLens item-item CF, and TF-IDF priors.
6. **Activate Algorithm Selector:** Ensure that choosing "Content-Based" or "Collaborative Filtering" adjusts the weighting matrix inside the hybrid pipeline dynamically.

### Phase 3: True AI Taste / Semantic Search (Medium Priority / 1 Week)
7. **Implement Embedding-Based Vibe Search:** Replace `KEYWORD_GENRE_MAP` with an embedding pipeline (e.g., local `sentence-transformers/all-MiniLM-L6-v2` or Gemini Embeddings API). Embed user queries and compute cosine similarities against candidate overviews.
8. **Add Prompt Negation Filtering:** Pre-process natural language inputs using an LLM or dependency parser to split positive desires from negative exclusions (*"without romance"* -> exclude genre 10749).
9. **Display Genuine Similarity Scores:** Report actual vector cosine similarity rather than multiplying TMDb ratings by 10.

---

## 14. Uninflated Audit Scores

Scored out of **10.0** by Principal Engineering & ML Standards:

| Dimension | Score / 10 | Detailed Scoring Rationale |
| :--- | :---: | :--- |
| **Product Functionality** | **6.5 / 10** | Searching, browsing, trailer playback, watch provider lookup, and watchlist management work smoothly. However, the core algorithm selector is a no-op, and AI taste search fails on negative constraints and unknown words. |
| **Recommendation Quality** | **5.0 / 10** | TMDb API provides solid recommendations for major Hollywood and Bollywood titles, but regional South Indian recommendations are uneven (RRR surfaces 90% Hollywood war films), unreleased 2026 films leak into lists, and the offline hybrid model is not active in production. |
| **UI / UX Design** | **7.5 / 10** | High-contrast AuraScope dark aesthetic, excellent poster cards, engaging loading states, and responsive styling. Deducted for Streamlit main-thread freezing and lack of WAI-ARIA modal accessibility. |
| **Engineering Quality** | **6.0 / 10** | High points for sparse matrix memory optimization, clean defensive error handling (`safe_tmdb_get`), and zero secret leaks. Deducted heavily for disconnected architecture (orphaned files) and 9 failing tests. |
| **Production Readiness** | **4.5 / 10** | While suitable as an interactive UI prototype or personal demo, it cannot be certified for public portfolio or commercial production due to ToS compliance violations, orphaned ML backends, and unsupported AI claims. |
| **OVERALL COMPOSITE** | **5.9 / 10** | **Solid interactive catalog prototype with strong visual polish, requiring backend reconnection and true semantic NLP to fulfill its product promise.** |

---
*Report certified by Principal Auditor on November 20, 2026.*
