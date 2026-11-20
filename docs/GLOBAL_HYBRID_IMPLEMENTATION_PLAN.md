# CineMatch AI — Global Movie Catalog + Hybrid Recommendation Engine
## Master Engineering Implementation Plan (Phase Zero Baseline & Architectural Specification)

**Project:** CineMatch AI  
**Workspace:** `C:\Projects\Movie Recommondation System`  
**Development Platform:** Windows 11 · Python 3.11.8 · Streamlit  
**Infrastructure Budget:** ₹0/month (Zero Paid Services, Zero Paid APIs)  
**Document Version:** 1.0.0 (Phase Zero Approval Gate)  
**Author:** Antigravity Senior Engineering Team  
**Date:** November 19, 2026  

---

## 1. Executive Summary & Core Invariants

This engineering plan details the architectural evolution of **CineMatch AI** from a local dataset recommendation prototype (MovieLens Latest Small, 9,742 titles) into an enterprise-grade **Global Movie Catalog & Hybrid Recommendation System** powered by official The Movie Database (TMDb) API integration, sparse TF-IDF vector projection, Bayesian quality blending, and verified India streaming availability.

### Non-Negotiable Invariants:
1. **Strict Dataset & Rating Provenance**: MovieLens ratings ($0.5 - 5.0$) and TMDb ratings ($0.0 - 10.0$) represent completely distinct populations, scales, and review mechanics. They will **never** be mixed, averaged together, or substituted for one another.
2. **Similarity Score Integrity**: Cosine similarity is strictly labeled as a **Cosine Similarity Match Score** measuring directional feature overlap in vector space. It is **never** presented as recommendation accuracy, probability, or confidence percentage.
3. **Memory Optimization**: The system strictly maintains sparse compressed row matrices (`scipy.sparse.csr_matrix`) and never materializes a dense $N \times N$ similarity matrix. Querying external movies uses runtime projection (`vectorizer.transform()`) without re-fitting or allocating dense arrays.
4. **Authenticity Guarantee**: Zero fabricated metadata, ratings, plot summaries, posters, or OTT provider links. Where data is absent, explicit transparent fallbacks are surfaced.
5. **Resilience & Offline Autonomy**: If the TMDb API is unavailable, throttled, or unconfigured, the application continues to run seamlessly on the local MovieLens engine.
6. **Zero Infrastructure Cost**: Standard free TMDb developer API tier with aggressive client-side caching and intelligent rate-limiting.

---

## 2. Phase Zero Repository Baseline Audit

### 2.1 Existing Architecture
```
[User Browser]
       │
       ▼
[Streamlit Web App: app.py] (Port 8501)
  ├── Minimalist Light UI + Glassmorphism Design System
  ├── Active Movie Spotlight (240px Poster, 2:3 Aspect Ratio)
  ├── India OTT Where to Watch Section (Stream, Rent, Buy via JustWatch/TMDb)
  ├── Catalog Selectbox (9,742 titles) + Custom Fuzzy Search Expander
  └── Top-N Recommendation Cards with "Explore Similar" Pivoting
       │
       ├──► [MovieRecommender Core: recommender.py]
       │      ├── Dataset: MovieLens Latest Small (9,742 movies, 100,836 ratings, 3,683 tags)
       │      ├── Metadata Soup: Clean Title + (Genres x 2) + (Tags x 2) + Decade + Format
       │      ├── TF-IDF Vectorizer: 12,000 features (unigrams & bigrams, sublinear tf)
       │      ├── Feature Matrix: scipy.sparse.csr_matrix (9742 x 12000, ~812 KB)
       │      ├── On-Demand 1 x N Linear Kernel (~1.5 ms execution)
       │      └── Bayesian Rating Quality Blending: (v / (v+5))*R + (5 / (v+5))*C
       │
       └──► [TMDb Client: tmdb_client.py]
              ├── Custom IPv4 Socket Override (_tmdb_getaddrinfo) bypassing ISP DNS blocks
              ├── Poster Fetch: /3/movie/{id} -> TMDb Image CDN (w342)
              ├── Fallback SVG Generator: Accessible SVG Data URI labeled "Official poster unavailable"
              └── India Watch Providers: /3/movie/{id}/watch/providers (region="IN")
```

### 2.2 Model Artifacts on Disk
- `models/movies_preprocessed.pkl`: 4.43 MB serialized Pandas DataFrame containing cleaned titles, parsed years, tokenized genres, aggregated ratings, and compiled metadata soup.
- `models/tfidf_vectorizer.pkl`: 487.8 KB fitted `sklearn.feature_extraction.text.TfidfVectorizer` (12,000 vocabulary items).
- `models/tfidf_matrix.npz`: 812.9 KB compressed sparse CSR matrix (`scipy.sparse.csr_matrix`).

### 2.3 Current Test Baseline
- **Total Test Cases**: 42 automated tests across 3 suites.
  - `tests/test_preprocessing.py`: 6 tests (Dataset loading, title/year extraction, genre parsing, rating aggregation, tag cleaning, soup composition).
  - `tests/test_recommender.py`: 20 tests (Sparse memory efficiency, score integrity, fuzzy lookups, exclusion of self, genre filters, min rating filters, edge case movies without tags/genres, ranking descent, tie-breakers, HTML escaping).
  - `tests/test_tmdb.py`: 16 tests (API key safety, fallback SVG generation, XML injection defense, poster paths, mock responses, HTTP error codes 401/404/429/500, India provider categories, URL sanitization, timeout resilience).
- **Current Test Status**: **42 / 42 Passed (100%)** in 143.9 seconds.

### 2.4 Current Working Tree & Uncommitted State
- Working directory is clean of uncommitted git index conflicts (not a git repository root).
- Working tree contains recently enhanced versions of:
  - `app.py`: Integrated spotlight poster alt logic, categorized Where to Watch in India cards, responsive tablet/mobile layout (`flex: 1 1 320px`).
  - `tmdb_client.py`: Hardened error handling, empty string key handling, and fallback SVGs.
  - `tests/test_tmdb.py`: Comprehensive 16-test suite for TMDb endpoints.

### 2.5 Identified Architectural Gaps
1. **Catalog Boundary**: MovieLens Latest Small was frozen in September 2018. Titles produced after 2018 (e.g., *Avengers: Endgame (2019)*, *Spider-Man: No Way Home (2021)*, *Oppenheimer (2023)*, *Dune (2021)*) and regional cinema (*RRR*, *KGF*, *Pushpa*, *Jawan*, *Kantara*) cannot be found.
2. **Search Scope**: Search currently operates exclusively on the local 9,742 `movies_df`.
3. **External Movie Recommendation**: No mechanism exists to take a movie found on TMDb that is not in MovieLens and recommend similar movies from the MovieLens benchmark corpus or TMDb catalog.
4. **Dual Data Format**: No normalized data schema exists that can represent both a local MovieLens entity and an external TMDb entity.

---

## 3. Proposed Target Architecture

```
                                 [Streamlit UI Layer (app.py)]
                                               │
             ┌─────────────────────────────────┴─────────────────────────────────┐
             ▼                                                                   ▼
   [Local Catalog Mode]                                                [Global TMDb Mode]
   (9,742 MovieLens Titles)                                    (All Global Titles / Indian Cinema)
             │                                                                   │
             │                                                                   ▼
             │                                                   [TMDb Global Search Service]
             │                                                    - tmdb_search.py
             │                                                    - /search/movie & /discover/movie
             │                                                    - Multi-language & Industry filters
             │                                                    - In-memory / TTL Cache
             │                                                                   │
             └─────────────────────────────────┬─────────────────────────────────┘
                                               │
                                               ▼
                             [Unified Movie Data Model (Entity)]
                              - tmdb_id, movielens_id, imdb_id
                              - title, original_title, language, country
                              - MovieLens stats vs. TMDb stats (Strict Provenance)
                                               │
                                               ▼
                              [Hybrid Recommendation Engine]
                                   (hybrid_recommender.py)
                                               │
                     ┌─────────────────────────┴─────────────────────────┐
                     ▼                                                   ▼
         [Branch A: Linked Movie]                            [Branch B: TMDb-Only Movie]
   (Has MovieLens ID via links.csv)                     (Absent from MovieLens snapshot)
                     │                                                   │
   - Use precomputed sparse CSR row                     - Construct dynamic feature soup:
   - 1 x N linear kernel product                        title + overview + genres + keywords
   - Exact cosine similarity match                      - Project into fitted vector space:
   - Optional Bayesian quality boost                      query_vec = vectorizer.transform([soup])
   - True MovieLens tag overlap                         - Compute 1 x N linear kernel against
                                                          9,742 candidate feature matrix!
                                                        - Optional blend with TMDb /similar API
                                               │
                                               ▼
                             [Unified Recommendation Assembler]
                              - Strictly descending ranking sort
                              - Provenance & Metric Labeling
                              - Interpretable Match Rationale Box
                              - Real Posters (TMDb CDN) + India OTT
```

---

## 4. Phase-by-Phase Technical Specifications

### Phase 1: Global Movie Search (`tmdb_search.py` & `tmdb_client.py`)
- **Endpoints**:
  - `GET /3/search/movie`: Query string, page, include_adult=false, language, primary_release_year.
  - `GET /3/discover/movie`: Used for structured country, language, and release year filters:
    - `with_origin_country`: Filter by country of production (e.g. `IN` for India, `US` for United States, `KR` for South Korea, `JP` for Japan).
    - `with_original_language`: Filter by original audio language:
      - Indian Languages: `hi` (Hindi), `ta` (Tamil), `te` (Telugu), `ml` (Malayalam), `kn` (Kannada), `bn` (Bengali).
      - Global Languages: `en` (English), `ko` (Korean), `ja` (Japanese).
    - `sort_by`: `popularity.desc`, `vote_count.desc`.
- **Indian Cinema Discovery Distinction**:
  - Note: Country of origin (`IN`) and original language (`hi`, `ta`, etc.) are completely independent attributes. An Indian movie may have Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali, or English audio.
  - User-facing labels such as *Bollywood* (Hindi / India), *Tollywood* (Telugu / India), *Kollywood* (Tamil / India), *Mollywood* (Malayalam / India), and *Sandalwood* (Kannada / India) are helpful presentation aliases mapped to valid `with_origin_country=IN` and `with_original_language` parameters, but are **never** treated or documented as guaranteed native TMDb industry classifications.
- **Search Capabilities**:
  - Exact & partial title matches.
  - Dual title rendering (English title + Original title if distinct).
  - Result deduplication by TMDb `id`.
  - Client-side memoization using `@st.cache_data(ttl=3600)`.
  - Resilient timeout handling (4.0s) with clear offline feedback.

### Phase 2: Unified Movie Data Model (`unified_data_model.py`)
A single normalized representation encapsulating source provenance and strict data integrity guarantees:
```python
@dataclass
class UnifiedMovie:
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

    # Provenance Tracking: "movielens_linked" | "tmdb_global" | "movielens_only"
    metadata_source: str = "tmdb_global"

    # Collaborative Data Flag
    collaborative_data_available: bool = False
```


### Phase 3: Hybrid Recommendation Engine (`hybrid_recommender.py`)
> [!IMPORTANT]
> **Candidate Pool Evolution & Terminology Precision (Architecture V2.2 - Final Approval Edition):**
> 1. **Baseline Model Terminology**: The existing MovieLens recommender is strictly a **content-based TF-IDF engine with an empirical Bayesian rating-quality boost**, NOT collaborative filtering.
> 2. **Item-Item Collaborative Filtering**: In Stage 3, a standalone item-item collaborative similarity matrix will be trained on `data/ml-latest-small/ratings.csv` ($100,836$ ratings across $610$ users) using Pearson correlation with Bayesian significance shrinkage ($\tau = 5$). This provides behavioral co-occurrence signals for MovieLens-linked items, but does NOT constitute user-personalized recommendations (CineMatch AI uses an item-seeded query paradigm).
> 3. **Canonical Identity Resolution Layer & IMDb Alias Bridge**: Multi-identifier graph (`movieId`, `links.csv` `tmdbId`, `imdbId`, and TMDb API ID) assigning an immutable `canonical_id` (`tmdb:{id}` or `imdb:{id}`). Records lacking TMDb IDs in MovieLens are dynamically bridged to incoming TMDb records via matching `imdb_id`. Movies existing in both catalogs are fused into a single entity (`metadata_source = "movielens_linked"`), unlocking collaborative signals and posters, while remakes (*Total Recall* 1990 vs 2012) and duplicate titles (*Emma* 1996 vs 2020) remain distinct nodes.
> 4. **Deterministic Global 100-Candidate Allocation Algorithm**:
>    - Channel Extraction Quotas: ML Content (40), ML Collaborative (20), TMDb Recommendations (25), TMDb Similar (25), TMDb Regional Discover (20).
>    - Multi-source candidates are merged and attributed.
>    - Allocation of 100-slot budget: Phase A (Multi-source intersections) $\rightarrow$ Phase B (Source-diversity floor reservations) $\rightarrow$ Phase C (Residual filling by best channel score and canonical ID).
>    - Mathematically guarantees that the candidate pool **never exceeds 100 items**.
> 5. **Strict 4.5-Second Shared Network Deadline Model (Policy A)**:
>    - Global 4.5-second hard wall clock limit across all TMDb network operations.
>    - Attempt 1 timeout capped at 3.0s; eligible retries share remaining time budget ($\le 1.25$s).
>    - On HTTP 429 or deadline expiration, immediately triggers Local Catalog Mode fallback ($< 10$ ms).

#### A. Query Projection Strategy & Empirically Calibrated Minimum-Evidence Policy
When an external movie (e.g. *Avengers: Endgame* or *RRR*) is selected:
1. Transform TMDb metadata strictly adhering to the training feature soup in `recommender.py`:
   - Title: Words extracted via `\b\w+\b`
   - Narrative Genres: `g.replace("-", "").replace(" ", "")` (repeated 2x)
   - Technical Format: `format_imax` (isolated, 1x)
   - Thematic Tags: TMDb keywords lowercased and joined into `tags_tokenized` (repeated 2x)
   - Era: `decade_{year // 10 * 10}s` (post-2019 yields `decade_2020s`, safely ignored as OOV)
   - *Note*: Raw plot text and language codes are omitted to prevent feature space distortion against the fitted $12,000$-feature vectorizer.
2. Empirically Calibrated Minimum-Evidence Policy:
   - Tokens classified reproducibly via model IDF weights: Generic ($\text{IDF} < 4.0$) vs Discriminative ($\text{IDF} \ge 4.0$).
   - Candidate must share $\ge 1$ discriminative token or $\ge 2$ generic tokens with query.
   - Non-zero vector norm verified; similarity must exceed null threshold $\gamma_{\text{sim}}$ calibrated strictly on validation data ($R_{\text{val}}$).
3. Project `soup` directly into the existing $12,000$-dimensional vector space:
   ```python
   query_vec = self.tfidf_vectorizer.transform([soup])  # 1 x 12,000 sparse CSR
   ```

#### B. Collaborative & Hybrid Score Integration (Relevance-First Policy)
All constituent metrics are normalized to $[0.0, 1.0]$:
- $\text{Sim}_{\text{content}}(q, c) \in [0.0, 1.0]$: Cosine similarity of $\ell_2$-normalized non-negative TF-IDF coordinates.
- $\text{Sim}_{\text{collab}}(q, c) \in [0.0, 1.0]$: Shrunk Pearson correlation rectified via $\max(0.0, s_{\text{shrunk}})$.
- $\text{Quality}_{\text{source}}(c) \in [0.0, 1.0]$: Evaluated within candidate provenance (MovieLens Bayesian mean $\frac{W}{5.0}$ vs TMDb damped vote score $\frac{\text{vote\_avg}}{10} \cdot \frac{v}{v + 500}$). TMDb ratings are never converted or conflated with MovieLens ratings.

**Adaptive Routing**:
- **Path A (Collaborative Available)**:
  $$\text{Score}(q, c) = (1 - \beta) \left[ \alpha \cdot \text{Sim}_{\text{content}}(q, c) + (1 - \alpha) \cdot \text{Sim}_{\text{collab}}(q, c) \right] + \beta \cdot \text{Quality}_{\text{ML}}(c)$$
- **Path B (Cold-Start / TMDb-only)**:
  $$\text{Score}(q, c) = (1 - \beta) \cdot \text{Sim}_{\text{content}}(q, c) + \beta \cdot \text{Quality}_{\text{source}}(c)$$
*(Weights $\alpha=0.65$ and $\beta=0.10$ are initial experimental hypotheses to be calibrated via validation-set grid search)*.
- **Relevance-First Gating**: Candidates failing the Content Relevance Gate receive $\text{Score} = 0.0$. Collaborative boosting is dynamically attenuated when content similarity $< 0.40$.

#### C. Deterministic Descending Sort
Sort key eliminates process-salted `hash()` in favor of deterministic multi-tier tie-breakers:
$$\text{SortKey}(c) = \left( \text{Score}_{\text{unrounded}}, \; \text{Sim}_{\text{content}}, \; \text{Quality}_{\text{source}}, \; v_c, \; -\text{LexicalKey}(\text{canonical\_id}) \right)$$
Sanitizes `NaN` and `Inf` to `-1.0`.

#### D. Recommendation Explanations
Explanations are dynamically generated based on verifiable feature intersections:
- *"Shares genres: Action, Sci-Fi"*
- *"Matching themes: superhero, time-travel"*
- *"Behavioral affinity: Frequently co-rated by viewers of The Dark Knight"*
- *"Top matching vector terms: marvel, avengers, space"*

#### E. Item-Seeded Offline Evaluation Protocol
- Matches production item-seeded discovery flow: Evaluates query seed $q_u$ (selected as the most recent positively rated training movie for eligible users in $H_u^{\text{train}}$).
- The query seed $q_u$ is strictly excluded from candidate ranking.
- Precision@K, Recall@K, NDCG@K, and MRR@K evaluate retrieval against held-out future positive interactions ($G_u^{\text{test}}$ where $r_{ui} \ge 4.0$).
- 3-way temporal split (70% train / 15% validation / 15% test) guarantees complete isolation of collaborative matrices, quality priors, and popularity baselines.




### Phase 4: Real Movie Posters & Details
- Retains existing tested CDN URL generator: `https://image.tmdb.org/t/p/w342/...`.
- Retains verified SVG fallback badges: `OFFICIAL POSTER UNAVAILABLE`.
- Dynamic alt text ensures WCAG accessibility compliance.

### Phase 5: India OTT Availability
- Queries `/3/movie/{id}/watch/providers` with `region="IN"`.
- Displays categorized pills:
  - `▶️ Available to Stream in India`
  - `🏷️ Available to Rent in India`
  - `💳 Available to Buy in India`
- Displays verified deep link to TMDb watch page.
- Employs exact required fallback: *"Streaming availability information is currently unavailable for this title in India."*

### Phase 6: User Interface Integration
- Retains light theme, crisp typography, and glassmorphism cards.
- Search Switcher:
  - **Catalog Search**: Fast autocomplete across 9,742 MovieLens benchmark titles.
  - **Global Movie Search**: Live API search across global and Indian cinema with filters.
- Filter Bar for Global Discovery:
  - **Industries / Regions**: All Countries, India (Bollywood & South Indian Cinema), Hollywood (US), South Korea, Japan.
  - **Languages**: Hindi, Tamil, Telugu, Malayalam, Kannada, English, Korean, Japanese.
- Responsive Layout: 0px horizontal overflow guaranteed across Desktop (1280x800), Tablet (768x1024), and Mobile (375x667).

---

## 5. File-Level Change Plan

| Phase | File | Action | Purpose |
| :--- | :--- | :---: | :--- |
| **0** | `docs/GLOBAL_HYBRID_IMPLEMENTATION_PLAN.md` | **Create** | Baseline documentation, data models, and implementation blueprint. |
| **1** | `tmdb_search.py` | **Create** | Global search, discover endpoint wrapper, language/country filters, pagination. |
| **1** | `tests/test_global_search.py` | **Create** | Unit tests for global search, Indian cinema filters, error handling, mock pagination. |
| **2** | `unified_data_model.py` | **Create** | Normalized dataclass reconciling MovieLens and TMDb records with source tags. |
| **2** | `tests/test_unified_model.py` | **Create** | Unit tests for data normalization, missing values, duplicate handling, serialization. |
| **3** | `hybrid_recommender.py` | **Create** | Hybrid recommendation engine with on-the-fly vector projection and hybrid scoring. |
| **3** | `tests/test_hybrid_recommender.py` | **Create** | Tests for TMDb-only queries (*Avengers*, *RRR*), linked queries, and ranking integrity. |
| **4 & 5** | `tmdb_client.py` | **Enhance** | Integrate search & provider caching utilities without breaking existing interfaces. |
| **6** | `app.py` | **Enhance** | Add Global Search UI, discovery filters, unified cards, and hybrid explanations. |
| **7** | `tests/test_regression.py` | **Create** | End-to-end regression verifying all 42 legacy tests + new hybrid features. |
| **8** | `README.md` | **Enhance** | Architecture diagrams, TMDb attribution, hybrid explanations, usage guide. |
| **8** | `docs/GLOBAL_HYBRID_FINAL_REPORT.md` | **Create** | Comprehensive final technical audit and delivery report. |

---

## 6. Migration Risks & Mitigation Strategy

| Risk | Impact | Likelihood | Mitigation Strategy |
| :--- | :---: | :---: | :--- |
| **TMDb Rate Limiting (HTTP 429)** | High | Low | Client-side 24-hour TTL caching with `@st.cache_data`; exponential backoff; graceful fallback to local MovieLens catalog. |
| **ISP DNS Poisoning / IPv6 Timeout in India** | Critical | High | Retain verified `_tmdb_getaddrinfo` socket hook directing requests to CloudFront IPv4 (`13.224.245.63`). |
| **Vocabulary Mismatch for Modern Films** | Medium | Medium | Vectorizer uses sublinear TF unigrams and bigrams. Title and genre tokens ensure high overlap even if modern slang is out-of-vocabulary. |
| **Conflation of Rating Scales** | High | Low | Strict type separation in `UnifiedMovie`. UI explicitly renders separate badges for MovieLens (⭐ 4.2/5.0) and TMDb (⭐ 8.4/10.0). |
| **Streamlit UI Layout Degradation** | Medium | Low | Viewport checks at 1280x800, 768x1024, and 375x667 using automated Playwright tests before each stage sign-off. |

---

## 7. Rollback & Verification Strategy

- **Granular Git Discipline**: No git commits or branch switches will be made until explicit user authorization is provided.
- **Independent Component Testing**: Each phase introduces standalone modules (`tmdb_search.py`, `unified_data_model.py`, `hybrid_recommender.py`) that do not disrupt `recommender.py` or existing tests.
- **Rollback Procedure**: In the event of an unresolvable defect during any stage, reverting the single affected file immediately restores 100% functionality to the baseline.
- **Continuous Test Gate**: Every stage execution must conclude with `pytest -v` confirming that 42/42 baseline tests and all new phase tests pass.

---

## 8. Implementation Order & Approval Gates

- **Stage 0**: Repository Baseline & Architecture Plan (**Completed — Documented in this file**).
- **Stage 1**: Global Search and TMDb Catalog Integration (`tmdb_search.py` + tests) — **COMPLETED & APPROVED** (21/21 tests passed, live queries verified).
- **Stage 2**: Unified Movie Data Model (`unified_data_model.py` + tests) — **COMPLETED** (13/13 tests passed, 76/76 full test suite passed; pending Stage 3 approval).
- **Stage 3**: Hybrid Recommendation Engine (`hybrid_recommender.py` + tests) — **ARCHITECTURE & DESIGN COMPLETE** (Detailed in [`docs/STAGE_3_HYBRID_ARCHITECTURE.md`](file:///C:/Projects/Movie%20Recommondation%20System/docs/STAGE_3_HYBRID_ARCHITECTURE.md); Implementation awaiting user approval).
- **Stage 4**: Posters and Movie Details (`tmdb_client.py` enhancement).
- **Stage 5**: OTT Availability (Where to Watch in India).
- **Stage 6**: Streamlit UI Integration (`app.py` update).
- **Stage 7**: Full Regression & Security Verification (All test suites).
- **Stage 8**: Final Documentation & Report (`README.md` & `GLOBAL_HYBRID_FINAL_REPORT.md`).

