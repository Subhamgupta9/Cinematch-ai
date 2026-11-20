"""
Automated Test Suite: Indian Cinema Discovery & Global Catalog Integration
--------------------------------------------------------------------------
Validates the complete discovery and recommendation pipeline for Indian regional cinema
(Bollywood, Tollywood, Kollywood, Mollywood, Sandalwood, Bengali, Marathi) and international
post-2018 titles across TMDb and MovieLens catalogs.

Coverage includes:
1. All 7 target queries: RRR, 3 Idiots, Dangal, Baahubali, Jawan, K.G.F: Chapter 1, Avengers: Endgame.
2. Indian regional languages: Telugu, Hindi, Tamil, Malayalam, Kannada, Bengali, Marathi.
3. Industry presets validation and parameter mapping.
4. Post-2018 international releases and non-Indian cinema discovery.
5. Missing API key & network error handling (guaranteeing honest user notices, never silent fallback).
6. Canonical deduplication and remake separation (e.g. Total Recall 1990 vs 2012).
7. Cross-catalog identity resolution (MovieLens-linked vs TMDb-only).
8. Dynamic TF-IDF content projection via MovieRecommender.recommend_tmdb().
"""

import pytest
from unittest.mock import patch, MagicMock
from tmdb_search import (
    search_movies_global,
    discover_movies_global,
    fetch_movie_details_global,
    SUPPORTED_LANGUAGES,
    INDUSTRY_PRESETS,
    normalize_tmdb_movie,
    _search_cache,
)
from recommender import MovieRecommender
from unified_data_model import UnifiedMovie


@pytest.fixture(autouse=True)
def clear_cache():
    _search_cache.clear()
    yield
    _search_cache.clear()


@pytest.fixture(scope="module")
def recommender():
    r = MovieRecommender()
    if not r.load_artifacts():
        r.fit()
    return r


# =============================================================================
# 1. Verification of the 7 Target Queries
# =============================================================================

@pytest.mark.parametrize(
    "query, expected_lang, expected_min_year",
    [
        ("RRR", "te", 2022),
        ("3 Idiots", "hi", 2009),
        ("Dangal", "hi", 2016),
        ("Baahubali", "te", 2015),
        ("Jawan", "hi", 2023),
        ("K.G.F: Chapter 1", "kn", 2018),
        ("Avengers: Endgame", "en", 2019),
    ],
)
def test_seven_target_queries_mocked(query, expected_lang, expected_min_year):
    """
    Verifies that search_movies_global correctly processes and parses each of the 7 target queries.
    Uses robust mocked TMDb payloads to ensure deterministic, network-independent CI execution.
    """
    mock_payload = {
        "page": 1,
        "total_pages": 1,
        "total_results": 1,
        "results": [
            {
                "id": 12345,
                "title": query,
                "original_title": f"{query} Original",
                "overview": f"A landmark cinematic work: {query}.",
                "release_date": f"{expected_min_year}-04-26",
                "original_language": expected_lang,
                "origin_country": ["IN"] if expected_lang != "en" else ["US"],
                "genre_ids": [28, 18],
                "poster_path": f"/poster_{query.replace(' ', '_').lower()}.jpg",
                "backdrop_path": f"/backdrop_{query.replace(' ', '_').lower()}.jpg",
                "vote_average": 8.1,
                "vote_count": 2500,
                "popularity": 75.5,
                "adult": False,
            }
        ],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_resp):
        res = search_movies_global(query, api_key="dummy_test_key")
        assert res["success"] is True
        assert res["status"] == "success"
        assert len(res["results"]) == 1

        top = res["results"][0]
        assert top["title"] == query
        assert top["original_language"] == expected_lang
        assert top["release_year"] == expected_min_year
        assert top["poster_url"].startswith("https://image.tmdb.org/t/p/w342/")
        assert top["genre_names"] == ["Action", "Drama"]


# =============================================================================
# 2. Indian Regional Cinema & Language Coverage
# =============================================================================

def test_marathi_and_regional_languages_supported():
    """Verifies that Marathi, Bengali, Kannada, Malayalam, Tamil, Telugu, and Hindi are in SUPPORTED_LANGUAGES."""
    required_indian_langs = ["hi", "te", "ta", "ml", "kn", "bn", "mr"]
    for lang in required_indian_langs:
        assert lang in SUPPORTED_LANGUAGES, f"Expected language '{lang}' to be in SUPPORTED_LANGUAGES"
        assert SUPPORTED_LANGUAGES[lang], f"Language label for '{lang}' must not be empty"

    assert "mr" in SUPPORTED_LANGUAGES
    assert "Marathi" in SUPPORTED_LANGUAGES["mr"]


def test_industry_presets_coverage():
    """Verifies that all major Indian film industries are mapped accurately in INDUSTRY_PRESETS."""
    expected_presets = [
        "All Cinema",
        "Indian Cinema (All Languages)",
        "Bollywood (Hindi / India)",
        "Tollywood (Telugu / India)",
        "Kollywood (Tamil / India)",
        "Mollywood (Malayalam / India)",
        "Sandalwood (Kannada / India)",
        "Bengali Cinema (India)",
        "Marathi Cinema (India)",
        "Hollywood (United States)",
        "Korean Cinema",
        "Japanese Cinema",
    ]
    for preset in expected_presets:
        assert preset in INDUSTRY_PRESETS, f"Expected preset '{preset}' in INDUSTRY_PRESETS"
        mapping = INDUSTRY_PRESETS[preset]
        assert "country" in mapping
        assert "language" in mapping

    # Verify specific regional parameter mappings
    assert INDUSTRY_PRESETS["Marathi Cinema (India)"] == {"country": "IN", "language": "mr"}
    assert INDUSTRY_PRESETS["Tollywood (Telugu / India)"] == {"country": "IN", "language": "te"}
    assert INDUSTRY_PRESETS["Sandalwood (Kannada / India)"] == {"country": "IN", "language": "kn"}
    assert INDUSTRY_PRESETS["Mollywood (Malayalam / India)"] == {"country": "IN", "language": "ml"}


def test_discover_movies_regional_parameters():
    """Verifies that discover_movies_global correctly constructs query parameters for regional cinema."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"page": 1, "total_pages": 1, "total_results": 0, "results": []}

    with patch("requests.get", return_value=mock_resp) as mock_get:
        discover_movies_global(
            origin_country="IN",
            original_language="mr",
            api_key="test_api_key",
        )
        assert mock_get.called
        call_params = mock_get.call_args[1]["params"]
        assert call_params["with_origin_country"] == "IN"
        assert call_params["with_original_language"] == "mr"
        assert call_params["api_key"] == "test_api_key"


# =============================================================================
# 3. Post-2018 Releases & International Cinema
# =============================================================================

def test_post_2018_release_year_handling():
    """Verifies that post-September 2018 films have correctly extracted integer years."""
    raw = {
        "id": 99999,
        "title": "Post 2018 Masterpiece",
        "release_date": "2024-03-15",
        "original_language": "hi",
        "genre_ids": [18],
    }
    norm = normalize_tmdb_movie(raw)
    assert norm["release_year"] == 2024
    assert norm["release_date"] == "2024-03-15"


# =============================================================================
# 4. Honest Error Handling & Missing Key Protection
# =============================================================================

def test_missing_api_key_honest_notice():
    """Verifies that when TMDb API key is missing, an explicit notice is returned (never silent fallback)."""
    with patch("tmdb_search.get_tmdb_api_key", return_value=None):
        res = search_movies_global("RRR", api_key=None)
        assert res["success"] is False
        assert res["status"] == "missing_key"
        assert "TMDb API key is not configured" in res["message"]
        assert res["results"] == []


def test_api_network_timeout_honest_notice():
    """Verifies that network timeouts produce an explicit timeout status and honest error message."""
    import requests
    with patch("requests.get", side_effect=requests.Timeout("Connection timed out")):
        res = search_movies_global("Dangal", api_key="dummy_key")
        assert res["success"] is False
        assert res["status"] == "timeout"
        assert "timed out" in res["message"]


def test_api_http_401_unauthorized():
    """Verifies that invalid credentials return an explicit API error."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    with patch("requests.get", return_value=mock_resp):
        res = search_movies_global("Jawan", api_key="invalid_key")
        assert res["success"] is False
        assert res["status"] == "api_error"
        assert "Invalid TMDb API credentials" in res["message"]


# =============================================================================
# 5. Canonical Deduplication & Remake Separation
# =============================================================================

def test_search_results_deduplication():
    """Verifies that results with the exact same TMDb ID are strictly deduplicated."""
    mock_payload = {
        "page": 1,
        "total_pages": 1,
        "total_results": 2,
        "results": [
            {"id": 579974, "title": "RRR", "release_date": "2022-03-24", "genre_ids": [28]},
            {"id": 579974, "title": "RRR (Duplicate)", "release_date": "2022-03-24", "genre_ids": [28]},
        ],
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_resp):
        res = search_movies_global("RRR", api_key="key")
        assert res["success"] is True
        assert len(res["results"]) == 1
        assert res["results"][0]["tmdb_id"] == "579974"


def test_remake_separation_total_recall():
    """Verifies that remakes (e.g. Total Recall 1990 vs 2012) remain separate entities."""
    mock_payload = {
        "page": 1,
        "total_pages": 1,
        "total_results": 2,
        "results": [
            {"id": 861, "title": "Total Recall", "release_date": "1990-06-01", "genre_ids": [28, 878]},
            {"id": 64635, "title": "Total Recall", "release_date": "2012-08-02", "genre_ids": [28, 878]},
        ],
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_resp):
        res = search_movies_global("Total Recall", api_key="key")
        assert res["success"] is True
        assert len(res["results"]) == 2
        r1990 = res["results"][0]
        r2012 = res["results"][1]
        assert r1990["tmdb_id"] == "861"
        assert r1990["release_year"] == 1990
        assert r2012["tmdb_id"] == "64635"
        assert r2012["release_year"] == 2012


# =============================================================================
# 6. Dynamic Content Projection via MovieRecommender.recommend_tmdb()
# =============================================================================

def test_recommend_tmdb_rrr_projection(recommender):
    """
    Verifies that an external TMDb payload for RRR (2022) generates coherent MovieLens
    recommendations via dynamic TF-IDF feature projection without modifying the offline model.
    """
    rrr_entity = {
        "tmdb_id": "579974",
        "title": "RRR",
        "release_year": 2022,
        "original_language": "te",
        "origin_country": ["IN"],
        "genre_names": ["Action", "Drama"],
        "keywords": ["revolution", "friendship", "british raj", "freedom fighter"],
        "overview": "A fictional history of two legendary revolutionaries and their journey away from home before they began fighting for their country in the 1920s.",
        "tmdb_vote_average": 7.8,
        "tmdb_vote_count": 1842,
    }

    recs = recommender.recommend_tmdb(rrr_entity, top_n=5)
    assert recs["success"] is True
    assert recs["query_movie"]["title"] == "RRR (2022)"
    assert recs["query_movie"]["metadata_source"] == "tmdb_global"
    assert len(recs["recommendations"]) == 5

    # Check recommendation integrity
    for rec in recs["recommendations"]:
        assert rec["movieId"] > 0
        assert rec["similarity_score"] > 0.0
        exp = rec["explanation"]
        assert "match_score_pct" in exp
        assert exp["metric_type"] == "Cosine Similarity Match Score"
        assert len(exp["explanation_text"]) > 0


def test_recommend_tmdb_avengers_endgame_projection(recommender):
    """
    Verifies that Avengers: Endgame (2019) recommends Avengers: Age of Ultron / Marvel superhero films.
    """
    endgame_entity = {
        "tmdb_id": "299534",
        "title": "Avengers: Endgame",
        "release_year": 2019,
        "original_language": "en",
        "genre_names": ["Action", "Adventure", "Sci-Fi"],
        "keywords": ["superhero", "marvel cinematic universe", "avengers", "time travel"],
        "overview": "After the devastating events of Avengers: Infinity War, the universe is in ruins.",
        "tmdb_vote_average": 8.3,
        "tmdb_vote_count": 24000,
    }

    recs = recommender.recommend(endgame_entity, top_n=5)
    assert recs["success"] is True
    rec_titles = [r["title"] for r in recs["recommendations"]]
    assert any("Avengers" in t for t in rec_titles), f"Expected Avengers in recommendations: {rec_titles}"


def test_recommend_tmdb_self_exclusion_for_linked_movies(recommender):
    """
    Verifies that when a TMDb movie exists in MovieLens (such as 3 Idiots, ML ID 73881),
    it is linked as movielens_linked and strictly excludes itself from its own recommendations.
    """
    three_idiots_entity = {
        "tmdb_id": "20453",
        "title": "3 Idiots",
        "release_year": 2009,
        "original_language": "hi",
        "genre_names": ["Comedy", "Drama"],
        "keywords": ["college", "friendship", "engineering", "student"],
        "overview": "Two friends embark on a quest for a lost buddy.",
        "tmdb_vote_average": 8.0,
        "tmdb_vote_count": 2200,
    }

    recs = recommender.recommend_tmdb(three_idiots_entity, top_n=5)
    assert recs["success"] is True
    assert recs["query_movie"]["metadata_source"] == "movielens_linked"
    assert recs["query_movie"]["movieId"] == 73881

    # Verify self-exclusion
    rec_ids = [r["movieId"] for r in recs["recommendations"]]
    assert 73881 not in rec_ids, "Movie must never recommend itself!"


def test_recommend_tmdb_genre_and_rating_filters(recommender):
    """Verifies that genre filtering and minimum rating filtering apply correctly to TMDb recommendations."""
    dangal_entity = {
        "tmdb_id": "360814",
        "title": "Dangal",
        "release_year": 2016,
        "original_language": "hi",
        "genre_names": ["Action", "Drama"],
        "keywords": ["wrestling", "biography", "female athlete", "father daughter"],
        "overview": "Former wrestler Mahavir Singh Phogat and his two wrestler daughters struggle towards glory at the Commonwealth Games.",
        "tmdb_vote_average": 8.0,
        "tmdb_vote_count": 900,
    }

    recs = recommender.recommend_tmdb(
        dangal_entity,
        top_n=5,
        genre_filter="Drama",
        min_rating=3.0,
    )
    assert recs["success"] is True
    for r in recs["recommendations"]:
        assert "Drama" in r["genre_list"]
        assert r["avg_rating"] >= 3.0


def test_unified_movie_compatibility(recommender):
    """Verifies that a UnifiedMovie instance can be directly passed to MovieRecommender.recommend()."""
    unified = UnifiedMovie(
        tmdb_id="872906",
        title="Jawan",
        release_year=2023,
        original_language="hi",
        origin_country=["IN"],
        genre_names=["Action", "Thriller"],
        keywords=["vigilante", "father son", "corruption"],
        overview="A high-octane action thriller which outlines the emotional journey of a man who is set to rectify the wrongs in the society.",
        tmdb_vote_average=7.2,
        tmdb_vote_count=230,
        metadata_source="tmdb_global",
    )

    recs = recommender.recommend(unified, top_n=5)
    assert recs["success"] is True
    assert len(recs["recommendations"]) == 5
    assert recs["query_movie"]["title"].startswith("Jawan")
