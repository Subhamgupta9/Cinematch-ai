"""
Unit Tests for TMDb Global Movie Search & Catalog Retrieval Module
------------------------------------------------------------------
Validates global movie search, Indian regional cinema discovery, pagination,
deduplication, error handling, rate limiting, and response normalization.
"""

import pytest
from unittest.mock import patch, MagicMock
import requests

from tmdb_search import (
    normalize_tmdb_movie,
    search_movies_global,
    discover_movies_global,
    fetch_movie_details_global,
    _search_cache,
)


@pytest.fixture(autouse=True)
def clear_cache_and_setup_env(monkeypatch):
    """Clears in-memory cache and provides a default mock key for all tests."""
    _search_cache.clear()
    monkeypatch.setenv("TMDB_API_KEY", "mock_test_key_12345")


def test_normalize_tmdb_movie_complete_fields():
    raw = {
        "id": 299534,
        "title": "Avengers: Endgame",
        "original_title": "Avengers: Endgame",
        "overview": "After the devastating events of Avengers: Infinity War...",
        "release_date": "2019-04-24",
        "original_language": "en",
        "origin_country": ["US"],
        "genre_ids": [12, 878, 28],
        "poster_path": "/or06FN3Dka5tukK1e9sl16pB3iy.jpg",
        "backdrop_path": "/7RyHsO4yDXtBv1zUU3mTpHeQ0d5.jpg",
        "vote_average": 8.263,
        "vote_count": 24890,
        "popularity": 89.456,
        "adult": False,
    }

    norm = normalize_tmdb_movie(raw)
    assert norm["tmdb_id"] == "299534"
    assert norm["title"] == "Avengers: Endgame"
    assert norm["release_year"] == 2019
    assert norm["release_date"] == "2019-04-24"
    assert norm["original_language"] == "en"
    assert norm["origin_country"] == ["US"]
    assert "Action" in norm["genre_names"]
    assert "Sci-Fi" in norm["genre_names"]
    assert "Adventure" in norm["genre_names"]
    assert "w342" in norm["poster_url"]
    assert "w780" in norm["backdrop_url"]
    assert norm["tmdb_vote_average"] == 8.3
    assert norm["tmdb_vote_count"] == 24890
    assert norm["adult"] is False


def test_normalize_tmdb_movie_missing_and_null_fields():
    raw = {
        "id": 12345,
        "title": None,
        "original_title": None,
        "overview": None,
        "release_date": "invalid-date",
        "genre_ids": ["not-a-number", None],
        "poster_path": None,
        "backdrop_path": "",
        "vote_average": None,
        "vote_count": None,
    }

    norm = normalize_tmdb_movie(raw)
    assert norm["tmdb_id"] == "12345"
    assert norm["title"] == "Untitled"
    assert norm["overview"] == ""
    assert norm["release_year"] is None
    assert norm["genre_ids"] == []
    assert norm["genre_names"] == []
    assert norm["poster_url"] is None
    assert norm["backdrop_url"] is None
    assert norm["tmdb_vote_average"] == 0.0
    assert norm["tmdb_vote_count"] == 0


def test_search_movies_global_empty_query():
    for empty_q in ["", "   ", None]:
        res = search_movies_global(empty_q)
        assert res["success"] is True
        assert res["status"] == "empty_query"
        assert res["results"] == []
        assert res["total_results"] == 0


def test_search_movies_global_missing_api_key(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "")
    with patch("streamlit.secrets", {}, create=True):
        res = search_movies_global("Inception")
        assert res["success"] is False
        assert res["status"] == "missing_key"
        assert res["results"] == []


def test_search_avengers_endgame_mock():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "page": 1,
        "total_pages": 1,
        "total_results": 1,
        "results": [
            {
                "id": 299534,
                "title": "Avengers: Endgame",
                "original_title": "Avengers: Endgame",
                "overview": "After the devastating events...",
                "release_date": "2019-04-24",
                "genre_ids": [12, 878, 28],
                "vote_average": 8.3,
                "vote_count": 24000,
                "poster_path": "/or06FN3Dka5tukK1e9sl16pB3iy.jpg",
            }
        ],
    }

    with patch("requests.get", return_value=mock_resp):
        res = search_movies_global("Avengers: Endgame")
        assert res["success"] is True
        assert res["status"] == "success"
        assert len(res["results"]) == 1
        movie = res["results"][0]
        assert movie["tmdb_id"] == "299534"
        assert movie["title"] == "Avengers: Endgame"
        assert movie["release_year"] == 2019


def test_search_rrr_mock():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "page": 1,
        "total_pages": 1,
        "total_results": 1,
        "results": [
            {
                "id": 579974,
                "title": "RRR",
                "original_title": "రౌద్రం రణం రుధిరం",
                "overview": "A fictional history of two legendary revolutionaries...",
                "release_date": "2022-03-24",
                "original_language": "te",
                "origin_country": ["IN"],
                "genre_ids": [28, 18],
                "vote_average": 7.8,
                "vote_count": 1600,
                "poster_path": "/wE0noFU2EToSHUrHz11Qcyv0uqN.jpg",
            }
        ],
    }

    with patch("requests.get", return_value=mock_resp):
        res = search_movies_global("RRR")
        assert res["success"] is True
        assert res["status"] == "success"
        assert len(res["results"]) == 1
        movie = res["results"][0]
        assert movie["tmdb_id"] == "579974"
        assert movie["title"] == "RRR"
        assert movie["original_title"] == "రౌద్రం రణం రుధిరం"
        assert movie["original_language"] == "te"
        assert movie["origin_country"] == ["IN"]


def test_search_3_idiots_mock():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "page": 1,
        "total_pages": 1,
        "total_results": 1,
        "results": [
            {
                "id": 20453,
                "title": "3 Idiots",
                "original_title": "3 Idiots",
                "overview": "Two friends embark on a quest for a lost buddy...",
                "release_date": "2009-12-25",
                "original_language": "hi",
                "origin_country": ["IN"],
                "genre_ids": [35, 18],
                "vote_average": 8.0,
                "vote_count": 2200,
                "poster_path": "/56whbX1s7xW8W5u3q0d8JkE9v4P.jpg",
            }
        ],
    }

    with patch("requests.get", return_value=mock_resp):
        res = search_movies_global("3 Idiots")
        assert res["success"] is True
        assert len(res["results"]) == 1
        movie = res["results"][0]
        assert movie["tmdb_id"] == "20453"
        assert movie["title"] == "3 Idiots"
        assert movie["original_language"] == "hi"
        assert movie["release_year"] == 2009


def test_search_dangal_mock():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "page": 1,
        "total_pages": 1,
        "total_results": 1,
        "results": [
            {
                "id": 360814,
                "title": "Dangal",
                "original_title": "दंगल",
                "overview": "Former wrestler Mahavir Singh Phogat trains his daughters...",
                "release_date": "2016-12-23",
                "original_language": "hi",
                "origin_country": ["IN"],
                "genre_ids": [18, 28],
                "vote_average": 8.0,
                "vote_count": 900,
                "poster_path": "/x2Ld2F1DqL4M6k8P1.jpg",
            }
        ],
    }

    with patch("requests.get", return_value=mock_resp):
        res = search_movies_global("Dangal")
        assert res["success"] is True
        assert len(res["results"]) == 1
        movie = res["results"][0]
        assert movie["tmdb_id"] == "360814"
        assert movie["title"] == "Dangal"
        assert movie["original_title"] == "दंगल"
        assert movie["release_year"] == 2016


def test_search_partial_title():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "page": 1,
        "total_pages": 1,
        "total_results": 1,
        "results": [{"id": 27205, "title": "Inception", "release_date": "2010-07-15"}],
    }

    with patch("requests.get", return_value=mock_resp):
        res = search_movies_global("Incept")
        assert res["success"] is True
        assert len(res["results"]) == 1
        assert res["results"][0]["title"] == "Inception"


def test_search_movies_global_empty_results():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"page": 1, "total_pages": 0, "total_results": 0, "results": []}

    with patch("requests.get", return_value=mock_resp):
        res = search_movies_global("NonExistentMovieTitleXYZ987654")
        assert res["success"] is True
        assert res["results"] == []
        assert res["total_results"] == 0


def test_search_movies_global_pagination():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "page": 2,
        "total_pages": 5,
        "total_results": 95,
        "results": [{"id": 999, "title": "Page 2 Movie", "release_date": "2020-01-01"}],
    }

    with patch("requests.get", return_value=mock_resp) as mock_get:
        res = search_movies_global("Matrix", page=2)
        assert res["success"] is True
        assert res["page"] == 2
        assert res["total_pages"] == 5
        # Verify page parameter passed
        call_params = mock_get.call_args[1]["params"]
        assert call_params["page"] == 2


def test_search_movies_global_deduplication():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {"id": 155, "title": "The Dark Knight", "release_date": "2008-07-16"},
            {"id": 155, "title": "The Dark Knight (Duplicate)", "release_date": "2008-07-16"},
            {"id": 27205, "title": "Inception", "release_date": "2010-07-15"},
        ]
    }

    with patch("requests.get", return_value=mock_resp):
        res = search_movies_global("Batman")
        assert res["success"] is True
        assert len(res["results"]) == 2
        ids = [m["tmdb_id"] for m in res["results"]]
        assert ids == ["155", "27205"]


def test_search_movies_global_adult_exclusion():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {"id": 1, "title": "Family Film", "adult": False},
            {"id": 2, "title": "Adult Film", "adult": True},
        ]
    }

    with patch("requests.get", return_value=mock_resp):
        res = search_movies_global("Movie", include_adult=False)
        assert res["success"] is True
        assert len(res["results"]) == 1
        assert res["results"][0]["title"] == "Family Film"


def test_search_movies_global_http_errors():
    for code in [401, 429, 500]:
        mock_resp = MagicMock()
        mock_resp.status_code = code
        with patch("requests.get", return_value=mock_resp):
            res = search_movies_global("Test")
            assert res["success"] is False
            assert res["status"] == "api_error"
            assert res["results"] == []


def test_search_movies_global_timeout():
    with patch("requests.get", side_effect=requests.Timeout("Connection timed out")):
        res = search_movies_global("Test")
        assert res["success"] is False
        assert res["status"] == "timeout"
        assert res["results"] == []


def test_search_movies_global_malformed_json():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = ["not-a-dict"]

    with patch("requests.get", return_value=mock_resp):
        res = search_movies_global("Test")
        assert res["success"] is False
        assert res["status"] == "api_error"


def test_discover_movies_indian_cinema_filters():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "page": 1,
        "total_pages": 10,
        "total_results": 200,
        "results": [{"id": 360814, "title": "Dangal", "original_language": "hi"}],
    }

    with patch("requests.get", return_value=mock_resp) as mock_get:
        res = discover_movies_global(origin_country="IN", original_language="hi", year=2016)
        assert res["success"] is True
        params = mock_get.call_args[1]["params"]
        assert params["with_origin_country"] == "IN"
        assert params["with_original_language"] == "hi"
        assert params["primary_release_year"] == 2016


def test_discover_movies_south_indian_languages():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": []}

    languages = ["te", "ta", "ml", "kn", "bn"]
    for lang in languages:
        with patch("requests.get", return_value=mock_resp) as mock_get:
            res = discover_movies_global(origin_country="IN", original_language=lang)
            assert res["success"] is True
            params = mock_get.call_args[1]["params"]
            assert params["with_origin_country"] == "IN"
            assert params["with_original_language"] == lang


def test_discover_movies_international_filters():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": []}

    # Korean Cinema
    with patch("requests.get", return_value=mock_resp) as mock_get:
        discover_movies_global(origin_country="KR", original_language="ko")
        params = mock_get.call_args[1]["params"]
        assert params["with_origin_country"] == "KR"
        assert params["with_original_language"] == "ko"

    # Japanese Cinema
    with patch("requests.get", return_value=mock_resp) as mock_get:
        discover_movies_global(origin_country="JP", original_language="ja")
        params = mock_get.call_args[1]["params"]
        assert params["with_origin_country"] == "JP"
        assert params["with_original_language"] == "ja"


def test_fetch_movie_details_global_with_keywords():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": 299534,
        "title": "Avengers: Endgame",
        "imdb_id": "tt4154796",
        "keywords": {
            "keywords": [
                {"id": 9715, "name": "superhero"},
                {"id": 180547, "name": "marvel cinematic universe"},
            ]
        },
    }

    with patch("requests.get", return_value=mock_resp):
        details = fetch_movie_details_global("299534")
        assert details is not None
        assert details["tmdb_id"] == "299534"
        assert details["imdb_id"] == "tt4154796"
        assert "superhero" in details["keywords"]
        assert "marvel cinematic universe" in details["keywords"]


def test_search_cache_in_memory():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [{"id": 100, "title": "Cached Film"}]
    }

    with patch("requests.get", return_value=mock_resp) as mock_get:
        res1 = search_movies_global("Cached Film")
        assert mock_get.call_count == 1

        # Second call with identical arguments must hit cache without triggering requests.get
        res2 = search_movies_global("Cached Film")
        assert mock_get.call_count == 1
        assert res1 == res2


def test_search_cache_keys_contain_no_api_key(monkeypatch):
    """Verify that in-memory cache keys do not contain API keys or credentials."""
    secret_key = "super_secret_tmdb_key_xyz999"
    monkeypatch.setenv("TMDB_API_KEY", secret_key)
    _search_cache.clear()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [{"id": 100, "title": "Security Film"}]
    }

    with patch("requests.get", return_value=mock_resp):
        search_movies_global("Security Film", api_key=secret_key)
        discover_movies_global(api_key=secret_key)

    for cache_key in _search_cache.keys():
        assert secret_key not in cache_key
        assert secret_key[:6] not in cache_key
        assert "super_secret" not in cache_key

