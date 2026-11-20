"""
Unit Tests for TMDb API Client (Posters & Streaming Providers)
--------------------------------------------------------------
Tests graceful fallback, missing keys, invalid IDs, network failure handling,
SVG placeholder generation, and India watch provider parsing across categories.
"""

import os
import urllib.parse
from unittest.mock import patch, MagicMock
import pytest
from tmdb_client import (
    get_tmdb_api_key,
    generate_fallback_poster_svg,
    fetch_movie_poster_path,
    fetch_watch_providers,
)


def test_get_tmdb_api_key_when_empty(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "")
    with patch("streamlit.secrets", {}, create=True):
        key = get_tmdb_api_key()
        assert isinstance(key, str)
        assert key == ""


def test_get_tmdb_api_key_from_env(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "test_mock_key_12345")
    key = get_tmdb_api_key()
    assert key == "test_mock_key_12345"


def test_generate_fallback_poster_svg():
    svg_uri = generate_fallback_poster_svg("The Dark Knight", 2008, ["Action", "Crime"])
    assert svg_uri.startswith("data:image/svg+xml;utf8,")
    unquoted = urllib.parse.unquote(svg_uri)
    assert "The Dark Knight" in unquoted
    assert "2008" in unquoted
    assert "Action" in unquoted
    # Verify explicit disclaimer labels exist in fallback artwork
    assert "OFFICIAL POSTER UNAVAILABLE" in unquoted
    assert "Placeholder Artwork" in unquoted


def test_generate_fallback_poster_svg_xml_escaping():
    malicious_title = '<script>alert("xss")</script>&"test"'
    svg_uri = generate_fallback_poster_svg(malicious_title, 2024, ["<Sci-Fi>", "Action & Adv"])
    unquoted = urllib.parse.unquote(svg_uri)
    assert "<script>" not in unquoted
    assert "&lt;script&gt;" in unquoted
    assert "&amp;" in unquoted


def test_fetch_movie_poster_path_no_key_returns_none(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "")
    with patch("streamlit.secrets", {}, create=True):
        poster = fetch_movie_poster_path("155")
        assert poster is None


def test_fetch_movie_poster_path_invalid_or_missing_id(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "dummy_key")
    assert fetch_movie_poster_path("") is None
    assert fetch_movie_poster_path("0") is None
    assert fetch_movie_poster_path("-155") is None
    assert fetch_movie_poster_path("abc") is None
    assert fetch_movie_poster_path("12.34") is None
    assert fetch_movie_poster_path("tt0468569") is None  # IMDb ID, not TMDb ID


def test_fetch_movie_poster_path_with_mock_api(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "dummy_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"poster_path": "/qJ2tW6WMUDux911r6m7haRef0WH.jpg"}

    with patch("requests.get", return_value=mock_resp):
        url = fetch_movie_poster_path("155")
        assert url == "https://image.tmdb.org/t/p/w342/qJ2tW6WMUDux911r6m7haRef0WH.jpg"


def test_fetch_movie_poster_path_missing_poster_path(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "dummy_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"poster_path": None}

    with patch("requests.get", return_value=mock_resp):
        url = fetch_movie_poster_path("155")
        assert url is None


def test_fetch_movie_poster_path_http_errors(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "dummy_key")
    for error_code in [401, 404, 429, 500]:
        mock_resp = MagicMock()
        mock_resp.status_code = error_code
        with patch("requests.get", return_value=mock_resp):
            url = fetch_movie_poster_path("155")
            assert url is None


def test_fetch_watch_providers_no_key_returns_graceful_message(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "")
    with patch("streamlit.secrets", {}, create=True):
        res = fetch_watch_providers("155", region="IN")
        assert isinstance(res, dict)
        assert res["available"] is False
        assert res["status"] == "missing_key"
        assert "Streaming availability information is currently unavailable for this title in India." in res["message"]


def test_fetch_watch_providers_missing_or_invalid_id(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "dummy_key")
    for bad_id in ["", "0", "-1", "abc", "tt0468569"]:
        res = fetch_watch_providers(bad_id, region="IN")
        assert res["available"] is False
        assert res["status"] == "invalid_id"


def test_fetch_watch_providers_mock_india_data(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "dummy_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": {
            "IN": {
                "link": "https://www.themoviedb.org/movie/155/watch?locale=IN",
                "flatrate": [
                    {"provider_name": "Netflix", "logo_path": "/pbpMk2JmcoNnQwx5JGpXngfoWtp.jpg", "provider_id": 8},
                    {"provider_name": "JioHotstar", "logo_path": "/hotstar.jpg", "provider_id": 122}
                ],
                "rent": [
                    {"provider_name": "Apple TV", "logo_path": "/appletv.jpg", "provider_id": 2}
                ],
                "buy": [
                    {"provider_name": "Google Play Movies", "logo_path": "/gplay.jpg", "provider_id": 3}
                ]
            }
        }
    }

    with patch("requests.get", return_value=mock_resp):
        res = fetch_watch_providers("155", region="IN")
        assert res["available"] is True
        assert res["status"] == "success"
        assert res["region"] == "IN"
        assert len(res["flatrate"]) == 2
        assert res["flatrate"][0]["provider_name"] == "Netflix"
        assert "pbpMk2JmcoNnQwx5JGpXngfoWtp.jpg" in res["flatrate"][0]["logo_url"]
        assert len(res["rent"]) == 1
        assert res["rent"][0]["provider_name"] == "Apple TV"
        assert len(res["buy"]) == 1
        assert res["buy"][0]["provider_name"] == "Google Play Movies"
        assert "themoviedb.org" in res["link"]
        assert "streaming, rent, buy" in res["message"]


def test_fetch_watch_providers_empty_region_result(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "dummy_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": {"US": {"flatrate": []}}}  # No 'IN'

    with patch("requests.get", return_value=mock_resp):
        res = fetch_watch_providers("155", region="IN")
        assert res["available"] is False
        assert res["status"] == "no_providers"
        assert "Streaming availability information is currently unavailable for this title in India." in res["message"]


def test_fetch_watch_providers_unsafe_link_rejected(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "dummy_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": {
            "IN": {
                "link": "javascript:alert('xss')",
                "flatrate": [{"provider_name": "Netflix", "logo_path": "/netflix.jpg", "provider_id": 8}],
            }
        }
    }

    with patch("requests.get", return_value=mock_resp):
        res = fetch_watch_providers("155", region="IN")
        assert res["available"] is True
        # Malicious non-http(s) link should be stripped to empty string
        assert res["link"] == ""


def test_fetch_watch_providers_http_errors(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "dummy_key")
    for error_code in [401, 404, 429, 500]:
        mock_resp = MagicMock()
        mock_resp.status_code = error_code
        with patch("requests.get", return_value=mock_resp):
            res = fetch_watch_providers("155", region="IN")
            assert res["available"] is False
            assert res["status"] == "api_error"
            assert "unavailable" in res["message"].lower()


def test_fetch_watch_providers_network_timeout(monkeypatch):
    monkeypatch.setenv("TMDB_API_KEY", "dummy_key")
    with patch("requests.get", side_effect=Exception("Connection timeout")):
        res = fetch_watch_providers("155", region="IN")
        assert res["available"] is False
        assert res["status"] == "api_error"
        assert "timeout" in res["message"].lower()
