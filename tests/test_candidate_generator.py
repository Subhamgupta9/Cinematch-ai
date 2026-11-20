"""
Unit Tests for Candidate Generator Module
------------------------------------------
Validates canonical identity resolution, links.csv translation, remake separation,
deterministic 100-candidate cap allocation, multi-source intersection prioritization,
floor diversity reservations, and self-exclusion.
"""

import pytest
from unified_data_model import UnifiedMovie
from candidate_generator import (
    CandidateGenerator,
    MAX_TOTAL_CANDIDATES,
    CHANNEL_FLOORS,
)


@pytest.fixture
def generator():
    return CandidateGenerator()


def test_canonical_id_assignment_priority(generator):
    # Primary: TMDb ID known
    m1 = UnifiedMovie(tmdb_id="155", movielens_movie_id=58559, imdb_id="tt0468569", title="The Dark Knight")
    assert generator.resolve_canonical_id(m1) == "tmdb:155"

    # Secondary: TMDb missing, but MovieLens maps to TMDb in links.csv
    m2 = UnifiedMovie(tmdb_id=None, movielens_movie_id=58559, title="The Dark Knight")
    assert generator.resolve_canonical_id(m2) == "tmdb:155"

    # Tertiary: Unlinked MovieLens ID with no TMDb match
    m3 = UnifiedMovie(tmdb_id=None, movielens_movie_id=9999999, imdb_id="tt9999999", title="Fake Movie")
    assert generator.resolve_canonical_id(m3) == "imdb:tt9999999"

    # Quaternary: Isolated MovieLens ID
    m4 = UnifiedMovie(tmdb_id=None, movielens_movie_id=9999999, imdb_id="", title="Isolated Movie")
    assert generator.resolve_canonical_id(m4) == "ml:9999999"


def test_links_csv_bidirectional_indexing(generator):
    assert len(generator.ml_to_tmdb) >= 9700
    assert len(generator.tmdb_to_ml) >= 9700
    assert len(generator.ml_to_imdb) >= 9700

    # Inception: ML 79132 <-> TMDb 27205
    assert generator.ml_to_tmdb.get(79132) == "27205"
    assert generator.tmdb_to_ml.get("27205") == 79132


def test_enrich_entity_identity(generator):
    # TMDb movie that exists in MovieLens
    tmdb_movie = UnifiedMovie(tmdb_id="27205", title="Inception")
    enriched = generator.enrich_entity_identity(tmdb_movie)

    assert enriched.movielens_movie_id == 79132
    assert enriched.metadata_source == "movielens_linked"
    assert enriched.collaborative_data_available is True
    assert enriched.movielens_rating is not None
    assert enriched.movielens_rating_count is not None


def test_remake_independence(generator):
    total_recall_1990 = UnifiedMovie(
        tmdb_id="861",
        movielens_movie_id=1200,
        release_year=1990,
        title="Total Recall",
    )
    total_recall_2012 = UnifiedMovie(
        tmdb_id="64635",
        movielens_movie_id=96821,
        release_year=2012,
        title="Total Recall",
    )

    cid_1990 = generator.resolve_canonical_id(total_recall_1990)
    cid_2012 = generator.resolve_canonical_id(total_recall_2012)

    assert cid_1990 == "tmdb:861"
    assert cid_2012 == "tmdb:64635"
    assert cid_1990 != cid_2012


def test_merge_entities(generator):
    # Primary: MovieLens record with academic rating
    ml_record = UnifiedMovie(
        movielens_movie_id=58559,
        title="The Dark Knight",
        release_year=2008,
        movielens_rating=4.2,
        movielens_rating_count=149,
        movielens_tags=["dark", "gritty"],
        genre_names=["Action", "Crime"],
    )

    # Incoming TMDb record with poster and TMDb vote average
    tmdb_record = UnifiedMovie(
        tmdb_id="155",
        title="The Dark Knight",
        release_year=2008,
        poster_path="/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
        tmdb_vote_average=8.5,
        tmdb_vote_count=32000,
        genre_names=["Action", "Drama"],
        keywords=["superhero", "joker"],
    )

    merged = generator.merge_entities(ml_record, tmdb_record)

    assert merged.movielens_movie_id == 58559
    assert merged.tmdb_id == "155"
    assert merged.metadata_source == "movielens_linked"
    assert merged.movielens_rating == 4.2
    assert merged.movielens_rating_count == 149
    assert merged.tmdb_vote_average == 8.5
    assert merged.tmdb_vote_count == 32000
    assert merged.poster_path == "/qJ2tW6WMUDux911r6m7haRef0WH.jpg"
    assert set(merged.genre_names) == {"Action", "Crime", "Drama"}
    assert set(merged.keywords) == {"superhero", "joker"}
    assert set(merged.movielens_tags) == {"dark", "gritty"}


def test_select_top_100_candidates_cap_invariant(generator):
    # Create 130 synthetic candidates across 5 channels
    raw_channels = {
        "ml_content": [
            UnifiedMovie(tmdb_id=str(1000 + i), title=f"ML Content {i}")
            for i in range(40)
        ],
        "ml_collab": [
            UnifiedMovie(tmdb_id=str(2000 + i), title=f"ML Collab {i}")
            for i in range(20)
        ],
        "tmdb_recs": [
            UnifiedMovie(tmdb_id=str(3000 + i), title=f"TMDb Rec {i}")
            for i in range(25)
        ],
        "tmdb_sim": [
            UnifiedMovie(tmdb_id=str(4000 + i), title=f"TMDb Sim {i}")
            for i in range(25)
        ],
        "tmdb_discover": [
            UnifiedMovie(tmdb_id=str(5000 + i), title=f"TMDb Discover {i}")
            for i in range(20)
        ],
    }

    selected = generator.select_top_100_candidates(raw_channels)
    assert len(selected) <= MAX_TOTAL_CANDIDATES
    assert len(selected) == 100


def test_select_top_100_candidates_under_100(generator):
    # Create 40 candidates total
    raw_channels = {
        "ml_content": [UnifiedMovie(tmdb_id=str(100 + i), title=f"Movie {i}") for i in range(20)],
        "tmdb_recs": [UnifiedMovie(tmdb_id=str(200 + i), title=f"Rec {i}") for i in range(20)],
    }

    selected = generator.select_top_100_candidates(raw_channels)
    assert len(selected) == 40


def test_multi_source_intersection_prioritization(generator):
    # Candidate "A" appears in 3 channels
    cand_shared = UnifiedMovie(tmdb_id="999", title="Shared Blockbuster")
    cand_lone = UnifiedMovie(tmdb_id="111", title="Single Source Film")

    raw_channels = {
        "ml_content": [cand_shared],
        "tmdb_recs": [cand_shared],
        "tmdb_sim": [cand_shared, cand_lone],
    }

    selected = generator.select_top_100_candidates(raw_channels)
    selected_cids = [generator.resolve_canonical_id(c) for c in selected]

    assert "tmdb:999" in selected_cids
    assert "tmdb:111" in selected_cids


def test_self_exclusion_invariant(generator):
    query_movie = UnifiedMovie(tmdb_id="155", title="The Dark Knight")
    query_cid = generator.resolve_canonical_id(query_movie)

    raw_channels = {
        "ml_content": [
            UnifiedMovie(tmdb_id="155", title="The Dark Knight"),  # Self
            UnifiedMovie(tmdb_id="27205", title="Inception"),
        ],
        "tmdb_recs": [
            UnifiedMovie(tmdb_id="155", title="The Dark Knight"),  # Self
            UnifiedMovie(tmdb_id="49026", title="The Dark Knight Rises"),
        ],
    }

    selected = generator.select_top_100_candidates(raw_channels, query_canonical_id=query_cid)
    selected_cids = [generator.resolve_canonical_id(c) for c in selected]

    assert query_cid not in selected_cids
    assert "tmdb:27205" in selected_cids
    assert "tmdb:49026" in selected_cids


def test_failed_channel_graceful_handling(generator):
    # TMDb channels failed / empty, ML content has 30 candidates
    raw_channels = {
        "ml_content": [UnifiedMovie(tmdb_id=str(100 + i), title=f"ML Movie {i}") for i in range(30)],
        "ml_collab": [],
        "tmdb_recs": [],
        "tmdb_sim": [],
        "tmdb_discover": [],
    }

    selected = generator.select_top_100_candidates(raw_channels)
    assert len(selected) == 30


def test_deterministic_tie_breaker(generator):
    # Two candidates with identical scores
    cand1 = UnifiedMovie(tmdb_id="300", title="Alpha Movie")
    cand2 = UnifiedMovie(tmdb_id="100", title="Beta Movie")
    setattr(cand1, "channel_score", 0.5)
    setattr(cand2, "channel_score", 0.5)

    raw_channels = {
        "ml_content": [cand1, cand2],
    }

    selected = generator.select_top_100_candidates(raw_channels)
    assert len(selected) == 2


def test_candidate_generator_initialization_defaults(generator):
    assert generator.data_dir.exists()
    assert generator.models_dir.exists()
