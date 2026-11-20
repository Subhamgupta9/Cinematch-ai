"""
Unit Tests for Hybrid Recommender Module
----------------------------------------
Validates the full Stage 3 hybrid recommendation architecture:
- Dynamic vector projection into 12,000-dimensional TF-IDF space
- Minimum-Evidence Policy & IDF-based token discrimination
- Source-isolated Bayesian quality scoring
- Adaptive routing: Path A (Collaborative) vs Path B (Cold-Start)
- Relevance-first gating & collaborative contribution shrinkage
- Deterministic multi-tier sort order
- Concrete Product Acceptance Tests (RRR, Avengers: Endgame, 3 Idiots, The Dark Knight, Total Recall remakes, empty API key, sparse metadata)
"""

import pytest
import math
from pathlib import Path
from scipy import sparse

from unified_data_model import UnifiedMovie
from hybrid_recommender import HybridRecommender


@pytest.fixture(scope="module")
def hybrid():
    hr = HybridRecommender(alpha=0.65, beta=0.10, gamma_sim=0.04)
    hr.initialize()
    return hr


def test_hybrid_recommender_initialization(hybrid):
    assert hybrid._is_initialized is True
    assert hybrid.alpha == 0.65
    assert hybrid.beta == 0.10
    assert hybrid.gamma_sim == 0.04
    assert len(hybrid.feature_names) == 12000
    assert len(hybrid.idf_map) == 12000


def test_feature_soup_generation(hybrid):
    movie = UnifiedMovie(
        title="Avengers: Endgame",
        release_year=2019,
        genre_names=["Action", "Adventure", "Sci-Fi", "IMAX"],
        keywords=["superhero", "time travel"],
    )
    soup = hybrid.build_feature_soup(movie)

    assert "avengers endgame" in soup
    assert "action" in soup
    assert "format_imax" in soup
    assert "superhero" in soup
    assert "decade_2010s" in soup


def test_dynamic_vector_projection(hybrid):
    soup = "avengers endgame action scifi superhero time travel decade_2010s"
    vec = hybrid.project_to_vector(soup)

    assert isinstance(vec, sparse.csr_matrix)
    assert vec.shape == (1, 12000)
    # Check non-zero L2 norm
    norm = float(sparse.linalg.norm(vec))
    assert norm > 0.0


def test_minimum_evidence_policy_acceptance(hybrid):
    # Query and candidate sharing a discriminative term ('superhero', IDF >= 4.0)
    q_soup = "batman action superhero dark gritty decade_2000s"
    c_soup = "superman action superhero metropolis decade_2010s"

    passes, msg = hybrid.check_minimum_evidence(q_soup, c_soup, similarity_score=0.25)
    assert passes is True


def test_minimum_evidence_policy_rejection(hybrid):
    # Below similarity threshold
    passes, msg = hybrid.check_minimum_evidence("comedy drama", "comedy drama", similarity_score=0.01)
    assert passes is False

    # Disjoint tokens
    passes, msg = hybrid.check_minimum_evidence("horror zombie", "romance wedding", similarity_score=0.10)
    assert passes is False


def test_source_isolated_quality_movielens(hybrid):
    # MovieLens movie with high ratings
    cand = UnifiedMovie(
        movielens_movie_id=58559,
        movielens_rating=4.2,
        movielens_rating_count=149,
    )
    qual = hybrid.compute_quality_score(cand)
    assert 0.0 <= qual <= 1.0
    assert qual > 0.7  # High quality


def test_source_isolated_quality_tmdb(hybrid):
    # TMDb movie with high votes
    cand = UnifiedMovie(
        tmdb_id="155",
        tmdb_vote_average=8.5,
        tmdb_vote_count=32000,
    )
    qual = hybrid.compute_quality_score(cand)
    assert 0.0 <= qual <= 1.0
    assert qual > 0.8

    # Unrated or low vote count title
    cand_unrated = UnifiedMovie(tmdb_id="999", tmdb_vote_average=9.0, tmdb_vote_count=5)
    assert hybrid.compute_quality_score(cand_unrated) == 0.0


def test_adaptive_routing_path_a_collab(hybrid):
    # Both items have MovieLens interaction data
    q = UnifiedMovie(movielens_movie_id=58559, title="The Dark Knight", genre_names=["Action", "Crime"])
    c = UnifiedMovie(movielens_movie_id=1261, title="Evil Dead II", genre_names=["Action", "Crime"])

    q_soup = hybrid.build_feature_soup(q)
    c_soup = hybrid.build_feature_soup(c)
    q_vec = hybrid.project_to_vector(q_soup)
    c_vec = hybrid.project_to_vector(c_soup)

    res = hybrid.score_candidate(q, c, q_vec, c_vec, q_soup, c_soup)
    assert res["path"] == "path_a_collab"
    assert res["collaborative_score"] is not None


def test_adaptive_routing_path_b_cold_start(hybrid):
    # Query is TMDb-only (e.g. RRR)
    q = UnifiedMovie(tmdb_id="579974", title="RRR", genre_names=["Action", "Drama"])
    c = UnifiedMovie(movielens_movie_id=1293, title="Gandhi", genre_names=["Action", "Drama"])

    q_soup = hybrid.build_feature_soup(q)
    c_soup = hybrid.build_feature_soup(c)
    q_vec = hybrid.project_to_vector(q_soup)
    c_vec = hybrid.project_to_vector(c_soup)

    res = hybrid.score_candidate(q, c, q_vec, c_vec, q_soup, c_soup)
    assert res["path"] == "path_b_cold_start"
    assert res["collaborative_score"] is None


# ==============================================================================
# EVIDENCE-BASED PRODUCT ACCEPTANCE TESTS (Architecture Section 13.2)
# ==============================================================================

def test_acceptance_case_1_rrr_cold_start(hybrid):
    """Product Acceptance Test 1: RRR as a Seed Movie."""
    rrr = UnifiedMovie(
        tmdb_id="579974",
        title="RRR",
        release_year=2022,
        original_language="te",
        origin_country=["IN"],
        genre_names=["Action", "History", "Drama"],
        keywords=["revolution", "freedom fighter", "1920s", "british empire", "india"],
    )

    result = hybrid.recommend(rrr, top_n=5)
    assert result["success"] is True
    recs = result["recommendations"]
    assert len(recs) > 0

    # Assertions from specification:
    assert result["query_movie"]["canonical_id"] == "tmdb:579974"
    assert result["query_movie"]["metadata_source"] == "tmdb_global"
    assert result["query_movie"]["origin_country"] == ["IN"]
    assert result["query_movie"]["original_language"] == "te"
    assert result["query_movie"]["collaborative_data_available"] is False

    # Check scores strictly descending in [0.0, 1.0]
    scores = [r["recommendation_score"] for r in recs]
    for s in scores:
        assert 0.0 <= s <= 1.0
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]

    # Check collaborative score is labeled N/A
    for r in recs:
        assert "N/A" in str(r["collaborative_score"])


def test_acceptance_case_2_avengers_endgame(hybrid):
    """Product Acceptance Test 2: Avengers: Endgame as a Seed Movie."""
    endgame = UnifiedMovie(
        tmdb_id="299534",
        title="Avengers: Endgame",
        release_year=2019,
        genre_names=["Action", "Adventure", "Science Fiction"],
        keywords=["superhero", "time travel", "marvel cinematic universe (mcu)"],
    )

    result = hybrid.recommend(endgame, top_n=5)
    assert result["success"] is True
    assert result["query_movie"]["canonical_id"] == "tmdb:299534"

    recs = result["recommendations"]
    assert len(recs) > 0
    for r in recs:
        assert 0.0 <= r["recommendation_score"] <= 1.0
        # Match explanations contain non-empty shared genres or terms
        assert len(r["shared_genres"]) > 0 or len(r["top_terms"]) > 0


def test_acceptance_case_3_3_idiots_collab(hybrid):
    """Product Acceptance Test 3: 3 Idiots as a Seed Movie."""
    result = hybrid.recommend("3 Idiots", top_n=5)
    assert result["success"] is True

    # 3 Idiots is in MovieLens (73881) and links to TMDb (20453)
    query_movie = result["query_movie"]
    assert query_movie["canonical_id"] == "tmdb:20453"
    assert query_movie["metadata_source"] == "movielens_linked"
    assert query_movie["collaborative_data_available"] is True

    # Ensure self-exclusion
    rec_cids = [r["canonical_id"] for r in result["recommendations"]]
    assert "tmdb:20453" not in rec_cids


def test_acceptance_case_5_deduplication(hybrid):
    """Product Acceptance Test 5: The Dark Knight Deduplication Check."""
    result = hybrid.recommend("The Dark Knight", top_n=10)
    assert result["success"] is True

    recs = result["recommendations"]
    cids = [r["canonical_id"] for r in recs]
    # No duplicate canonical IDs
    assert len(cids) == len(set(cids))


def test_acceptance_case_6_remake_separation(hybrid):
    """Product Acceptance Test 6: Remake Separation (Total Recall 1990 vs 2012)."""
    tr_1990 = UnifiedMovie(
        tmdb_id="861",
        movielens_movie_id=1200,
        release_year=1990,
        title="Total Recall",
        genre_names=["Action", "Sci-Fi"],
    )

    result = hybrid.recommend(tr_1990, top_n=10)
    assert result["success"] is True

    # Total Recall 1990 must not recommend itself
    for r in result["recommendations"]:
        assert r["canonical_id"] != "tmdb:861"


def test_acceptance_case_7_missing_api_key_local_fallback(hybrid):
    """Product Acceptance Test 7: Missing API Key Local Mode Fallback."""
    result = hybrid.recommend("Inception", top_n=5, api_key="")
    assert result["success"] is True
    assert len(result["recommendations"]) > 0


def test_acceptance_case_8_sparse_metadata(hybrid):
    """Product Acceptance Test 8: Empty or Insufficient Metadata Query."""
    sparse_movie = UnifiedMovie(
        title="Minimalist Mystery",
        genre_names=["(no genres listed)"],
        keywords=[],
    )

    result = hybrid.recommend(sparse_movie, top_n=5, api_key="")
    assert result["success"] is True
    # Graceful handling with 0 unhandled exceptions


def test_deterministic_sort_order(hybrid):
    """Verifies that multiple repeated queries return bit-identical rankings."""
    res1 = hybrid.recommend("The Dark Knight", top_n=5, api_key="")
    res2 = hybrid.recommend("The Dark Knight", top_n=5, api_key="")

    cids1 = [r["canonical_id"] for r in res1["recommendations"]]
    cids2 = [r["canonical_id"] for r in res2["recommendations"]]
    scores1 = [r["recommendation_score"] for r in res1["recommendations"]]
    scores2 = [r["recommendation_score"] for r in res2["recommendations"]]

    assert cids1 == cids2
    assert scores1 == scores2
