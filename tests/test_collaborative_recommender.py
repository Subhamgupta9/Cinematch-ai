"""
Unit Tests for Collaborative Recommender Module
------------------------------------------------
Validates item-item user-centered Pearson correlation, significance shrinkage,
rectification, symmetry, zero self-similarity, sparse serialization, and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

from collaborative_recommender import CollaborativeRecommender


@pytest.fixture
def sample_ratings_df():
    """Constructs a deterministic synthetic ratings dataset."""
    data = [
        # User 1 rates movies 10, 20, 30 highly
        {"userId": 1, "movieId": 10, "rating": 5.0, "timestamp": 100},
        {"userId": 1, "movieId": 20, "rating": 4.5, "timestamp": 101},
        {"userId": 1, "movieId": 30, "rating": 4.0, "timestamp": 102},
        # User 2 rates movies 10, 20 highly, movie 40 low
        {"userId": 2, "movieId": 10, "rating": 4.0, "timestamp": 103},
        {"userId": 2, "movieId": 20, "rating": 5.0, "timestamp": 104},
        {"userId": 2, "movieId": 40, "rating": 1.0, "timestamp": 105},
        # User 3 rates movies 10, 20, 30 highly
        {"userId": 3, "movieId": 10, "rating": 4.5, "timestamp": 106},
        {"userId": 3, "movieId": 20, "rating": 4.5, "timestamp": 107},
        {"userId": 3, "movieId": 30, "rating": 5.0, "timestamp": 108},
        # User 4 rates movies 10, 20, 30, 40
        {"userId": 4, "movieId": 10, "rating": 4.0, "timestamp": 109},
        {"userId": 4, "movieId": 20, "rating": 4.0, "timestamp": 110},
        {"userId": 4, "movieId": 30, "rating": 4.5, "timestamp": 111},
        {"userId": 4, "movieId": 40, "rating": 1.5, "timestamp": 112},
        # User 5 rates movies 10, 20
        {"userId": 5, "movieId": 10, "rating": 5.0, "timestamp": 113},
        {"userId": 5, "movieId": 20, "rating": 5.0, "timestamp": 114},
        # User 6 rates only movie 50 (isolated)
        {"userId": 6, "movieId": 50, "rating": 5.0, "timestamp": 115},
    ]
    return pd.DataFrame(data)


def test_collaborative_recommender_initialization():
    cr = CollaborativeRecommender(tau=5.0)
    assert cr.tau == 5.0
    assert cr._is_fitted is False
    assert cr.R_centered is None
    assert cr.R_binary is None


def test_fit_from_ratings_dataframe(sample_ratings_df):
    cr = CollaborativeRecommender(tau=5.0)
    cr.fit_from_ratings(sample_ratings_df)

    assert cr._is_fitted is True
    assert len(cr.user_map) == 6
    assert len(cr.movie_map) == 5  # 10, 20, 30, 40, 50
    assert cr.R_centered.shape == (6, 5)
    assert cr.R_binary.shape == (6, 5)


def test_is_collaborative_available(sample_ratings_df):
    cr = CollaborativeRecommender().fit_from_ratings(sample_ratings_df)
    assert cr.is_collaborative_available(10) is True
    assert cr.is_collaborative_available(20) is True
    assert cr.is_collaborative_available(999) is False
    assert cr.is_collaborative_available(None) is False


def test_pairwise_similarity_symmetry(sample_ratings_df):
    cr = CollaborativeRecommender(tau=5.0).fit_from_ratings(sample_ratings_df)
    sim_10_20 = cr.get_pairwise_similarity(10, 20)
    sim_20_10 = cr.get_pairwise_similarity(20, 10)
    assert sim_10_20 == pytest.approx(sim_20_10, rel=1e-5)
    assert sim_10_20 > 0.0


def test_significance_shrinkage(sample_ratings_df):
    # Movies 10 & 20 share 5 users (1, 2, 3, 4, 5) -> co_count = 5 >= tau (5) -> shrinkage factor = 1.0
    cr5 = CollaborativeRecommender(tau=5.0).fit_from_ratings(sample_ratings_df)
    sim_5 = cr5.get_pairwise_similarity(10, 20)

    # If tau is increased to 10, shrinkage factor = 5 / 10 = 0.5
    cr10 = CollaborativeRecommender(tau=10.0).fit_from_ratings(sample_ratings_df)
    sim_10 = cr10.get_pairwise_similarity(10, 20)

    assert sim_10 == pytest.approx(sim_5 * 0.5, rel=1e-4)


def test_disjoint_users_zero_similarity(sample_ratings_df):
    cr = CollaborativeRecommender(tau=5.0).fit_from_ratings(sample_ratings_df)
    # Movie 50 is only rated by User 6; Movie 10 is rated by Users 1..5. Zero common users.
    sim = cr.get_pairwise_similarity(10, 50)
    assert sim == 0.0


def test_exclude_self_similarity(sample_ratings_df):
    cr = CollaborativeRecommender(tau=5.0).fit_from_ratings(sample_ratings_df)
    # Pairwise self similarity returns 0.0
    assert cr.get_pairwise_similarity(10, 10) == 0.0

    neighbors = cr.get_top_collaborative_neighbors(10, top_n=10)
    neighbor_ids = [mid for mid, score in neighbors]
    assert 10 not in neighbor_ids


def test_top_neighbors_strictly_descending(sample_ratings_df):
    cr = CollaborativeRecommender(tau=5.0).fit_from_ratings(sample_ratings_df)
    neighbors = cr.get_top_collaborative_neighbors(10, top_n=5)
    assert len(neighbors) > 0
    scores = [score for mid, score in neighbors]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_unknown_movie_handling(sample_ratings_df):
    cr = CollaborativeRecommender(tau=5.0).fit_from_ratings(sample_ratings_df)
    assert cr.get_top_collaborative_neighbors(99999) == []
    assert cr.get_pairwise_similarity(99999, 10) == 0.0
    assert cr.get_pairwise_similarity(10, None) == 0.0


def test_empty_dataframe_raises_value_error():
    cr = CollaborativeRecommender()
    with pytest.raises(ValueError, match="empty ratings DataFrame"):
        cr.fit_from_ratings(pd.DataFrame())


def test_missing_column_raises_value_error():
    cr = CollaborativeRecommender()
    with pytest.raises(ValueError, match="missing required column"):
        cr.fit_from_ratings(pd.DataFrame({"userId": [1], "rating": [5.0]}))


def test_precompute_sparse_matrix(sample_ratings_df, tmp_path):
    cr = CollaborativeRecommender(tau=5.0, models_dir=tmp_path)
    cr.fit_from_ratings(sample_ratings_df)
    mat = cr.precompute_sparse_matrix(top_k_per_item=3)

    assert isinstance(mat, sparse.csr_matrix)
    assert mat.shape == (5, 5)

    # Save and reload
    save_file = cr.save_matrix(tmp_path / "test_matrix.npz")
    assert save_file.exists()

    cr2 = CollaborativeRecommender(tau=5.0, models_dir=tmp_path)
    cr2.fit_from_ratings(sample_ratings_df)
    cr2.load_matrix(save_file)
    assert cr2.item_collab_matrix is not None
    assert cr2.item_collab_matrix.shape == (5, 5)
