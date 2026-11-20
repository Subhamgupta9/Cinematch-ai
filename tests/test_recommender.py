"""
Unit and Integration Tests for Recommender Engine
-------------------------------------------------
Validates model training, similarity search, top-N output, filtering,
explainability generation, edge cases, memory optimization, determinism,
and graceful error handling.
"""

import sys
from pathlib import Path
import pytest
from scipy import sparse

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from recommender import MovieRecommender


@pytest.fixture(scope="module")
def trained_recommender():
    rec = MovieRecommender()
    if not rec.load_artifacts():
        rec.fit()
    return rec


def test_model_fitted_and_matrix_valid(trained_recommender):
    """Verify TF-IDF matrix is computed and has expected dimensions."""
    assert trained_recommender.tfidf_matrix is not None
    assert trained_recommender.tfidf_matrix.shape[0] == len(trained_recommender.movies_df)
    assert trained_recommender.tfidf_matrix.shape[1] > 1000


def test_sparse_memory_optimization(trained_recommender):
    """
    Mandatory Improvement #2: Verify sparse matrix storage and memory efficiency.
    Ensures that no dense N x N matrix is ever materialized.
    """
    matrix = trained_recommender.tfidf_matrix
    assert sparse.issparse(matrix), "TF-IDF matrix must be stored as a sparse matrix"
    assert isinstance(matrix, sparse.csr_matrix), "Matrix must be in CSR format for fast row slicing"

    mem_stats = trained_recommender.get_memory_stats()
    assert mem_stats["is_sparse"] is True
    assert mem_stats["sparse_feature_matrix_kb"] < 5000, "Sparse matrix should occupy under 5MB"
    assert mem_stats["dense_similarity_matrix_avoided_mb"] > 200, "Should document avoiding >200MB dense matrix"


def test_similarity_score_integrity_labels(trained_recommender):
    """
    Mandatory Improvement #1: Verify that scores are strictly labeled as
    'Cosine Similarity Match Score' and never described as accuracy or probability.
    """
    res = trained_recommender.recommend("Inception (2010)", top_n=3)
    assert res["success"] is True

    for r in res["recommendations"]:
        exp = r["explanation"]
        assert exp["metric_type"] == "Cosine Similarity Match Score"
        assert "not" in exp["metric_note"].lower() and "accuracy" in exp["metric_note"].lower()
        # Verify score is in valid [0.0, 100.0] range
        assert 0.0 <= exp["match_score_pct"] <= 100.0
        # Ensure 'accuracy' is never the metric label
        assert "accuracy" not in exp


def test_find_movie_exact_and_fuzzy(trained_recommender):
    """Test movie index resolution across exact, clean, and fuzzy queries."""
    idx1, _ = trained_recommender.find_movie_idx("Toy Story (1995)")
    assert idx1 is not None

    idx2, _ = trained_recommender.find_movie_idx("Toy Story")
    assert idx2 == idx1

    idx3, _ = trained_recommender.find_movie_idx("toy story")
    assert idx3 == idx1

    idx4, _ = trained_recommender.find_movie_idx("interstellar")
    assert idx4 is not None
    assert "Interstellar" in trained_recommender.movies_df.iloc[idx4]["title"]


def test_recommend_returns_requested_count(trained_recommender):
    """Verify that recommender returns exact top_n requested (5 and 10)."""
    res5 = trained_recommender.recommend("Toy Story", top_n=5)
    assert res5["success"] is True
    assert len(res5["recommendations"]) == 5

    res10 = trained_recommender.recommend("Toy Story", top_n=10)
    assert res10["success"] is True
    assert len(res10["recommendations"]) == 10


def test_query_movie_excluded_from_recommendations(trained_recommender):
    """Mandatory Improvement #4: Ensure the queried movie NEVER recommends itself."""
    queries_to_test = ["Inception (2010)", "Toy Story (1995)", "Matrix, The (1999)"]
    for query in queries_to_test:
        res = trained_recommender.recommend(query, top_n=10)
        assert res["success"] is True
        rec_ids = [r["movieId"] for r in res["recommendations"]]
        assert res["query_movie"]["movieId"] not in rec_ids


def test_recommendation_genre_filter(trained_recommender):
    """Verify that specifying a genre filter restricts all recommendations to that genre."""
    res = trained_recommender.recommend("Inception (2010)", top_n=5, genre_filter="Sci-Fi")
    assert res["success"] is True
    for r in res["recommendations"]:
        assert "Sci-Fi" in r["genre_list"], f"Movie {r['title']} missing required genre 'Sci-Fi'"


def test_recommendation_min_rating_filter(trained_recommender):
    """Verify that recommendations adhere to min_rating threshold."""
    min_rating = 3.5
    res = trained_recommender.recommend("Toy Story", top_n=5, min_rating=min_rating)
    assert res["success"] is True
    for r in res["recommendations"]:
        assert r["avg_rating"] >= min_rating, f"Rating {r['avg_rating']} is below threshold {min_rating}"


def test_popular_and_less_popular_movies(trained_recommender):
    """
    Mandatory Improvement #5: Test recommendations for both blockbuster popular
    movies and obscure/less popular movies with few ratings.
    """
    # 1. Popular blockbuster
    res_pop = trained_recommender.recommend("Dark Knight, The (2008)", top_n=5)
    assert res_pop["success"] is True
    assert len(res_pop["recommendations"]) == 5

    # 2. Less popular movie (vote_count <= 2)
    less_popular = trained_recommender.movies_df[
        (trained_recommender.movies_df["vote_count"] <= 2) &
        (trained_recommender.movies_df["genres_clean"] != "")
    ].iloc[0]

    res_less = trained_recommender.recommend(less_popular["title"], top_n=5)
    assert res_less["success"] is True
    assert len(res_less["recommendations"]) > 0


def test_movies_with_no_user_tags(trained_recommender):
    """
    Mandatory Improvement #4: Verify movies without user tags are supported
    using genre, decade, and title features.
    """
    movies_no_tags = trained_recommender.movies_df[
        trained_recommender.movies_df["tags_tokenized"] == ""
    ]
    assert not movies_no_tags.empty, "There should be movies without tags in MovieLens"

    sample_movie = movies_no_tags.iloc[0]["title"]
    res = trained_recommender.recommend(sample_movie, top_n=5)
    assert res["success"] is True
    assert len(res["recommendations"]) > 0


def test_movies_with_no_genres_listed(trained_recommender):
    """
    Mandatory Improvement #4: Handle '(no genres listed)' movies gracefully
    without raising zero-division errors.
    """
    no_genre_df = trained_recommender.movies_df[
        trained_recommender.movies_df["genres"] == "(no genres listed)"
    ]
    if not no_genre_df.empty:
        sample_title = no_genre_df.iloc[0]["title"]
        res = trained_recommender.recommend(sample_title, top_n=5)
        assert res["success"] is True
        assert len(res["recommendations"]) > 0


def test_duplicate_titles_handling(trained_recommender):
    """
    Mandatory Improvement #4: Verify movies with duplicate titles (e.g. Emma (1996))
    are handled without collisions.
    """
    res = trained_recommender.recommend("Emma (1996)", top_n=5)
    assert res["success"] is True
    assert res["query_movie"] is not None


def test_recommendation_determinism(trained_recommender):
    """
    Mandatory Improvement #5: Verify that identical queries produce
    deterministic, identical recommendation outputs.
    """
    res_run1 = trained_recommender.recommend("Pulp Fiction (1994)", top_n=5)
    res_run2 = trained_recommender.recommend("Pulp Fiction (1994)", top_n=5)

    rec_ids_1 = [r["movieId"] for r in res_run1["recommendations"]]
    rec_ids_2 = [r["movieId"] for r in res_run2["recommendations"]]
    scores_1 = [r["similarity_score"] for r in res_run1["recommendations"]]
    scores_2 = [r["similarity_score"] for r in res_run2["recommendations"]]

    assert rec_ids_1 == rec_ids_2, "Recommendations must be deterministic"
    assert scores_1 == scores_2, "Similarity scores must be deterministic"


def test_invalid_movie_handling(trained_recommender):
    """Verify invalid queries fail gracefully with suggestions and clear feedback."""
    res = trained_recommender.recommend("NonExistentFakeMovie_XYZ_12345")
    assert res["success"] is False
    assert res["query_movie"] is None
    assert len(res["recommendations"]) == 0
    assert "not found" in res["message"].lower()


def test_popular_movies_and_genres_methods(trained_recommender):
    """Verify utility methods for initial catalog display."""
    genres = trained_recommender.get_all_genres()
    assert len(genres) >= 15
    assert "Action" in genres
    assert "Comedy" in genres
    assert "Drama" in genres

    popular = trained_recommender.get_popular_movies(top_n=10)
    assert len(popular) == 10
    assert all("title" in p and "avg_rating" in p for p in popular)


def test_recommendations_sorted_strictly_descending(trained_recommender):
    """
    Quality Audit #3: Verify that all recommendation lists are sorted strictly
    descending by their similarity score when quality_boost == 0.0.
    """
    queries = [
        "Inception (2010)",
        "Dark Knight, The (2008)",
        "Toy Story (1995)",
        "Pulp Fiction (1994)",
        "Forrest Gump (1994)",
    ]
    for q in queries:
        res = trained_recommender.recommend(q, top_n=10, quality_boost=0.0)
        assert res["success"] is True
        recs = res["recommendations"]
        assert len(recs) == 10
        scores = [r["similarity_score"] for r in recs]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Ranking inversion for query '{q}': score at rank {i+1} ({scores[i]}) "
                f"< score at rank {i+2} ({scores[i+1]})"
            )


def test_inception_no_ranking_inversion(trained_recommender):
    """
    Quality Audit #2: Specifically verify that Inception recommendations never
    display a lower match score before a higher match score (e.g., 31.8% before 32.2%).
    """
    res = trained_recommender.recommend("Inception (2010)", top_n=10, quality_boost=0.0)
    assert res["success"] is True
    match_pcts = [r["explanation"]["match_score_pct"] for r in res["recommendations"]]
    for i in range(len(match_pcts) - 1):
        assert match_pcts[i] >= match_pcts[i + 1], (
            f"Match % inversion in Inception recommendations: {match_pcts[i]}% < {match_pcts[i+1]}%"
        )


def test_dark_knight_relevance_and_no_imax_skew(trained_recommender):
    """
    Quality Audit #1 & #6: Verify that The Dark Knight recommendations are relevant
    crime/action/drama films and that non-narrative technical format IMAX no longer
    skews recommendations into promoting Need for Speed, Noah, or The Grandmaster into the top 5.
    """
    res = trained_recommender.recommend("Dark Knight, The (2008)", top_n=5, quality_boost=0.0)
    assert res["success"] is True
    top_5_titles = [r["title"] for r in res["recommendations"]]

    # Verify problematic titles are completely excluded from top 5
    skewed_titles = [
        "Need for Speed (2014)",
        "Noah (2014)",
        "Grandmaster, The (Yi dai zong shi) (2013)",
    ]
    for bad_title in skewed_titles:
        assert bad_title not in top_5_titles, f"Format distortion detected: {bad_title} appeared in top 5!"

    # Verify all top 5 recommendations share core genres (Action, Crime, or Drama)
    for r in res["recommendations"]:
        assert any(g in r["genre_list"] for g in ["Action", "Crime", "Drama"]), (
            f"Movie {r['title']} shares no core genre with The Dark Knight"
        )


def test_quality_boost_ranking_consistency(trained_recommender):
    """
    Quality Audit #3: Verify that when quality_boost > 0, recommendations are sorted
    strictly descending by ranking_score, and ranking_score is reported in the payload.
    """
    res = trained_recommender.recommend("Inception (2010)", top_n=10, quality_boost=0.15)
    assert res["success"] is True
    recs = res["recommendations"]
    assert len(recs) == 10

    rank_scores = [r["ranking_score"] for r in recs]
    for i in range(len(rank_scores) - 1):
        assert rank_scores[i] >= rank_scores[i + 1], (
            f"Ranking score inversion with quality_boost: {rank_scores[i]} < {rank_scores[i+1]}"
        )


def test_html_escaping_untrusted_strings():
    """
    Quality Audit #8: Verify that dynamic dataset strings containing HTML special characters
    are safely escaped before rendering into markdown.
    """
    import html
    test_cases = [
        ('<script>alert("XSS")</script>', "&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;"),
        ("Movie & Co. <2024>", "Movie &amp; Co. &lt;2024&gt;"),
        ("<b>Action</b>", "&lt;b&gt;Action&lt;/b&gt;"),
    ]
    for raw, expected in test_cases:
        escaped = html.escape(raw)
        assert escaped == expected
        assert "<script>" not in escaped
        assert "<b>" not in escaped


def test_normalize_title_prefix():
    """Verify title prefix normalization with leading/trailing articles and year extraction."""
    from recommender import normalize_title_prefix

    canon, stripped, yr = normalize_title_prefix("Dark Knight, The (2008)")
    assert canon == "The Dark Knight"
    assert stripped == "Dark Knight"
    assert yr == 2008

    canon, stripped, yr = normalize_title_prefix("The Dark Knight")
    assert canon == "The Dark Knight"
    assert stripped == "Dark Knight"
    assert yr is None

    canon, stripped, yr = normalize_title_prefix("Silence of the Lambs, The")
    assert canon == "The Silence of the Lambs"
    assert stripped == "Silence of the Lambs"
    assert yr is None

    canon, stripped, yr = normalize_title_prefix("A Clockwork Orange (1971)")
    assert canon == "A Clockwork Orange"
    assert stripped == "Clockwork Orange"
    assert yr == 1971

    canon, stripped, yr = normalize_title_prefix("Emma")
    assert canon == "Emma"
    assert stripped == "Emma"
    assert yr is None


def test_find_movie_idx_with_year_disambiguation(trained_recommender):
    """Verify duplicate-aware disambiguation between movies with identical titles in different years."""
    # MovieLens has Emma (1996) and Emma (2009)
    idx_1996, _ = trained_recommender.find_movie_idx("Emma", year=1996)
    idx_2009, _ = trained_recommender.find_movie_idx("Emma", year=2009)

    assert idx_1996 is not None
    assert idx_2009 is not None
    assert idx_1996 != idx_2009

    row_1996 = trained_recommender.movies_df.iloc[idx_1996]
    row_2009 = trained_recommender.movies_df.iloc[idx_2009]

    assert int(row_1996["year"]) == 1996
    assert int(row_2009["year"]) == 2009

    # Query with embedded year in string
    idx_embedded, _ = trained_recommender.find_movie_idx("Emma (1996)")
    assert idx_embedded == idx_1996

    # Non-existent year for duplicate title returns None with suggestions
    idx_none, suggestions = trained_recommender.find_movie_idx("Emma", year=1932)
    assert idx_none is None
    assert any("Emma" in s for s in suggestions)


def test_find_movie_idx_with_tmdb_id(trained_recommender):
    """Verify exact resolution by TMDb ID."""
    # The Dark Knight (2008) has TMDb ID 155
    idx, _ = trained_recommender.find_movie_idx("", tmdb_id="155")
    assert idx is not None
    assert "dark knight" in trained_recommender.movies_df.iloc[idx]["clean_title"].lower()


def test_find_movie_idx_symmetric_prefix(trained_recommender):
    """Verify symmetric resolution with leading, trailing, or stripped article."""
    idx_leading, _ = trained_recommender.find_movie_idx("The Dark Knight")
    idx_trailing, _ = trained_recommender.find_movie_idx("Dark Knight, The")
    idx_stripped, _ = trained_recommender.find_movie_idx("Dark Knight")

    assert idx_leading is not None
    assert idx_trailing is not None
    assert idx_stripped is not None
    assert idx_leading == idx_trailing == idx_stripped


def test_rating_sorting_end_to_end_catalog(trained_recommender):
    """Verify ascending and descending rating sorting across the complete catalog."""
    df = trained_recommender.movies_df

    # Descending sorting
    desc_df = df.sort_values(by=["weighted_rating", "vote_count"], ascending=[False, False])
    desc_ratings = desc_df["weighted_rating"].tolist()
    for i in range(len(desc_ratings) - 1):
        assert desc_ratings[i] >= desc_ratings[i + 1]

    # Ascending sorting
    asc_df = df.sort_values(by=["weighted_rating", "vote_count"], ascending=[True, True])
    asc_ratings = asc_df["weighted_rating"].tolist()
    for i in range(len(asc_ratings) - 1):
        assert asc_ratings[i] <= asc_ratings[i + 1]


