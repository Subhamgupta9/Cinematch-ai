"""
Unit Tests for Data Preprocessing & Feature Engineering
-------------------------------------------------------
Validates raw dataset loading, title parsing, genre tokenization,
user tag aggregation, and Bayesian rating computations.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from recommender import MovieRecommender


@pytest.fixture(scope="module")
def recommender():
    rec = MovieRecommender()
    rec.load_and_preprocess()
    return rec


def test_dataset_loaded_properly(recommender):
    """Verify that dataset loads with the expected non-empty structure."""
    df = recommender.movies_df
    assert not df.empty, "Preprocessed DataFrame must not be empty"
    assert len(df) >= 9000, f"Expected at least 9,000 movies, found {len(df)}"
    assert "movieId" in df.columns
    assert "title" in df.columns
    assert "genres" in df.columns
    assert "clean_title" in df.columns
    assert "year" in df.columns
    assert "avg_rating" in df.columns
    assert "vote_count" in df.columns
    assert "weighted_rating" in df.columns
    assert "metadata_soup" in df.columns


def test_clean_title_and_year_extraction(recommender):
    """Verify title cleaning and year regex extraction."""
    df = recommender.movies_df

    # Toy Story (1995)
    toy_story = df[df["movieId"] == 1].iloc[0]
    assert toy_story["clean_title"] == "Toy Story"
    assert toy_story["year"] == 1995

    # Check movie with trailing article: 'Dark Knight, The (2008)' -> 'The Dark Knight'
    dark_knight = df[df["clean_title"] == "The Dark Knight"]
    assert not dark_knight.empty, "Should resolve 'Dark Knight, The' to 'The Dark Knight'"
    assert dark_knight.iloc[0]["year"] == 2008


def test_genre_parsing(recommender):
    """Verify genre splitting and handling of (no genres listed)."""
    df = recommender.movies_df

    # Toy Story has multiple genres
    toy_story = df[df["movieId"] == 1].iloc[0]
    assert "Animation" in toy_story["genre_list"]
    assert "Children" in toy_story["genre_list"]
    assert "Comedy" in toy_story["genre_list"]

    # Movies with no genres listed should have empty genre_list, not error
    no_genre_movies = df[df["genres"] == "(no genres listed)"]
    if not no_genre_movies.empty:
        sample = no_genre_movies.iloc[0]
        assert sample["genre_list"] == []


def test_ratings_aggregation(recommender):
    """Verify rating calculation, positive counts, and Bayesian weighted ratings."""
    df = recommender.movies_df
    toy_story = df[df["movieId"] == 1].iloc[0]

    assert toy_story["vote_count"] > 100, "Toy Story should have over 100 ratings"
    assert 3.5 <= toy_story["avg_rating"] <= 4.5, "Toy Story rating should be in plausible range"
    assert toy_story["weighted_rating"] > 0, "Weighted rating should be positive"


def test_tag_cleaning_and_aggregation(recommender):
    """Verify tags are aggregated into lists and noise tags are filtered out."""
    df = recommender.movies_df
    assert "tag_list" in df.columns
    assert "tags_tokenized" in df.columns

    # Verify no 'in netflix queue' tags remain
    for tag_list in df["tag_list"]:
        assert "in netflix queue" not in tag_list


def test_metadata_soup_composition(recommender):
    """Verify that metadata soup contains title, genres, and era information."""
    df = recommender.movies_df
    toy_story = df[df["movieId"] == 1].iloc[0]
    soup = toy_story["metadata_soup"]

    assert "toy" in soup
    assert "story" in soup
    assert "animation" in soup
    assert "decade_1990s" in soup
