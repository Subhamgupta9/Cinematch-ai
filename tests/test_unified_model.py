"""
Unit Tests for Unified Movie Data Model
---------------------------------------
Validates normalization, source provenance, rating isolation, identity collision
prevention (duplicate titles and remakes), and factory conversions.
"""

import pytest
from unified_data_model import UnifiedMovie


def test_unified_movie_movielens_linked_record():
    movie = UnifiedMovie(
        tmdb_id="155",
        movielens_movie_id=58559,
        imdb_id="tt0468569",
        title="The Dark Knight",
        original_title="The Dark Knight",
        overview="Batman raises the stakes in his war on crime.",
        release_date="2008-07-16",
        release_year=2008,
        original_language="en",
        origin_country=["US"],
        genre_ids=[18, 28, 80],
        genre_names=["Drama", "Action", "Crime"],
        keywords=["superhero", "joker", "dc comics"],
        poster_path="/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
        backdrop_path="/nMKdUUepR0i5zn0y1T4CsSB5chy.jpg",
        tmdb_vote_average=8.5,
        tmdb_vote_count=32000,
        movielens_rating=4.2,
        movielens_rating_count=149,
        movielens_tags=["batman", "heath ledger", "dark"],
    )

    assert movie.tmdb_id == "155"
    assert movie.movielens_movie_id == 58559
    assert movie.imdb_id == "tt0468569"
    assert movie.title == "The Dark Knight"
    assert movie.release_year == 2008
    assert movie.metadata_source == "movielens_linked"
    assert movie.collaborative_data_available is True
    assert movie.primary_uid == "tmdb_155"
    assert movie.has_official_poster is True
    assert movie.get_poster_url() == "https://image.tmdb.org/t/p/w342/qJ2tW6WMUDux911r6m7haRef0WH.jpg"
    assert movie.movielens_rating == 4.2
    assert movie.tmdb_vote_average == 8.5


def test_unified_movie_tmdb_only_record():
    movie = UnifiedMovie(
        tmdb_id="299534",
        movielens_movie_id=None,
        imdb_id="tt4154796",
        title="Avengers: Endgame",
        original_title="Avengers: Endgame",
        overview="After the devastating events of Infinity War...",
        release_date="2019-04-24",
        release_year=2019,
        original_language="en",
        origin_country=["US"],
        genre_names=["Adventure", "Sci-Fi", "Action"],
        tmdb_vote_average=8.3,
        tmdb_vote_count=24890,
        movielens_rating=None,
        movielens_rating_count=None,
    )

    assert movie.tmdb_id == "299534"
    assert movie.movielens_movie_id is None
    assert movie.metadata_source == "tmdb_global"
    assert movie.collaborative_data_available is False
    assert movie.movielens_rating is None
    assert movie.movielens_rating_count is None
    assert movie.primary_uid == "tmdb_299534"
    assert movie.tmdb_vote_average == 8.3


def test_unified_movie_movielens_only_record():
    movie = UnifiedMovie(
        tmdb_id=None,
        movielens_movie_id=99999,
        title="Indie Archive Film",
        release_year=1995,
        movielens_rating=3.8,
        movielens_rating_count=12,
    )

    assert movie.tmdb_id is None
    assert movie.movielens_movie_id == 99999
    assert movie.metadata_source == "movielens_only"
    assert movie.collaborative_data_available is True
    assert movie.primary_uid == "ml_99999"


def test_rating_isolation_strict_provenance():
    """Verify that MovieLens and TMDb ratings are strictly isolated and never conflated."""
    tmdb_movie = UnifiedMovie(
        tmdb_id="100",
        title="Global Film",
        tmdb_vote_average=9.2,
        tmdb_vote_count=5000,
    )
    # MovieLens ratings must remain None, not inferred from TMDb
    assert tmdb_movie.movielens_rating is None
    assert tmdb_movie.movielens_rating_count is None
    assert tmdb_movie.collaborative_data_available is False

    ml_movie = UnifiedMovie(
        movielens_movie_id=200,
        title="Local Film",
        movielens_rating=4.5,
        movielens_rating_count=60,
    )
    # TMDb ratings must remain None, not inferred from MovieLens
    assert ml_movie.tmdb_vote_average is None
    assert ml_movie.tmdb_vote_count is None


def test_country_and_language_independence():
    """Verify that original language is never inferred from country of origin."""
    telugu_indian_film = UnifiedMovie(
        tmdb_id="579974",
        title="RRR",
        original_title="రౌద్రం రణం రుధిరం",
        original_language="te",
        origin_country=["IN"],
    )
    assert telugu_indian_film.origin_country == ["IN"]
    assert telugu_indian_film.original_language == "te"

    hindi_indian_film = UnifiedMovie(
        tmdb_id="360814",
        title="Dangal",
        original_language="hi",
        origin_country=["IN"],
    )
    assert hindi_indian_film.origin_country == ["IN"]
    assert hindi_indian_film.original_language == "hi"

    us_spanish_film = UnifiedMovie(
        tmdb_id="777",
        title="Bilingual Drama",
        original_language="es",
        origin_country=["US"],
    )
    assert us_spanish_film.origin_country == ["US"]
    assert us_spanish_film.original_language == "es"


def test_duplicate_titles_different_identities():
    """Test movies with identical titles are distinguished by primary catalog IDs."""
    emma_1996 = UnifiedMovie(
        tmdb_id="9426",
        movielens_movie_id=838,
        title="Emma",
        release_year=1996,
    )
    emma_2020 = UnifiedMovie(
        tmdb_id="556678",
        movielens_movie_id=None,
        title="Emma.",
        release_year=2020,
    )

    assert emma_1996.primary_uid != emma_2020.primary_uid
    assert emma_1996.primary_uid == "tmdb_9426"
    assert emma_2020.primary_uid == "tmdb_556678"


def test_remakes_different_identities():
    """Test remake titles maintain distinct identities."""
    total_recall_1990 = UnifiedMovie(
        tmdb_id="861",
        movielens_movie_id=2916,
        title="Total Recall",
        release_year=1990,
    )
    total_recall_2012 = UnifiedMovie(
        tmdb_id="64635",
        movielens_movie_id=96079,
        title="Total Recall",
        release_year=2012,
    )

    assert total_recall_1990.primary_uid == "tmdb_861"
    assert total_recall_2012.primary_uid == "tmdb_64635"
    assert total_recall_1990.release_year == 1990
    assert total_recall_2012.release_year == 2012


def test_invalid_identifiers_handling():
    """Ensure invalid, zero, or non-numeric identifiers are cleaned safely to None."""
    movie = UnifiedMovie(
        tmdb_id="0",
        movielens_movie_id=-5,
        title="Test Movie",
    )
    assert movie.tmdb_id is None
    assert movie.movielens_movie_id is None

    movie_nan = UnifiedMovie(
        tmdb_id="abc",
        movielens_movie_id=None,
        title="Test Movie 2",
    )
    assert movie_nan.tmdb_id is None


def test_missing_and_null_fields_defaults():
    """Ensure model does not crash with missing or null attributes."""
    movie = UnifiedMovie()
    assert movie.title == "Untitled"
    assert movie.overview == ""
    assert movie.release_year is None
    assert movie.genre_names == []
    assert movie.origin_country == []
    assert movie.collaborative_data_available is False
    assert movie.has_official_poster is False
    assert movie.get_poster_url() is None


def test_from_tmdb_dict_factory():
    """Test factory creation from Stage 1 normalized TMDb dictionary."""
    raw_tmdb = {
        "tmdb_id": "579974",
        "title": "RRR",
        "original_title": "రౌద్రం రణం రుధిరం",
        "overview": "A tale of two revolutionaries.",
        "release_date": "2022-03-24",
        "release_year": 2022,
        "original_language": "te",
        "origin_country": ["IN"],
        "genre_ids": [28, 18],
        "genre_names": ["Action", "Drama"],
        "poster_path": "/wE0noFU2EToSHUrHz11Qcyv0uqN.jpg",
        "backdrop_path": "/backdrop.jpg",
        "tmdb_vote_average": 7.8,
        "tmdb_vote_count": 1600,
    }

    entity = UnifiedMovie.from_tmdb_dict(raw_tmdb)
    assert entity.tmdb_id == "579974"
    assert entity.title == "RRR"
    assert entity.original_title == "రౌద్రం రణం రుధిరం"
    assert entity.original_language == "te"
    assert entity.origin_country == ["IN"]
    assert entity.genre_names == ["Action", "Drama"]
    assert entity.tmdb_vote_average == 7.8
    assert entity.movielens_rating is None
    assert entity.collaborative_data_available is False
    assert entity.metadata_source == "tmdb_global"


def test_from_movielens_row_factory():
    """Test factory creation from MovieLens DataFrame row dictionary."""
    ml_row = {
        "movieId": 1,
        "title": "Toy Story (1995)",
        "clean_title": "Toy Story",
        "year": 1995,
        "genre_list": ["Adventure", "Animation", "Children", "Comedy", "Fantasy"],
        "avg_rating": 3.9,
        "vote_count": 215,
        "description": "Release Year: 1995 | Genres: Adventure, Animation...",
        "tag_list": ["pixar", "fun", "animation"],
        "imdbId": "tt0114709",
        "tmdbId": "862",
    }

    entity = UnifiedMovie.from_movielens_row(ml_row)
    assert entity.movielens_movie_id == 1
    assert entity.tmdb_id == "862"
    assert entity.imdb_id == "tt0114709"
    assert entity.title == "Toy Story"
    assert entity.release_year == 1995
    assert entity.movielens_rating == 3.9
    assert entity.movielens_rating_count == 215
    assert entity.collaborative_data_available is True
    assert entity.metadata_source == "movielens_linked"
    assert "pixar" in entity.movielens_tags


def test_enrich_with_tmdb_preserves_movielens_ratings():
    """Test enriching a MovieLens entity with TMDb metadata without polluting ratings."""
    ml_row = {
        "movieId": 122922,
        "clean_title": "Doctor Strange",
        "year": 2016,
        "genre_list": ["Action", "Adventure", "Sci-Fi"],
        "avg_rating": 3.7,
        "vote_count": 22,
        "tag_list": ["visually appealing"],
        "imdbId": "tt1211837",
        "tmdbId": "284052",
    }
    entity = UnifiedMovie.from_movielens_row(ml_row)

    tmdb_data = {
        "poster_path": "/uGBVj3bEbCoZbDjjl9wTxcygko1.jpg",
        "backdrop_path": "/backdrop_strange.jpg",
        "overview": "After his career is destroyed, a brilliant surgeon gets a new lease on life.",
        "tmdb_vote_average": 7.4,
        "tmdb_vote_count": 21000,
        "keywords": ["marvel", "sorcerer", "magic"],
    }

    enriched = entity.enrich_with_tmdb(tmdb_data)

    # TMDb attributes merged
    assert enriched.poster_path == "/uGBVj3bEbCoZbDjjl9wTxcygko1.jpg"
    assert enriched.overview.startswith("After his career is destroyed")
    assert enriched.tmdb_vote_average == 7.4
    assert enriched.tmdb_vote_count == 21000
    assert "marvel" in enriched.keywords

    # MovieLens ratings and tags strictly preserved
    assert enriched.movielens_rating == 3.7
    assert enriched.movielens_rating_count == 22
    assert enriched.movielens_tags == ["visually appealing"]
    assert enriched.collaborative_data_available is True
    assert enriched.metadata_source == "movielens_linked"


def test_to_dict_and_from_dict_serialization():
    """Test dictionary serialization roundtrip."""
    original = UnifiedMovie(
        tmdb_id="550",
        movielens_movie_id=2959,
        title="Fight Club",
        release_year=1999,
        original_language="en",
        origin_country=["US"],
        genre_names=["Drama"],
        movielens_rating=4.3,
        movielens_rating_count=218,
        tmdb_vote_average=8.4,
        tmdb_vote_count=27000,
    )

    d = original.to_dict()
    assert isinstance(d, dict)
    assert d["title"] == "Fight Club"
    assert d["tmdb_id"] == "550"

    reconstructed = UnifiedMovie.from_dict(d)
    assert reconstructed.title == original.title
    assert reconstructed.tmdb_id == original.tmdb_id
    assert reconstructed.movielens_rating == original.movielens_rating
    assert reconstructed.collaborative_data_available == original.collaborative_data_available
