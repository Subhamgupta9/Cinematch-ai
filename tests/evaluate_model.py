"""
Comprehensive Evaluation of Content-Based Recommender on MovieLens
------------------------------------------------------------------
Calculates Precision@K, Recall@K, and NDCG@K using real user rating interactions,
and compares feature weighting strategies.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml-latest-small"


def load_data():
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")

    # Clean tags
    tags["clean_tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tags = tags[~tags["clean_tag"].isin(["in netflix queue", "", "none", "nan"])]
    movie_tags = tags.groupby("movieId")["clean_tag"].agg(lambda s: list(dict.fromkeys(s))).reset_index()
    movie_tags = movie_tags.rename(columns={"clean_tag": "tag_list"})
    movies = movies.merge(movie_tags, on="movieId", how="left")
    movies["tag_list"] = movies["tag_list"].apply(lambda x: x if isinstance(x, list) else [])
    movies["tags_tokenized"] = movies["tag_list"].apply(lambda tags: " ".join(tags))

    # Clean genres
    movies["genre_list"] = movies["genres"].apply(lambda x: [g.strip() for g in x.split("|")] if x != "(no genres listed)" else [])

    return movies, ratings


def run_evaluation(soup_column_name, movies, ratings, k=10, sample_users=100, seed=42):
    """
    Evaluates recommender on user interaction data:
    Users with >= 20 ratings of 4.0+ are sampled.
    For each user, a high-rated seed movie is queried, and top-K recommendations
    are evaluated against the user's remaining high-rated movies.
    """
    np.random.seed(seed)

    # 1. Fit TF-IDF on the given soup
    vec = TfidfVectorizer(stop_words="english", max_features=12000, ngram_range=(1, 2))
    matrix = vec.fit_transform(movies[soup_column_name])

    # Map movieId to matrix row index
    movie_id_to_idx = {m_id: idx for idx, m_id in enumerate(movies["movieId"])}

    # 2. Find eligible users (users with >= 20 ratings >= 4.0)
    high_ratings = ratings[ratings["rating"] >= 4.0]
    user_counts = high_ratings["userId"].value_counts()
    eligible_users = user_counts[user_counts >= 20].index.tolist()

    if len(eligible_users) > sample_users:
        selected_users = np.random.choice(eligible_users, size=sample_users, replace=False)
    else:
        selected_users = eligible_users

    precisions = []
    recalls = []
    ndcgs = []

    for u_id in selected_users:
        u_liked = set(high_ratings[high_ratings["userId"] == u_id]["movieId"])
        # Pick 1 seed movie liked by the user that exists in our matrix
        valid_seeds = [m for m in u_liked if m in movie_id_to_idx]
        if len(valid_seeds) < 2:
            continue

        seed_movie_id = np.random.choice(valid_seeds)
        ground_truth = u_liked - {seed_movie_id}

        seed_idx = movie_id_to_idx[seed_movie_id]
        scores = linear_kernel(matrix[seed_idx], matrix).flatten()

        # Rank all other movies
        ranked_indices = np.argsort(scores)[::-1]

        # Get top-K recommendations (excluding the seed movie itself)
        rec_ids = []
        for r_idx in ranked_indices:
            if r_idx == seed_idx:
                continue
            rec_ids.append(movies.iloc[r_idx]["movieId"])
            if len(rec_ids) >= k:
                break

        # Calculate Precision@K
        hits = [1 if m in ground_truth else 0 for m in rec_ids]
        n_hits = sum(hits)
        precision = n_hits / k
        recall = n_hits / len(ground_truth) if len(ground_truth) > 0 else 0.0

        # Calculate DCG@K and IDCG@K for NDCG@K
        dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal_hits = [1] * min(k, len(ground_truth))
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(len(ideal_hits)))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "evaluated_users": len(precisions),
    }


def main():
    movies, ratings = load_data()
    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings.")

    # Model A: Baseline Feature Soup (original)
    # Includes IMAX in genres, genres weighted 2x, tags 1x
    def soup_baseline(row):
        parts = []
        clean = row["title"].split(" (")[0]
        parts.append(clean.lower())
        genres_clean = [g for g in row["genre_list"]]
        parts.append((" ".join(genres_clean) + " ") * 2)
        if row["tags_tokenized"]:
            parts.append(row["tags_tokenized"])
        return " ".join(parts).lower()

    movies["soup_baseline"] = movies.apply(soup_baseline, axis=1)

    # Model B: Content-Refined Feature Soup
    # 1. Exclude 'IMAX' from narrative genres (keep IMAX only as technical tag)
    # 2. Boost semantic tags 2x (so rich user tags like 'superhero', 'dark', 'quentin tarantino' have parity with genres)
    # 3. Clean franchise title words
    def soup_refined(row):
        parts = []
        clean = row["title"].split(" (")[0]
        parts.append(clean.lower())
        # Narrative genres only (IMAX is projection format, not a narrative genre)
        narrative_genres = [g for g in row["genre_list"] if g != "IMAX"]
        if narrative_genres:
            parts.append((" ".join(narrative_genres) + " ") * 2)
        # Weight semantic community tags 2x
        if row["tags_tokenized"]:
            parts.append((row["tags_tokenized"] + " ") * 2)
        # Add technical format tag if IMAX without narrative genre repetition
        if "IMAX" in row["genre_list"]:
            parts.append("format_imax")
        return " ".join(parts).lower()

    movies["soup_refined"] = movies.apply(soup_refined, axis=1)

    print("\n--- Running Evaluation Protocol (Precision@10, Recall@10, NDCG@10) across 100 Active Users ---")
    res_a = run_evaluation("soup_baseline", movies, ratings, k=10)
    print("\n[Baseline Model (Original Feature Soup)]:")
    print(f"  * Precision@10: {res_a['precision_at_k']*100:.2f}%")
    print(f"  * Recall@10:    {res_a['recall_at_k']*100:.2f}%")
    print(f"  * NDCG@10:      {res_a['ndcg_at_k']*100:.2f}%")

    res_b = run_evaluation("soup_refined", movies, ratings, k=10)
    print("\n[Refined Model (Narrative Genres + Weighted Semantic Tags + Format Tag)]:")
    print(f"  * Precision@10: {res_b['precision_at_k']*100:.2f}%")
    print(f"  * Recall@10:    {res_b['recall_at_k']*100:.2f}%")
    print(f"  * NDCG@10:      {res_b['ndcg_at_k']*100:.2f}%")

    # Let's also check Dark Knight recommendations under refined soup
    vec_b = TfidfVectorizer(stop_words="english", max_features=12000, ngram_range=(1, 2))
    mat_b = vec_b.fit_transform(movies["soup_refined"])
    dk_idx = movies[movies["title"].str.contains("Dark Knight, The")].index[0]
    scores_b = linear_kernel(mat_b[dk_idx], mat_b).flatten()
    top_b = np.argsort(scores_b)[::-1][1:6]

    print("\n[Dark Knight Recommendations with Refined Model]:")
    for idx in top_b:
        m = movies.iloc[idx]
        print(f"  * {m['title']} ({scores_b[idx]*100:.1f}%) | Genres: {m['genres']} | Tags: {m['tag_list'][:4]}")


if __name__ == "__main__":
    main()
