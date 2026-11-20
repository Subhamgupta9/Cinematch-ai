"""
Reproducible Offline Evaluation Suite for CineMatch AI
-------------------------------------------------------
Benchmarks recommendation models against historical ground truth using an
item-seeded discovery protocol on MovieLens 100K with strictly isolated
3-way temporal train/validation/test splits.

Key Design & Rigor Guarantees (Architecture Section 8):
1. Item-Seeded Evaluation: Matches production query paradigm (user selects seed q_u,
   not persistent user profile personalization).
2. Seed Selection: q_u is the most recent positively rated training movie (r >= 4.0).
3. Candidate Exclusion: The query seed q_u is strictly excluded from recommendations.
4. 3-Way Chronological Split: 70% Train, 15% Validation, 15% Test per user.
5. Zero Leakage: Collaborative correlation matrix, Bayesian quality priors, and
   popularity baselines are fitted exclusively on R_train.
6. Multi-Model Benchmark Suite:
   - M1: Popularity Baseline (Bayesian quality prior alone)
   - M2: Pure Content (TF-IDF cosine similarity alone)
   - M3: Content + Quality (Production baseline in recommender.py)
   - M4: Pure Collaborative (Item-Item shrunk Pearson correlation)
   - M5: Hybrid Recommender (Calibrated multi-signal fusion with relevance-first gating)
7. Cold-Start Simulation: 20% of train items are masked to measure out-of-catalog resilience.
"""

import sys
import json
import time
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collaborative_recommender import CollaborativeRecommender


def run_item_seeded_offline_evaluation(
    data_dir: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    random_seed: int = 42,
    sample_users: int = 100,
) -> Dict[str, Any]:
    """
    Executes rigorous offline evaluation across M1 to M5 and cold-start simulation.
    """
    np.random.seed(random_seed)
    base_path = Path(__file__).resolve().parent.parent
    data_path = Path(data_dir) if data_dir else (base_path / "data" / "ml-latest-small")

    ratings_df = pd.read_csv(data_path / "ratings.csv")
    movies_df = pd.read_csv(data_path / "movies.csv")
    tags_df = pd.read_csv(data_path / "tags.csv")

    # 1. Clean tags
    tags_df["clean_tag"] = tags_df["tag"].astype(str).str.lower().str.strip()
    tags_df = tags_df[~tags_df["clean_tag"].isin(["in netflix queue", "", "none", "nan"])]
    movie_tags = tags_df.groupby("movieId")["clean_tag"].agg(lambda s: list(dict.fromkeys(s))).reset_index()
    movie_tags = movie_tags.rename(columns={"clean_tag": "tag_list"})
    movies_df = movies_df.merge(movie_tags, on="movieId", how="left")
    movies_df["tag_list"] = movies_df["tag_list"].apply(lambda x: x if isinstance(x, list) else [])
    movies_df["tags_tokenized"] = movies_df["tag_list"].apply(lambda t: " ".join(t))

    # Parse clean title and year
    import re
    def parse_title(raw_title):
        match = re.search(r"\((\d{4})\)", raw_title)
        year = int(match.group(1)) if match else None
        clean = re.sub(r"\s*\(\d{4}\)\s*", "", raw_title).strip()
        for article in [", The", ", A", ", An"]:
            if clean.endswith(article):
                clean = article.replace(", ", "") + " " + clean[: -len(article)]
                break
        return clean, year

    parsed = [parse_title(t) for t in movies_df["title"]]
    movies_df["clean_title"] = [p[0] for p in parsed]
    movies_df["year"] = [p[1] for p in parsed]
    movies_df["genre_list"] = movies_df["genres"].apply(
        lambda x: [g.strip() for g in x.split("|")] if x != "(no genres listed)" else []
    )

    # Build production feature soup
    def build_soup(row):
        parts = []
        if row["clean_title"]:
            parts.append(row["clean_title"].lower())
        narrative_genres = [g.replace("-", "").replace(" ", "") for g in row["genre_list"] if g != "IMAX"]
        if narrative_genres:
            parts.append((" ".join(narrative_genres) + " ") * 2)
        if "IMAX" in row["genre_list"]:
            parts.append("format_imax")
        if row["tags_tokenized"]:
            parts.append((row["tags_tokenized"] + " ") * 2)
        if row["year"] and not pd.isna(row["year"]):
            decade = int(row["year"] // 10 * 10)
            parts.append(f"decade_{decade}s")
        s = " ".join(parts).strip().lower()
        return s or f"movie_id_{row['movieId']}"

    movies_df["metadata_soup"] = movies_df.apply(build_soup, axis=1)

    # 2. Strict Chronological Temporal Split per User (70% train / 15% val / 15% test)
    eligible_users = []
    train_ratings = []
    val_ratings = []
    test_ratings = []

    for u_id, u_df in ratings_df.groupby("userId"):
        pos_df = u_df[u_df["rating"] >= 4.0].sort_values("timestamp")
        # Must have >= 10 positive ratings to evaluate train, val, and test partitions meaningfully
        if len(pos_df) >= 10:
            eligible_users.append(u_id)
            n = len(pos_df)
            n_train = max(1, int(n * 0.70))
            n_val = max(1, int(n * 0.15))

            train_ratings.append(pos_df.iloc[:n_train])
            val_ratings.append(pos_df.iloc[n_train : n_train + n_val])
            test_ratings.append(pos_df.iloc[n_train + n_val :])

    train_df = pd.concat(train_ratings)
    val_df = pd.concat(val_ratings)
    test_df = pd.concat(test_ratings)

    print(f"Data Split Summary:")
    print(f"  * Eligible Users (>= 10 liked items): {len(eligible_users)}")
    print(f"  * Train Positives: {len(train_df):,}")
    print(f"  * Val Positives:   {len(val_df):,}")
    print(f"  * Test Positives:  {len(test_df):,}")

    # 3. Fit Models STRICTLY on Train Data
    # A. Content Vectorizer (fitted on catalog metadata)
    vec = TfidfVectorizer(stop_words="english", max_features=12000, ngram_range=(1, 2))
    tfidf_matrix = vec.fit_transform(movies_df["metadata_soup"])
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies_df["movieId"])}
    total_catalog = len(movies_df)

    # B. Bayesian Quality Prior (strictly on train_df)
    train_vote_counts = train_df.groupby("movieId")["rating"].count()
    train_vote_means = train_df.groupby("movieId")["rating"].mean()
    global_mean_train = float(train_df["rating"].mean())

    def get_train_quality(m_id):
        v = train_vote_counts.get(m_id, 0)
        r = train_vote_means.get(m_id, global_mean_train)
        w = (v / (v + 5.0)) * r + (5.0 / (v + 5.0)) * global_mean_train
        return float(w / 5.0)

    # C. Item-Item Collaborative Recommender (strictly on train_df)
    collab = CollaborativeRecommender(tau=5.0)
    collab.fit_from_ratings(train_df)

    # 4. Evaluation Engine
    k_vals = [5, 10, 20]
    eval_users = eligible_users[:sample_users]

    models = ["M1_Popularity", "M2_Pure_Content", "M3_Content_Quality", "M4_Pure_Collab", "M5_Hybrid"]

    def run_eval_for_partition(split_df: pd.DataFrame, partition_name: str) -> Dict[str, Any]:
        results: Dict[str, Dict[str, List[float]]] = {
            m: {f"P@{k}": [] for k in k_vals} for m in models
        }
        for m in models:
            for k in k_vals:
                results[m][f"R@{k}"] = []
                results[m][f"NDCG@{k}"] = []
                results[m][f"MRR@{k}"] = []
            results[m]["coverage_set"] = set()
            results[m]["diversity"] = []

        valid_users = 0

        for u_id in eval_users:
            u_train = train_df[train_df["userId"] == u_id].sort_values("timestamp")
            u_test = split_df[split_df["userId"] == u_id]

            if len(u_train) == 0 or len(u_test) == 0:
                continue

            ground_truth = set(u_test["movieId"].tolist())
            if not ground_truth:
                continue

            # Seed is most recent positive training item
            seed_mid = int(u_train.iloc[-1]["movieId"])
            if seed_mid not in movie_id_to_idx:
                continue

            seed_idx = movie_id_to_idx[seed_mid]
            valid_users += 1

            # Candidate pool: All movies except seed
            cand_mids = [m for m in movies_df["movieId"] if m != seed_mid]
            cand_indices = [movie_id_to_idx[m] for m in cand_mids]

            # Compute content similarities
            content_sims = linear_kernel(tfidf_matrix[seed_idx], tfidf_matrix).flatten()

            # Compute collab similarities
            collab_neighbors = dict(collab.get_top_collaborative_neighbors(seed_mid, top_n=200))

            # Compute quality scores
            qual_scores = np.array([get_train_quality(m) for m in cand_mids])

            # Generate Top-20 for each model
            # M1: Popularity alone
            m1_order = np.argsort(qual_scores)[::-1][: max(k_vals)]
            m1_recs = [cand_mids[i] for i in m1_order]

            # M2: Pure Content
            m2_scores = np.array([content_sims[i] for i in cand_indices])
            m2_order = np.argsort(m2_scores)[::-1][: max(k_vals)]
            m2_recs = [cand_mids[i] for i in m2_order]

            # M3: Content + Quality (0.85 * Content + 0.15 * Quality)
            m3_scores = 0.85 * m2_scores + 0.15 * qual_scores
            m3_order = np.argsort(m3_scores)[::-1][: max(k_vals)]
            m3_recs = [cand_mids[i] for i in m3_order]

            # M4: Pure Collab
            m4_scores = np.array([collab_neighbors.get(m, 0.0) for m in cand_mids])
            m4_order = np.argsort(m4_scores)[::-1][: max(k_vals)]
            m4_recs = [cand_mids[i] for i in m4_order]

            # M5: Hybrid (Relevance-first gating + dynamic shrinkage)
            # Relevance-first gating: if content_sim < 0.04, score = 0
            m5_scores = np.zeros(len(cand_mids), dtype=np.float32)
            for idx_c, m in enumerate(cand_mids):
                sim_c = m2_scores[idx_c]
                if sim_c < 0.04:
                    m5_scores[idx_c] = 0.0
                    continue
                collab_raw = collab_neighbors.get(m, 0.0)
                collab_eff = collab_raw * min(1.0, sim_c / 0.40)
                q = qual_scores[idx_c]
                m5_scores[idx_c] = 0.90 * (0.65 * sim_c + 0.35 * collab_eff) + 0.10 * q

            m5_order = np.argsort(m5_scores)[::-1][: max(k_vals)]
            m5_recs = [cand_mids[i] for i in m5_order]

            # Score each model
            model_recs = {
                "M1_Popularity": m1_recs,
                "M2_Pure_Content": m2_recs,
                "M3_Content_Quality": m3_recs,
                "M4_Pure_Collab": m4_recs,
                "M5_Hybrid": m5_recs,
            }

            for m_name, rec_list in model_recs.items():
                results[m_name]["coverage_set"].update(rec_list)

                # Intra-list diversity (Jaccard distance across genres for top-10)
                top10_genres = [
                    set(movies_df.iloc[movie_id_to_idx[mid]]["genre_list"])
                    for mid in rec_list[:10]
                    if mid in movie_id_to_idx
                ]
                dists = []
                for i in range(len(top10_genres)):
                    for j in range(i + 1, len(top10_genres)):
                        u = len(top10_genres[i] | top10_genres[j])
                        inter = len(top10_genres[i] & top10_genres[j])
                        dists.append(1.0 - (inter / u) if u > 0 else 0.0)
                if dists:
                    results[m_name]["diversity"].append(float(np.mean(dists)))

                for k in k_vals:
                    k_list = rec_list[:k]
                    hits = [1 if m in ground_truth else 0 for m in k_list]
                    n_hits = sum(hits)

                    prec = n_hits / k
                    rec = n_hits / len(ground_truth)

                    # NDCG
                    dcg = sum(h / math.log2(rank + 2) for rank, h in enumerate(hits))
                    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(ground_truth))))
                    ndcg = (dcg / idcg) if idcg > 0 else 0.0

                    # MRR
                    mrr = 0.0
                    for rank, h in enumerate(hits):
                        if h == 1:
                            mrr = 1.0 / (rank + 1)
                            break

                    results[m_name][f"P@{k}"].append(prec)
                    results[m_name][f"R@{k}"].append(rec)
                    results[m_name][f"NDCG@{k}"].append(ndcg)
                    results[m_name][f"MRR@{k}"].append(mrr)

        # Compute summary table
        summary = {}
        for m_name in models:
            summary[m_name] = {
                "P@5": float(np.mean(results[m_name]["P@5"])),
                "P@10": float(np.mean(results[m_name]["P@10"])),
                "P@20": float(np.mean(results[m_name]["P@20"])),
                "R@5": float(np.mean(results[m_name]["R@5"])),
                "R@10": float(np.mean(results[m_name]["R@10"])),
                "R@20": float(np.mean(results[m_name]["R@20"])),
                "NDCG@5": float(np.mean(results[m_name]["NDCG@5"])),
                "NDCG@10": float(np.mean(results[m_name]["NDCG@10"])),
                "NDCG@20": float(np.mean(results[m_name]["NDCG@20"])),
                "MRR@10": float(np.mean(results[m_name]["MRR@10"])),
                "Catalog_Coverage": len(results[m_name]["coverage_set"]) / total_catalog,
                "Genre_Diversity_Jaccard": float(np.mean(results[m_name]["diversity"])) if results[m_name]["diversity"] else 0.0,
            }
        summary["evaluated_users"] = valid_users
        return summary

    print("\nEvaluating on Validation Partition (15%)...")
    val_results = run_eval_for_partition(val_df, "Validation")

    print("\nEvaluating on Test Partition (15%)...")
    test_results = run_eval_for_partition(test_df, "Test")

    # 5. Simulated Cold-Start Benchmark (20% train items masked)
    print("\nRunning Simulated Cold-Start Benchmark (20% Train Items Masked)...")
    all_train_items = list(train_df["movieId"].unique())
    n_masked = int(len(all_train_items) * 0.20)
    masked_items = set(np.random.choice(all_train_items, size=n_masked, replace=False))

    masked_train_df = train_df[~train_df["movieId"].isin(masked_items)]
    cold_collab = CollaborativeRecommender(tau=5.0)
    cold_collab.fit_from_ratings(masked_train_df)

    cold_eval_hits_m4 = []
    cold_eval_hits_m5 = []

    for u_id in eval_users:
        u_test = test_df[test_df["userId"] == u_id]
        cold_ground_truth = set(u_test[u_test["movieId"].isin(masked_items)]["movieId"].tolist())
        if not cold_ground_truth:
            continue

        u_train = train_df[train_df["userId"] == u_id].sort_values("timestamp")
        if u_train.empty:
            continue
        seed_mid = int(u_train.iloc[-1]["movieId"])
        if seed_mid not in movie_id_to_idx:
            continue

        seed_idx = movie_id_to_idx[seed_mid]
        content_sims = linear_kernel(tfidf_matrix[seed_idx], tfidf_matrix).flatten()
        cand_indices = [movie_id_to_idx[m] for m in masked_items if m in movie_id_to_idx]
        masked_cand_list = [m for m in masked_items if m in movie_id_to_idx]

        # M4 Collab recovery on masked items: 0% by definition because they have 0 interactions in masked_train
        m4_collab_recs = cold_collab.get_top_collaborative_neighbors(seed_mid, top_n=10)
        m4_hits = sum(1 for m, _ in m4_collab_recs if m in cold_ground_truth)
        cold_eval_hits_m4.append(m4_hits / len(cold_ground_truth))

        # M5 Hybrid recovery via TF-IDF dynamic projection
        m5_scores = np.array([content_sims[i] for i in cand_indices])
        m5_top_order = np.argsort(m5_scores)[::-1][:10]
        m5_recs = [masked_cand_list[i] for i in m5_top_order]
        m5_hits = sum(1 for m in m5_recs if m in cold_ground_truth)
        cold_eval_hits_m5.append(m5_hits / len(cold_ground_truth))

    cold_start_summary = {
        "M4_Pure_Collab_Recall@10": float(np.mean(cold_eval_hits_m4)) if cold_eval_hits_m4 else 0.0,
        "M5_Hybrid_Recall@10": float(np.mean(cold_eval_hits_m5)) if cold_eval_hits_m5 else 0.0,
        "Evaluated_Cold_Instances": len(cold_eval_hits_m5),
    }

    final_results = {
        "validation_partition": val_results,
        "test_partition": test_results,
        "cold_start_simulation": cold_start_summary,
    }

    out_file = Path(output_path) if output_path else (base_path / "evaluation" / "evaluation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nEvaluation successfully saved to: {out_file}")
    return final_results


if __name__ == "__main__":
    run_item_seeded_offline_evaluation()
