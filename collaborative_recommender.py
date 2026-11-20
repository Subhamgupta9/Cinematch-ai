"""
Collaborative Recommender Engine for CineMatch AI
-------------------------------------------------
Implements Item-Item Collaborative Filtering using user-mean-centered Pearson
correlation with Bayesian significance shrinkage (tau = 5).

Key Mathematical Foundations (Stage 3 Architecture Section 6.2):
1. User-Mean Centering:
   r_tilde_{u,i} = r_{u,i} - r_bar_u
2. Exact Pearson Correlation on Co-Rated Users U_{i,j}:
   s_raw(i, j) = sum_{u in U_ij} (r_tilde_{u,i} * r_tilde_{u,j}) /
                 ( sqrt(sum_{u in U_ij} r_tilde_{u,i}^2) * sqrt(sum_{u in U_ij} r_tilde_{u,j}^2) )
3. Significance Shrinkage (Penalizes low co-rating support |U_ij| < tau):
   s_shrunk(i, j) = (min(|U_ij|, tau) / tau) * s_raw(i, j)  [default tau = 5]
4. Rectification (Negative correlation indicates divergent tastes, not recommendation relevance):
   Sim_collab(i, j) = max(0.0, s_shrunk(i, j)) in [0.0, 1.0]

Memory & Compute Strategy:
- Sparse CSR representation (scipy.sparse.csr_matrix). Zero dense N x N matrix allocation.
- On-demand single-item vector dot-product runs in ~10-15 ms across all 9,742 movies.
- Supports precomputing and caching top-N neighbors (N <= 100) to disk.
- Zero fabrication: For items outside MovieLens (e.g. TMDb-only), collaborative signal
  is strictly unavailable (None), triggering cold-start handling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from scipy import sparse


class CollaborativeRecommender:
    """
    Item-Item Collaborative Filtering Recommender using shrunk Pearson correlation.
    """

    def __init__(
        self,
        data_dir: Optional[Path | str] = None,
        models_dir: Optional[Path | str] = None,
        tau: float = 5.0,
    ):
        base_path = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir else (base_path / "data" / "ml-latest-small")
        self.models_dir = Path(models_dir) if models_dir else (base_path / "models")
        self.tau = float(tau)

        # Internal state
        self.user_map: Dict[int, int] = {}
        self.movie_map: Dict[int, int] = {}
        self.rev_movie_map: Dict[int, int] = {}
        self.R_centered: Optional[sparse.csr_matrix] = None
        self.R_binary: Optional[sparse.csr_matrix] = None
        self.item_collab_matrix: Optional[sparse.csr_matrix] = None
        self._is_fitted: bool = False

    def fit_from_ratings(self, ratings_df: pd.DataFrame) -> "CollaborativeRecommender":
        """
        Fits sparse user-movie centered interaction matrices from a ratings DataFrame.
        Enables strict train/val/test temporal split isolation during evaluation.
        """
        if ratings_df.empty:
            raise ValueError("Cannot fit CollaborativeRecommender with an empty ratings DataFrame.")

        # Require standard columns
        for col in ["userId", "movieId", "rating"]:
            if col not in ratings_df.columns:
                raise ValueError(f"Ratings DataFrame missing required column: {col}")

        # Compute user mean ratings
        user_means = ratings_df.groupby("userId")["rating"].transform("mean")
        ratings_copy = ratings_df.copy()
        ratings_copy["centered"] = ratings_copy["rating"] - user_means

        # Construct contiguous integer indices
        unique_users = sorted(ratings_copy["userId"].unique())
        unique_movies = sorted(ratings_copy["movieId"].unique())

        self.user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_map = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.rev_movie_map = {idx: mid for mid, idx in self.movie_map.items()}

        rows = ratings_copy["userId"].map(self.user_map).values
        cols = ratings_copy["movieId"].map(self.movie_map).values
        centered_vals = ratings_copy["centered"].values.astype(np.float32)
        binary_vals = np.ones(len(ratings_copy), dtype=np.float32)

        n_users = len(self.user_map)
        n_movies = len(self.movie_map)

        self.R_centered = sparse.csr_matrix(
            (centered_vals, (rows, cols)), shape=(n_users, n_movies)
        )
        self.R_binary = sparse.csr_matrix(
            (binary_vals, (rows, cols)), shape=(n_users, n_movies)
        )

        self._is_fitted = True
        return self

    def load_or_fit(self) -> "CollaborativeRecommender":
        """
        Loads ratings from data_dir and fits the recommender if not already fitted.
        """
        if self._is_fitted:
            return self

        # Check if precomputed sparse matrix exists
        matrix_path = self.models_dir / "item_collaborative_matrix.npz"
        if matrix_path.exists():
            try:
                self.load_matrix(matrix_path)
            except Exception:
                pass

        ratings_path = self.data_dir / "ratings.csv"
        if not ratings_path.exists():
            raise FileNotFoundError(f"Ratings file not found at: {ratings_path}")

        ratings_df = pd.read_csv(ratings_path)
        return self.fit_from_ratings(ratings_df)

    def is_collaborative_available(self, movie_id: Optional[int]) -> bool:
        """
        Returns True if the movie has interaction data in the fitted collaborative space.
        """
        if not self._is_fitted or movie_id is None:
            return False
        return int(movie_id) in self.movie_map

    def get_pairwise_similarity(self, movie_id_1: Optional[int], movie_id_2: Optional[int]) -> float:
        """
        Computes the shrunk Pearson correlation similarity between two specific movies.
        Returns 0.0 if either movie is not in MovieLens or co-ratings are insufficient.
        """
        if (
            not self._is_fitted
            or movie_id_1 is None
            or movie_id_2 is None
            or movie_id_1 not in self.movie_map
            or movie_id_2 not in self.movie_map
            or movie_id_1 == movie_id_2
        ):
            return 0.0

        idx1 = self.movie_map[movie_id_1]
        idx2 = self.movie_map[movie_id_2]

        # Use precomputed matrix if available
        if self.item_collab_matrix is not None:
            val = self.item_collab_matrix[idx1, idx2]
            return float(val)

        # On-demand calculation
        assert self.R_centered is not None and self.R_binary is not None
        v1 = self.R_centered.getcol(idx1)
        v2 = self.R_centered.getcol(idx2)
        b1 = self.R_binary.getcol(idx1)
        b2 = self.R_binary.getcol(idx2)

        # Co-rating count
        co_count = float((b1.T.dot(b2)).toarray()[0, 0])
        if co_count < 1.0:
            return 0.0

        num = float((v1.T.dot(v2)).toarray()[0, 0])
        denom1 = float((v1.power(2).T.dot(b2)).toarray()[0, 0])
        denom2 = float((b1.T.dot(v2.power(2))).toarray()[0, 0])
        denom = np.sqrt(denom1) * np.sqrt(denom2)

        if denom <= 1e-9:
            return 0.0

        raw_sim = num / denom
        shrinkage = min(co_count, self.tau) / self.tau
        return float(max(0.0, raw_sim * shrinkage))

    def get_top_collaborative_neighbors(
        self, movie_id: Optional[int], top_n: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Retrieves the top-N collaborative neighbors for a given MovieLens movieId.
        Returns list of (movieId, similarity_score) sorted strictly descending.
        """
        if not self._is_fitted or movie_id is None or movie_id not in self.movie_map:
            return []

        target_idx = self.movie_map[movie_id]

        # Use precomputed matrix if available
        if self.item_collab_matrix is not None:
            row = self.item_collab_matrix.getrow(target_idx)
            indices = row.indices
            data = row.data
            if len(data) == 0:
                return []
            sorted_order = np.argsort(data)[::-1][:top_n]
            return [(self.rev_movie_map[indices[i]], float(data[i])) for i in sorted_order]

        # Vectorized on-demand computation
        assert self.R_centered is not None and self.R_binary is not None
        target_vec = self.R_centered.getcol(target_idx)
        target_bin = self.R_binary.getcol(target_idx)

        # Co-rated counts: 1 x n_movies
        co_counts = (target_bin.T.dot(self.R_binary)).toarray().flatten()

        # Numerator: 1 x n_movies
        num = (target_vec.T.dot(self.R_centered)).toarray().flatten()

        # Denominator components
        sq_target = target_vec.power(2)
        target_var = (sq_target.T.dot(self.R_binary)).toarray().flatten()

        sq_all = self.R_centered.power(2)
        cand_var = (target_bin.T.dot(sq_all)).toarray().flatten()

        denom = np.sqrt(target_var) * np.sqrt(cand_var)
        valid = (denom > 1e-9) & (co_counts >= 1)

        sims = np.zeros(len(self.movie_map), dtype=np.float32)
        sims[valid] = num[valid] / denom[valid]

        shrinkage = np.minimum(co_counts, self.tau) / self.tau
        shrunk_sims = np.maximum(0.0, sims * shrinkage)
        shrunk_sims[target_idx] = 0.0  # Exclude self

        top_indices = np.argsort(shrunk_sims)[::-1][:top_n]
        results = []
        for idx in top_indices:
            score = float(shrunk_sims[idx])
            if score > 0.0:
                results.append((self.rev_movie_map[idx], score))
        return results

    def precompute_sparse_matrix(self, top_k_per_item: int = 100) -> sparse.csr_matrix:
        """
        Precomputes and returns a sparse top-K item-item collaborative matrix.
        Retains only up to top_k_per_item neighbors per movie to bound memory (~3-5 MB).
        """
        if not self._is_fitted:
            self.load_or_fit()

        n_movies = len(self.movie_map)
        rows_list = []
        cols_list = []
        data_list = []

        for mid, idx in self.movie_map.items():
            neighbors = self.get_top_collaborative_neighbors(mid, top_n=top_k_per_item)
            for n_mid, sim in neighbors:
                if sim > 0.0:
                    rows_list.append(idx)
                    cols_list.append(self.movie_map[n_mid])
                    data_list.append(sim)

        self.item_collab_matrix = sparse.csr_matrix(
            (data_list, (rows_list, cols_list)), shape=(n_movies, n_movies), dtype=np.float32
        )
        return self.item_collab_matrix

    def save_matrix(self, filepath: Optional[Path | str] = None) -> Path:
        """
        Saves precomputed sparse collaborative matrix to disk.
        """
        if self.item_collab_matrix is None:
            self.precompute_sparse_matrix()

        out_path = Path(filepath) if filepath else (self.models_dir / "item_collaborative_matrix.npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(out_path, self.item_collab_matrix)
        return out_path

    def load_matrix(self, filepath: Optional[Path | str] = None) -> None:
        """
        Loads precomputed sparse collaborative matrix from disk.
        """
        in_path = Path(filepath) if filepath else (self.models_dir / "item_collaborative_matrix.npz")
        if in_path.exists():
            self.item_collab_matrix = sparse.load_npz(in_path)
