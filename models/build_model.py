"""
Model Precomputation and Serialization Script
---------------------------------------------
Executes the data preprocessing, constructs the feature soup, fits the TF-IDF
vectorizer, and serializes the artifacts into the models/ directory for instant
runtime loading in the web application.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from recommender import MovieRecommender


def build_and_save():
    print("=" * 60)
    print("Movie Recommender Pipeline: Building and Caching Artifacts")
    print("=" * 60)

    start_time = time.time()
    recommender = MovieRecommender(
        data_dir=PROJECT_ROOT / "data" / "ml-latest-small",
        models_dir=PROJECT_ROOT / "models",
    )

    print("\n1. Loading and preprocessing MovieLens dataset...")
    df = recommender.load_and_preprocess()
    print(f"[OK] Preprocessed {len(df):,} movies.")
    print(f"     Columns: {list(df.columns)}")

    print("\n2. Fitting TF-IDF Vectorizer and computing sparse feature matrix...")
    recommender.fit()
    print(f"[OK] TF-IDF Matrix shape: {recommender.tfidf_matrix.shape}")
    print(f"     Vocabulary size: {len(recommender.tfidf_vectorizer.vocabulary_):,} features")

    print("\n3. Serializing artifacts to disk...")
    recommender.save_artifacts()

    elapsed = round(time.time() - start_time, 2)
    print(f"\n[SUCCESS] Pipeline completed in {elapsed} seconds!")
    print("=" * 60)


if __name__ == "__main__":
    build_and_save()
