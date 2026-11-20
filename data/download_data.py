"""
MovieLens Dataset Downloader & Validator
----------------------------------------
Downloads and validates the official MovieLens Latest Small dataset from
GroupLens Research (University of Minnesota).

Usage Terms & Attribution:
- Dataset: MovieLens Latest Small (100k ratings, 3.6k tags, 9.7k movies)
- Source: https://grouplens.org/datasets/movielens/latest/
- Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
  History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4.
- License: Permitted for academic, research, and non-commercial educational use.
"""

import os
import zipfile
import requests
import pandas as pd
from pathlib import Path

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
BASE_DIR = Path(__file__).resolve().parent
TARGET_DIR = BASE_DIR / "ml-latest-small"
ZIP_PATH = BASE_DIR / "ml-latest-small.zip"

REQUIRED_SCHEMAS = {
    "movies.csv": ["movieId", "title", "genres"],
    "ratings.csv": ["userId", "movieId", "rating", "timestamp"],
    "tags.csv": ["userId", "movieId", "tag", "timestamp"],
    "links.csv": ["movieId", "imdbId", "tmdbId"],
}


def download_movielens_dataset(force_download: bool = False) -> Path:
    """
    Downloads and extracts the MovieLens Latest Small dataset.

    Parameters:
        force_download (bool): If True, re-downloads even if files exist.

    Returns:
        Path: Path to extracted dataset directory.
    """
    os.makedirs(BASE_DIR, exist_ok=True)

    all_files_exist = TARGET_DIR.exists() and all(
        (TARGET_DIR / fname).is_file() for fname in REQUIRED_SCHEMAS.keys()
    )

    if all_files_exist and not force_download:
        print(f"[OK] MovieLens dataset already exists at {TARGET_DIR}")
        validate_dataset(TARGET_DIR)
        return TARGET_DIR

    print(f"[*] Downloading MovieLens dataset from {DATA_URL} ...")
    response = requests.get(DATA_URL, stream=True, timeout=30)
    response.raise_for_status()

    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
    print(f"[OK] Downloaded archive to {ZIP_PATH} ({os.path.getsize(ZIP_PATH):,} bytes)")

    print(f"[*] Extracting {ZIP_PATH.name} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(BASE_DIR)
    print(f"[OK] Successfully extracted dataset to {TARGET_DIR}")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    validate_dataset(TARGET_DIR)
    return TARGET_DIR


def validate_dataset(dataset_dir: Path) -> dict:
    """
    Performs comprehensive integrity validation on all MovieLens tables:
    - Checks file existence and non-zero size
    - Validates expected schema and column names
    - Checks for duplicate records and null values
    - Displays dataset statistics and GroupLens attribution
    """
    stats = {}
    print("\n" + "=" * 65)
    print(" MOVIELENS DATASET INTEGRITY & VALIDATION REPORT")
    print(" GroupLens Research Group | University of Minnesota")
    print("=" * 65)

    for fname, expected_cols in REQUIRED_SCHEMAS.items():
        fpath = dataset_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"[ERROR] Missing required file: {fpath}")

        file_size = fpath.stat().st_size
        if file_size == 0:
            raise ValueError(f"[ERROR] File is empty: {fpath}")

        df = pd.read_csv(fpath)

        # Check required columns
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"[ERROR] {fname} missing required columns: {missing_cols}")

        # Duplicate check on primary ID if applicable
        duplicate_count = 0
        if fname == "movies.csv":
            duplicate_count = df.duplicated(subset=["movieId"]).sum()
        elif fname == "links.csv":
            duplicate_count = df.duplicated(subset=["movieId"]).sum()

        null_counts = df[expected_cols].isnull().sum().to_dict()

        stats[fname] = {
            "rows": len(df),
            "columns": list(df.columns),
            "file_size_kb": round(file_size / 1024, 2),
            "duplicate_ids": duplicate_count,
            "null_counts": null_counts,
        }

        print(f"\n* File: {fname}")
        print(f"  - Size: {stats[fname]['file_size_kb']} KB")
        print(f"  - Total Records: {len(df):,}")
        print(f"  - Columns: {', '.join(df.columns)}")
        print(f"  - Duplicate Primary Keys: {duplicate_count}")
        print(f"  - Null Values: {null_counts}")

    print("\n" + "-" * 65)
    print(" ATTRIBUTION & USAGE TERMS:")
    print(" - Source: GroupLens Research (https://grouplens.org/datasets/movielens/)")
    print(" - License: Educational, Academic, and Non-Commercial Research Use")
    print(" - Poster & Description Note: MovieLens does not provide posters or full")
    print("   plot summaries. All ratings, genres, and tags are authentic.")
    print("-" * 65)
    print("[SUCCESS] All dataset files validated successfully!\n")
    return stats


if __name__ == "__main__":
    download_movielens_dataset()
