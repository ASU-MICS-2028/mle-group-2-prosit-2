"""Build data/processed/features.parquet and the fold splits; fails loudly on any leakage.

    python -m scripts.build_features        (same as: python -m src.features, or make features)
"""
from src.features import main

if __name__ == "__main__":
    main()
