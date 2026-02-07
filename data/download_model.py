"""Pre-download a word embedding model so the app starts instantly.

Usage:
    python data/download_model.py
    python data/download_model.py glove-wiki-gigaword-100
"""
import sys

import gensim.downloader as api


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "glove-wiki-gigaword-50"
    print(f"Downloading '{model_name}' ...")
    api.load(model_name)
    print("Done. Model is cached for future use.")


if __name__ == "__main__":
    main()
