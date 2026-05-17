# data folder

This folder holds the dataset for the project. The actual data files are not committed to git (they are too large, around 820 MiB total). What is committed is the MANIFEST and the download script, so anyone with access can reproduce the same files.

## Contents

- `MANIFEST.yaml` is committed. It lists the files, sizes, and SHA-256 hashes of the dataset, plus the source. This is what tells you which version of the data you have.
- `../scripts/download_data.sh` is committed. Running it downloads everything into `data/raw/`.
- `raw/` is gitignored. After running the download script, it contains
  `topics.tsv`, `qrels.txt`, `subsample.parquet`, and `subsample_embeddings.parquet`.
- `processed/`, `interim/`, `external/` are gitignored. They are empty for now and will be used for intermediate or computed files later.

## Dataset

The dataset is the Radboud validation split of the Open Web Search Curlie 2025 collection (ows-curlie-2025 / radboud-val). The MANIFEST file has the exact versions and hashes.
