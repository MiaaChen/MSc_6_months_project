# Data Availability

This repository is intended to contain reproducible research code, lightweight examples, metadata, and selected final summary outputs.

Do not commit large generated or downloaded files, including:

- full gene-gene matrix files
- NCBI cache directories
- per-disease generated intermediate folders
- model rerun output folders
- notebook checkpoints
- operating system files such as `.DS_Store`

Suggested locations:

- `data/`: small public example inputs and selected final summary CSV files
- `database/`: local-only external data needed to rerun the full analysis
- `metadata/`: dependency and documentation files

Included expression similarity matrix:

- `database/Matrix/gene_gene_correlation_TPM_updated.csv`

This is the reduced TPM matrix used by the iteration workflow. Larger raw/proxy matrix files and the log2/z-score matrix copies are excluded.
