# Disease Gene Similarity and Genomic Neighbour Analysis

This repository contains code and selected data for a 6-month research project on disease-associated genes. The project analyses gene expression similarity, genomic neighbours, and iterative candidate-gene replacement performance for disease gene sets.

The main analysis uses disease-gene association files, a TPM gene-gene expression similarity matrix, and NCBI genomic annotation data to test whether neighbouring or similar genes can recover known disease-associated genes.

## Repository Structure

```text
data/       Raw/project data used by the notebooks, including parquet source files
database/   Disease-gene CSVs and the TPM matrix used for expression similarity
metadata/   Dependency and data availability notes
scrip/      Analysis notebooks and Python helper scripts
```

Current included data:

- `data/*.snappy.parquet`: source disease-gene association data
- `database/Disease-genes/`: per-disease gene CSV files
- `database/Matrix/gene_gene_correlation_TPM_updated.csv`: TPM expression similarity matrix

## Main Workflow

1. Prepare disease-gene association tables.
2. Split disease-gene associations into per-disease CSV files.
3. Check disease gene counts and matrix coverage.
4. Calculate gene-gene expression similarity.
5. Generate NCBI genomic neighbour candidates.
6. Build disease-specific matrix subsets.
7. Run iterative candidate-gene replacement experiments.
8. Evaluate hit percentage, accuracy, and causal-gene recovery.

## Key Notebooks

- `scrip/Histogram.ipynb`: builds disease gene count summaries and histograms.
- `scrip/Separate_disease.ipynb`: separates source disease-gene data into disease-level CSV files.
- `scrip/Similarity.ipynb`: calculates disease-level gene similarity scores.
- `scrip/Random_gene_similarity.ipynb`: compares disease gene similarity against random gene sets.
- `scrip/Threshold_similarity.ipynb`: summarises similarity thresholds by disease group.
- `scrip/find_neighbors_matrix_filtered.ipynb`: creates NCBI neighbour files filtered by matrix availability.
- `scrip/itterations.ipynb`: runs repeated candidate-gene replacement experiments using TPM similarity.
- `scrip/hit_percent_by_gene_count.ipynb`: summarises hit percentage by disease gene count.
- `scrip/accuracy.ipynb`: evaluates exact causal-set recovery from iteration outputs.
- `scrip/accuracy_by_gene_count.ipynb`: groups accuracy results by number of disease genes.
- `scrip/evaluate_final_gene_set_accuracy.ipynb`: evaluates final selected gene sets.
- `scrip/rerun_pocket_models_from_separate_disease_no_missing.ipynb`: reruns pocket-model experiments after filtering missing matrix genes.

## Python Scripts

- `scrip/enrich_ncbi_neighbors.py`: downloads/caches NCBI data and enriches disease genes with nearby genomic neighbours.
- `scrip/disease_template_support.py`: shared helper functions for disease-level neighbour and matrix workflows.
- `scrip/rerun_pocket_models_from_separate_disease.py`: command-line version of the pocket-model rerun workflow.
- `scrip/plot_pocket_similarity_graphs.py`: plots model iteration outputs.
- `scrip/similarity_zscore.py`: calculates Z-score based similarity if the Z-score matrix is available locally.
- `scrip/summarize_similarity_min_max.py`: summarises minimum and maximum disease similarity values.

## Requirements

The project was developed with Python 3.13. Main dependencies are listed in `metadata/requirements.txt`.

Install dependencies with:

```bash
pip install -r metadata/requirements.txt
```

If reading the parquet files directly with pandas, install a parquet engine if needed:

```bash
pip install pyarrow
```

## Usage Notes

Run notebooks from the repository root. Some notebooks were originally developed with absolute local paths, so check the first configuration cell before running and update `PROJECT_ROOT` to your local repository path.

The TPM matrix used by the iteration workflow is:

```text
database/Matrix/gene_gene_correlation_TPM_updated.csv
```

Example NCBI neighbour enrichment command:

```bash
python scrip/enrich_ncbi_neighbors.py \
  --input-root data/separate_disease \
  --output-root data/separate_disease_ncbi_neighbors \
  --cache-dir database/ncbi_cache \
  --matrix-file database/Matrix/gene_gene_correlation_TPM_updated.csv
```

## Data Availability

The repository includes selected input data and the reduced TPM matrix needed for the main iteration workflow. Large generated folders, NCBI caches, and intermediate model outputs should not be committed unless specifically needed.

See `metadata/data_availability.md` for more detail.

## Author

Your Name  
King's College London  
6-month research project
