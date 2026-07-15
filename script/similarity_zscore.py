import os
import itertools
import numpy as np
import pandas as pd

# Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_dir = os.path.join(project_root, "data", "disease_gene_table")
matrix_file = os.path.join(project_root, "database", "gene_gene_correlation_Zscore_filtered.txt")
output_dir = os.path.join(project_root, "data", "similarity_Zscore")

os.makedirs(output_dir, exist_ok=True)

summary_out = os.path.join(output_dir, "disease_avg_similarity_summary.csv")
pairs_out = os.path.join(output_dir, "disease_pairwise_similarity_details.csv")

print("Loading gene-gene matrix...")
sim_matrix = pd.read_csv(matrix_file, sep="\t", index_col=0)

sim_matrix.index = sim_matrix.index.astype(str).str.strip()
sim_matrix.columns = sim_matrix.columns.astype(str).str.strip()

print(f"Matrix shape: {sim_matrix.shape}")

summary_rows = []
pair_rows = []

files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])

print(f"Found {len(files)} disease files")

for fname in files:
    fpath = os.path.join(input_dir, fname)

    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        print(f"Skipping {fname}: cannot read file ({e})")
        continue
    if "targetId" not in df.columns:
        print(f"Skipping {fname}: no targetId column")
        continue

    if "diseaseFromSourceMappedId" in df.columns and df["diseaseFromSourceMappedId"].notna().any():
        disease_id = str(df["diseaseFromSourceMappedId"].dropna().iloc[0]).strip()
    else:
        disease_id = os.path.splitext(fname)[0]

    genes = (
        df["targetId"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    n_genes_original = len(genes)

    genes_in_matrix = [g for g in genes if g in sim_matrix.index and g in sim_matrix.columns]
    missing_genes = [g for g in genes if g not in sim_matrix.index or g not in sim_matrix.columns]

    n_genes_used = len(genes_in_matrix)

    if n_genes_used < 2:
        summary_rows.append({
            "diseaseFromSourceMappedId": disease_id,
            "file_name": fname,
            "n_genes_original": n_genes_original,
            "n_genes_used": n_genes_used,
            "n_missing_genes": len(missing_genes),
            "n_pairs_used": 0,
            "avg_pairwise_similarity": np.nan
        })
        continue

    sims = []

    for g1, g2 in itertools.combinations(genes_in_matrix, 2):
        val = sim_matrix.loc[g1, g2]

        if pd.notna(val):
            sims.append(float(val))
            pair_rows.append({
                "diseaseFromSourceMappedId": disease_id,
                "file_name": fname,
                "gene1": g1,
                "gene2": g2,
                "similarity_zscore": float(val)
            })

    avg_sim = np.mean(sims) if len(sims) > 0 else np.nan

    summary_rows.append({
        "diseaseFromSourceMappedId": disease_id,
        "file_name": fname,
        "n_genes_original": n_genes_original,
        "n_genes_used": n_genes_used,
        "n_missing_genes": len(missing_genes),
        "n_pairs_used": len(sims),
        "avg_pairwise_similarity": avg_sim
    })

    print(f"Done: {fname} | genes used={n_genes_used} | pairs={len(sims)} | avg={avg_sim}")

# Save output
summary_df = pd.DataFrame(summary_rows)
pairs_df = pd.DataFrame(pair_rows)

summary_df.to_csv(summary_out, index=False)
pairs_df.to_csv(pairs_out, index=False)

print("\nFinished.")
print(f"Summary saved to: {summary_out}")
print(f"Pair details saved to: {pairs_out}")
