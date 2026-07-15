from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEPARATE_DISEASE_DIR = PROJECT_ROOT / "data" / "separate_disease"
TARGET_FOLDER = "3_gene_related_disease"
OUTPUT_CSV = PROJECT_ROOT / "data" / "threshold.csv"

SIMILARITY_DIR_CANDIDATES = [
    PROJECT_ROOT / "data" / "similarity",
    PROJECT_ROOT / "data" / "Similariy",
]

TPM_AVG_FILE = "TPM_disease_avg_similarity_summary.csv"


def find_similarity_dir() -> Path:
    for path in SIMILARITY_DIR_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a similarity directory. Checked: "
        + ", ".join(str(path) for path in SIMILARITY_DIR_CANDIDATES)
    )


def build_disease_folder_map(root: Path) -> dict[str, str]:
    disease_to_folder: dict[str, str] = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.endswith("_gene_related_disease"):
            continue
        for csv_file in folder.glob("*.csv"):
            disease_to_folder[csv_file.stem] = folder.name
    return disease_to_folder


def summarize_tpm_average(csv_path: Path, disease_to_folder: dict[str, str]) -> dict[str, dict]:
    folder_rows: dict[str, list[dict]] = defaultdict(list)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            disease_id = row["diseaseFromSourceMappedId"]
            folder = disease_to_folder.get(disease_id)
            if not folder:
                continue

            folder_rows[folder].append(
                {
                    "disease_id": disease_id,
                    "file_name": row["file_name"],
                    "avg_pairwise_similarity": float(row["avg_pairwise_similarity"]),
                    "n_pairs_used": int(row["n_pairs_used"]),
                    "n_genes_original": int(row["n_genes_original"]),
                    "n_genes_used": int(row["n_genes_used"]),
                    "n_missing_genes": int(row["n_missing_genes"]),
                }
            )

    summary: dict[str, dict] = {}
    for folder, rows in folder_rows.items():
        min_row = min(rows, key=lambda item: item["avg_pairwise_similarity"])
        max_row = max(rows, key=lambda item: item["avg_pairwise_similarity"])
        summary[folder] = {
            "n_diseases": len(rows),
            "min": min_row,
            "max": max_row,
        }
    return summary


def write_summary_csv(output_path: Path, summary: dict[str, dict]) -> None:
    fieldnames = [
        "folder",
        "n_diseases",
        "min_similarity",
        "min_disease_id",
        "min_n_pairs_used",
        "max_similarity",
        "max_disease_id",
        "max_n_pairs_used",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for folder in sorted(summary):
            item = summary[folder]
            writer.writerow(
                {
                    "folder": folder,
                    "n_diseases": item["n_diseases"],
                    "min_similarity": item["min"]["avg_pairwise_similarity"],
                    "min_disease_id": item["min"]["disease_id"],
                    "min_n_pairs_used": item["min"]["n_pairs_used"],
                    "max_similarity": item["max"]["avg_pairwise_similarity"],
                    "max_disease_id": item["max"]["disease_id"],
                    "max_n_pairs_used": item["max"]["n_pairs_used"],
                }
            )


def main() -> None:
    similarity_dir = find_similarity_dir()
    disease_to_folder = build_disease_folder_map(SEPARATE_DISEASE_DIR)
    summary = summarize_tpm_average(similarity_dir / TPM_AVG_FILE, disease_to_folder)

    target = summary[TARGET_FOLDER]
    print(f"Folder: {TARGET_FOLDER}")
    print(
        "Min similarity:",
        target["min"]["avg_pairwise_similarity"],
        "| disease:",
        target["min"]["disease_id"],
    )
    print(
        "Max similarity:",
        target["max"]["avg_pairwise_similarity"],
        "| disease:",
        target["max"]["disease_id"],
    )

    write_summary_csv(OUTPUT_CSV, summary)
    print(f"Saved summary: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
