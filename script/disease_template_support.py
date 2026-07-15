from __future__ import annotations

import csv
import importlib.util
import math
import random
from pathlib import Path
from types import ModuleType, SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd


def require_file_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def resolve_input_file(path: Path, label: str, allow_checkpoint: bool = False):
    if path.exists():
        return path

    if allow_checkpoint:
        checkpoint_path = path.parent / ".ipynb_checkpoints" / f"{path.stem}-checkpoint{path.suffix}"
        if checkpoint_path.exists():
            print(f"Using checkpoint {label}: {checkpoint_path}")
            return checkpoint_path

    require_file_exists(path, label)
    return path


def ordered_unique(values):
    seen = set()
    output = []

    for value in values:
        key = tuple(sorted(value.items())) if isinstance(value, dict) else value
        if key not in seen:
            seen.add(key)
            output.append(value)

    return output


def find_unique_disease_csv(project_root: Path, disease_id: str) -> Path:
    disease_id = disease_id.strip()
    if not disease_id:
        raise ValueError("Set DISEASE_ID in the notebook first.")

    separate_disease_root = project_root / "data" / "separate_disease"
    require_file_exists(separate_disease_root, "separate_disease root")

    matches = sorted(
        path
        for path in separate_disease_root.rglob(f"{disease_id}.csv")
        if ".ipynb_checkpoints" not in path.parts
    )

    if not matches:
        raise FileNotFoundError(f"No causal CSV found for {disease_id} under {separate_disease_root}")

    if len(matches) > 1:
        match_text = "\n".join(str(path) for path in matches)
        raise ValueError(f"More than one causal CSV matched {disease_id}.csv:\n{match_text}")

    return matches[0]


def build_config(
    disease_id: str,
    project_root: Path,
    n_rounds: int = 5000,
    random_seed: int = 42,
    max_genes_to_change: int = 9,
    neighbor_count_per_side: int = 5,
    assembly: str = "GCF_000001405.40",
):
    project_root = Path(project_root).expanduser().resolve()
    causal_csv = find_unique_disease_csv(project_root, disease_id)
    pocket_dir = causal_csv.parent
    file_stem = causal_csv.stem

    return SimpleNamespace(
        disease_id=disease_id.strip(),
        project_root=project_root,
        causal_csv=causal_csv,
        pocket_dir=pocket_dir,
        file_stem=file_stem,
        summary_csv=project_root / "data" / "Similariy" / "TPM_disease_avg_similarity_summary.csv",
        source_matrix=project_root / "database" / "Matrix" / "gene_gene_correlation_TPM_updated.csv",
        matrix_path=pocket_dir / f"{file_stem}_gene_gene_correlation_TPM_updated_subset.csv",
        pocket_output_csv=pocket_dir / f"{file_stem}_pocket_candidates.csv",
        round_output_csv=pocket_dir / f"{file_stem}_pocket_similarity_rounds.csv",
        graph_output_png=pocket_dir / f"{file_stem}_pocket_similarity_rounds.png",
        graph_output_png_simple=pocket_dir / f"{file_stem}_pocket_similarity_rounds_simple.png",
        graph_output_png_sketch=pocket_dir / f"{file_stem}_pocket_similarity_rounds_sketch.png",
        graph_output_png_kept_only=pocket_dir / f"{file_stem}_pocket_similarity_rounds_kept_only.png",
        script_path=project_root / "scrip" / "enrich_ncbi_neighbors.py",
        cache_dir=project_root / "database" / "ncbi_cache",
        assembly=assembly,
        neighbor_count_per_side=neighbor_count_per_side,
        neighbor_csv=pocket_dir / f"{file_stem}_left5_right5_ncbi_neighbors_with_ensembl.csv",
        n_rounds=n_rounds,
        random_seed=random_seed,
        max_genes_to_change=max_genes_to_change,
        effective_disease_id=file_stem,
    )


def describe_config(config):
    print(f"Configured disease: {config.disease_id}")
    print(f"Causal CSV: {config.causal_csv}")
    print(f"Pocket folder: {config.pocket_dir}")
    print(f"Neighbor CSV: {config.neighbor_csv}")
    print(f"Matrix CSV: {config.matrix_path}")
    print(f"Summary CSV: {config.summary_csv}")
    print(f"Rounds CSV: {config.round_output_csv}")
    print(f"Main graph: {config.graph_output_png}")


def load_ncbi_neighbors_module(script_path: Path) -> ModuleType:
    require_file_exists(script_path, "Neighbor enrichment script")
    spec = importlib.util.spec_from_file_location("enrich_ncbi_neighbors", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_ensembl_id(value: str) -> str:
    if not value:
        return ""

    for item in value.split("|"):
        item = item.strip()
        if item.startswith("ENSG"):
            return item

    return ""


def first_matrix_ensembl_id(value: str, matrix_gene_ids: set[str] | None = None) -> str:
    if not value:
        return ""

    for item in value.split("|"):
        item = item.strip()
        if item.startswith("ENSG") and (matrix_gene_ids is None or item in matrix_gene_ids):
            return item

    return ""


def load_matrix_gene_ids(matrix_path: Path) -> set[str]:
    require_file_exists(matrix_path, "Source matrix")
    header = pd.read_csv(matrix_path, nrows=0)
    return {str(column).strip() for column in header.columns[1:] if str(column).strip()}


def matching_matrix_ensembl_ids(ensembl_ids, matrix_gene_ids: set[str]) -> list[str]:
    return [
        ensembl_id
        for ensembl_id in ensembl_ids
        if ensembl_id and ensembl_id in matrix_gene_ids
    ]


def format_similarity(value):
    if value is None:
        return ""

    return f"{value:.12f}"


def serialize_candidates(candidates, field):
    values = []

    for candidate in candidates:
        value = candidate.get(field, "")
        values.append("" if value is None else str(value))

    return "|".join(values)


def infer_disease_id(causal_rows, fallback_id: str, causal_csv_path: Path):
    csv_ids = ordered_unique(
        [
            (row.get("diseaseFromSourceMappedId") or "").strip()
            for row in causal_rows
            if (row.get("diseaseFromSourceMappedId") or "").strip()
        ]
    )

    if csv_ids:
        csv_id = csv_ids[0]
        if fallback_id and fallback_id != csv_id:
            print(f"Using disease ID from causal CSV: {csv_id} (instead of configured {fallback_id})")
        return csv_id

    return fallback_id or causal_csv_path.stem


def iter_neighbor_rows_with_ensembl(
    seed: dict,
    genes_by_chr: dict,
    neighbor_count_per_side: int,
    ncbi_neighbors,
    matrix_gene_ids: set[str],
):
    counts = {"upstream": 0, "downstream": 0}
    for neighbor_info in ncbi_neighbors.iter_neighbor_rows(seed, genes_by_chr, len(genes_by_chr[seed["chr"]])):
        neighbor = neighbor_info["neighbor"]
        matrix_ensembl_ids = matching_matrix_ensembl_ids(neighbor["ensembl_ids"], matrix_gene_ids)
        if not matrix_ensembl_ids:
            continue

        direction = neighbor_info["neighbor_direction"]
        if counts[direction] >= neighbor_count_per_side:
            continue

        counts[direction] += 1
        yield {
            **neighbor_info,
            "neighbor_rank": counts[direction],
            "matrix_ensembl_ids": matrix_ensembl_ids,
        }

        if all(count >= neighbor_count_per_side for count in counts.values()):
            break


def generate_neighbor_csv(config):
    ncbi_neighbors = load_ncbi_neighbors_module(config.script_path)
    matrix_gene_ids = load_matrix_gene_ids(config.source_matrix)
    headers = ncbi_neighbors.build_headers()
    gene_report_path = ncbi_neighbors.ensure_gene_report_cache(
        cache_dir=config.cache_dir,
        filename="ncbi_human_gene_report.jsonl",
        headers=headers,
        refresh_cache=False,
    )
    genome_zip = ncbi_neighbors.ensure_zip_download(
        cache_dir=config.cache_dir,
        filename=f"{config.assembly}_annotation.zip",
        candidate_urls=ncbi_neighbors.genome_package_urls(config.assembly, f"{config.assembly}_annotation.zip"),
        headers=headers,
        refresh_cache=False,
    )

    ensembl_to_gene_ids, gene_records, symbol_to_gene_ids = ncbi_neighbors.build_gene_indexes(
        gene_report_path=gene_report_path,
        assembly=config.assembly,
    )
    genes_by_chr, gene_index = ncbi_neighbors.build_gff_index(
        genome_zip=genome_zip,
        assembly=config.assembly,
        gene_records=gene_records,
    )
    local_symbol_map = ncbi_neighbors.build_local_symbol_map(config.project_root / "database" / "combined.tsv")

    seed_target_ids = []
    disease_id = config.file_stem
    with config.causal_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            disease_id = (row.get("diseaseFromSourceMappedId") or disease_id).strip() or disease_id
            seed_target_ids.append((row.get("targetId") or "").strip())
    seed_target_ids = ordered_unique(seed_target_ids)

    rows = []
    unresolved = []
    for seed_target_id in seed_target_ids:
        seed_gene_id, reason = ncbi_neighbors.choose_seed_gene_id(
            seed_target_id,
            ensembl_to_gene_ids=ensembl_to_gene_ids,
            gene_index=gene_index,
            local_symbol_map=local_symbol_map,
            symbol_to_gene_ids=symbol_to_gene_ids,
        )
        if not seed_gene_id:
            unresolved.append({"seed_targetId": seed_target_id, "reason": reason or "unresolved"})
            continue

        seed = gene_index[seed_gene_id]
        for neighbor_info in iter_neighbor_rows_with_ensembl(
            seed,
            genes_by_chr,
            config.neighbor_count_per_side,
            ncbi_neighbors,
            matrix_gene_ids,
        ):
            neighbor = neighbor_info["neighbor"]
            matrix_ensembl_ids = neighbor_info["matrix_ensembl_ids"]
            rows.append(
                {
                    "diseaseFromSourceMappedId": disease_id,
                    "seed_targetId": seed_target_id,
                    "seed_ncbi_gene_id": seed["gene_id"],
                    "seed_symbol": seed["symbol"],
                    "seed_biotype": seed["biotype"],
                    "seed_chr": seed["chr"],
                    "seed_start": seed["start"],
                    "seed_end": seed["end"],
                    "seed_strand": seed["strand"],
                    "neighbor_direction": neighbor_info["neighbor_direction"],
                    "neighbor_rank": neighbor_info["neighbor_rank"],
                    "neighbor_ncbi_gene_id": neighbor["gene_id"],
                    "neighbor_ensembl_ids": "|".join(matrix_ensembl_ids),
                    "neighbor_symbol": neighbor["symbol"],
                    "neighbor_biotype": neighbor["biotype"],
                    "neighbor_chr": neighbor["chr"],
                    "neighbor_start": neighbor["start"],
                    "neighbor_end": neighbor["end"],
                    "neighbor_strand": neighbor["strand"],
                    "distance_bp": ncbi_neighbors.distance_bp(seed, neighbor),
                }
            )

    fieldnames = [
        "diseaseFromSourceMappedId",
        "seed_targetId",
        "seed_ncbi_gene_id",
        "seed_symbol",
        "seed_biotype",
        "seed_chr",
        "seed_start",
        "seed_end",
        "seed_strand",
        "neighbor_direction",
        "neighbor_rank",
        "neighbor_ncbi_gene_id",
        "neighbor_ensembl_ids",
        "neighbor_symbol",
        "neighbor_biotype",
        "neighbor_chr",
        "neighbor_start",
        "neighbor_end",
        "neighbor_strand",
        "distance_bp",
    ]

    config.neighbor_csv.parent.mkdir(parents=True, exist_ok=True)
    with config.neighbor_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resolved seed genes: {len(seed_target_ids) - len(unresolved)}/{len(seed_target_ids)}")
    print(f"Rows written: {len(rows)}")
    print(f"Neighbor CSV: {config.neighbor_csv}")
    if unresolved:
        print("Unresolved seeds:")
        for item in unresolved:
            print(item)

    return config.neighbor_csv


def choose_initial_random_selection(pockets):
    selection = {}

    for pocket_id, pocket in pockets.items():
        selection[pocket_id] = random.choice(pocket["candidates"])

    return selection


def build_pockets(causal_rows, neighbor_rows, matrix_gene_ids: set[str] | None = None):
    pockets = {}

    for row in causal_rows:
        seed = row["targetId"].strip()
        pockets[seed] = {
            "seed_targetId": seed,
            "candidates": [
                {
                    "candidate_type": "causal",
                    "candidate_label": seed,
                    "matrix_gene_id": seed,
                    "neighbor_direction": "causal",
                    "neighbor_rank": 0,
                    "neighbor_symbol": "",
                    "neighbor_ncbi_gene_id": "",
                }
            ],
        }

    for row in neighbor_rows:
        seed = row["seed_targetId"].strip()
        if seed not in pockets:
            continue

        matrix_gene_id = first_matrix_ensembl_id(row["neighbor_ensembl_ids"].strip(), matrix_gene_ids)
        if not matrix_gene_id:
            continue

        pockets[seed]["candidates"].append(
            {
                "candidate_type": "nearby",
                "candidate_label": row["neighbor_ensembl_ids"].strip()
                or row["neighbor_symbol"].strip()
                or row["neighbor_ncbi_gene_id"].strip(),
                "matrix_gene_id": matrix_gene_id,
                "neighbor_direction": row["neighbor_direction"].strip(),
                "neighbor_rank": int(row["neighbor_rank"]),
                "neighbor_symbol": row["neighbor_symbol"].strip(),
                "neighbor_ncbi_gene_id": row["neighbor_ncbi_gene_id"].strip(),
            }
        )

    for pocket in pockets.values():
        pocket["candidates"] = ordered_unique(pocket["candidates"])

    return pockets


def write_pocket_table(pockets, output_path: Path):
    fieldnames = [
        "seed_targetId",
        "candidate_index",
        "candidate_type",
        "candidate_label",
        "matrix_gene_id",
        "neighbor_direction",
        "neighbor_rank",
        "neighbor_symbol",
        "neighbor_ncbi_gene_id",
    ]
    rows = []

    for seed_target_id, pocket in pockets.items():
        for idx, candidate in enumerate(pocket["candidates"], start=1):
            rows.append(
                {
                    "seed_targetId": seed_target_id,
                    "candidate_index": idx,
                    "candidate_type": candidate["candidate_type"],
                    "candidate_label": candidate["candidate_label"],
                    "matrix_gene_id": candidate["matrix_gene_id"],
                    "neighbor_direction": candidate["neighbor_direction"],
                    "neighbor_rank": candidate["neighbor_rank"],
                    "neighbor_symbol": candidate["neighbor_symbol"],
                    "neighbor_ncbi_gene_id": candidate["neighbor_ncbi_gene_id"],
                }
            )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_baseline_similarity(summary_csv: Path, disease_id: str):
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["diseaseFromSourceMappedId"].strip() == disease_id:
                return float(row["avg_pairwise_similarity"])

    raise ValueError(f"Baseline similarity not found for {disease_id} in {summary_csv}")


def collect_matrix_gene_ids(pockets):
    gene_ids = set()

    for pocket in pockets.values():
        for candidate in pocket["candidates"]:
            gene_id = candidate["matrix_gene_id"]
            if gene_id:
                gene_ids.add(gene_id)

    return gene_ids


def load_similarity_subset(matrix_path: Path, required_gene_ids):
    if not required_gene_ids:
        return {}

    required_gene_ids = set(required_gene_ids)
    header = pd.read_csv(matrix_path, nrows=0)
    available_gene_ids = {str(col).strip() for col in header.columns[1:]}
    selected_gene_ids = sorted(required_gene_ids & available_gene_ids)
    usecols = ["Name", *selected_gene_ids]
    matrix = {}

    for chunk in pd.read_csv(matrix_path, usecols=usecols, index_col=0, chunksize=256):
        chunk.index = chunk.index.astype(str).str.strip()
        chunk.columns = chunk.columns.astype(str).str.strip()

        for row_gene, row in chunk.iterrows():
            if row_gene not in selected_gene_ids:
                continue

            row_values = {}
            for col_gene, value in row.items():
                if pd.isna(value) or col_gene not in selected_gene_ids:
                    continue
                row_values[col_gene] = float(value)

            matrix[row_gene] = row_values

    return matrix


def compute_average_similarity(selected_candidates, similarity_matrix):
    pair_values = []
    missing_pairs = []
    ids = []
    labels = []

    for candidate in selected_candidates:
        ids.append(candidate["matrix_gene_id"])
        labels.append(candidate["candidate_label"])

    for i in range(len(selected_candidates)):
        for j in range(i + 1, len(selected_candidates)):
            gene_i = ids[i]
            gene_j = ids[j]

            if not gene_i or not gene_j:
                missing_pairs.append(f"{labels[i]}::{labels[j]}")
                continue

            value = similarity_matrix.get(gene_i, {}).get(gene_j)
            if value is None:
                value = similarity_matrix.get(gene_j, {}).get(gene_i)
            if value is None:
                missing_pairs.append(f"{gene_i}::{gene_j}")
                continue

            pair_values.append(value)

    if not pair_values:
        return {
            "avg_similarity": None,
            "avg_similarity_display": "cannot find similarity",
            "n_pairs_used": 0,
            "n_pairs_missing": len(missing_pairs),
            "missing_pairs_example": " | ".join(missing_pairs[:5]),
        }

    avg_similarity = sum(pair_values) / len(pair_values)
    return {
        "avg_similarity": avg_similarity,
        "avg_similarity_display": format_similarity(avg_similarity),
        "n_pairs_used": len(pair_values),
        "n_pairs_missing": len(missing_pairs),
        "missing_pairs_example": " | ".join(missing_pairs[:5]),
    }


def is_better(candidate_score, current_score):
    if candidate_score is None:
        return False
    if current_score is None:
        return True
    return candidate_score > current_score


def create_similarity_plot(round_rows, baseline_value, output_path: Path, disease_id: str):
    if not round_rows:
        raise ValueError("round_rows is empty")

    kept_rounds = []
    kept_values = []
    proposal_rounds = []
    proposal_values = []
    hit_round = None
    stop_round = None

    for row in round_rows:
        round_number = int(row["round"])
        kept_value = row.get("kept_avg_similarity")
        if kept_value is not None and kept_value != "":
            kept_rounds.append(round_number)
            kept_values.append(float(kept_value))

        proposal_value = row.get("proposal_avg_similarity")
        if proposal_value is not None and proposal_value != "":
            proposal_rounds.append(round_number)
            proposal_values.append(float(proposal_value))

        baseline_hit_value = row.get("baseline_hit_round")
        if baseline_hit_value not in (None, "") and hit_round is None:
            hit_round = int(baseline_hit_value)

        planned_stop_value = row.get("planned_stop_round")
        if planned_stop_value not in (None, ""):
            stop_round = int(planned_stop_value)

    y_values = [baseline_value, *kept_values, *proposal_values]
    y_min = min(y_values)
    y_max = max(y_values)
    if math.isclose(y_min, y_max):
        y_min -= 0.05
        y_max += 0.05
    else:
        pad = (y_max - y_min) * 0.12
        y_min -= pad
        y_max += pad

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if proposal_rounds:
        ax.plot(
            proposal_rounds,
            proposal_values,
            color="#94a3b8",
            linewidth=1.6,
            alpha=0.95,
            label="Proposal score",
        )

    if kept_rounds:
        ax.plot(
            kept_rounds,
            kept_values,
            color="#1e3a8a",
            linewidth=2.7,
            label="Kept template score",
        )

    ax.axhline(
        baseline_value,
        color="#b91c1c",
        linestyle="--",
        linewidth=2.0,
        label="Causal gene score",
    )

    if hit_round is not None:
        ax.axvline(
            hit_round,
            color="#475569",
            linestyle=":",
            linewidth=1.4,
            label="First baseline hit",
        )

    if stop_round is not None and stop_round != hit_round:
        ax.axvline(
            stop_round,
            color="#0f172a",
            linestyle="-.",
            linewidth=1.2,
            label="Stop round",
        )

    ax.set_xlim(1, max(int(row["round"]) for row in round_rows))
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{disease_id} pairwise score across iterations", fontsize=18, pad=12)
    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("Average pairwise score", fontsize=14)
    ax.grid(True, axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def run_ga(
    pockets,
    similarity_matrix,
    n_rounds,
    output_path,
    baseline,
    disease_id,
    max_genes_to_change,
):
    print("PIPELINE STARTED")

    changeable_pocket_ids = [
        pocket_id
        for pocket_id, pocket in pockets.items()
        if len(pocket["candidates"]) > 1
    ]

    if not pockets:
        raise ValueError("No pockets available")
    if not changeable_pocket_ids:
        raise ValueError("No pockets have alternative genes to test")

    best_selection = choose_initial_random_selection(pockets)
    best_metrics = compute_average_similarity(
        list(best_selection.values()),
        similarity_matrix,
    )
    best_round = 1
    baseline_hit_round = None
    planned_stop_round = None

    if best_metrics["avg_similarity"] is not None and best_metrics["avg_similarity"] >= baseline:
        baseline_hit_round = 1
        planned_stop_round = min(n_rounds, 2)

    round_rows = [
        {
            "round": 1,
            "round_kind": "initial_random_template",
            "n_changed_genes": 0,
            "pocket_seed_targetId": "",
            "previous_gene": "",
            "proposed_gene": "",
            "proposed_gene_matrix_id": "",
            "proposed_gene_type": "",
            "proposed_neighbor_direction": "",
            "proposed_neighbor_rank": "",
            "proposal_avg_similarity": best_metrics["avg_similarity"],
            "proposal_avg_similarity_display": best_metrics["avg_similarity_display"],
            "proposal_n_pairs_used": best_metrics["n_pairs_used"],
            "proposal_n_pairs_missing": best_metrics["n_pairs_missing"],
            "proposal_missing_pairs_example": best_metrics["missing_pairs_example"],
            "accepted": "initial_random",
            "kept_gene_after_round": "",
            "kept_avg_similarity": best_metrics["avg_similarity"],
            "kept_avg_similarity_display": best_metrics["avg_similarity_display"],
            "kept_n_pairs_used": best_metrics["n_pairs_used"],
            "kept_n_pairs_missing": best_metrics["n_pairs_missing"],
            "kept_missing_pairs_example": best_metrics["missing_pairs_example"],
            "template_round_after_round": best_round,
            "baseline_hit_round": baseline_hit_round or "",
            "planned_stop_round": planned_stop_round or "",
            "causal_baseline_similarity": baseline,
        }
    ]

    for round_number in range(2, n_rounds + 1):
        change_count = random.randint(1, min(max_genes_to_change, len(changeable_pocket_ids)))
        chosen_pocket_ids = random.sample(changeable_pocket_ids, change_count)
        proposal_selection = dict(best_selection)

        previous_candidates = []
        proposed_candidates = []

        for pocket_id in chosen_pocket_ids:
            current_candidate = best_selection[pocket_id]
            alternatives = [
                candidate
                for candidate in pockets[pocket_id]["candidates"]
                if candidate != current_candidate
            ]
            if not alternatives:
                continue

            proposed_candidate = random.choice(alternatives)
            proposal_selection[pocket_id] = proposed_candidate
            previous_candidates.append(current_candidate)
            proposed_candidates.append(proposed_candidate)

        if not proposed_candidates:
            continue

        proposal_metrics = compute_average_similarity(
            list(proposal_selection.values()),
            similarity_matrix,
        )

        proposal_similarity = proposal_metrics["avg_similarity"]
        accepted = is_better(proposal_similarity, best_metrics["avg_similarity"])

        if accepted:
            best_selection = proposal_selection
            best_metrics = proposal_metrics
            best_round = round_number

            if baseline_hit_round is None and proposal_similarity is not None and proposal_similarity >= baseline:
                baseline_hit_round = round_number
                planned_stop_round = min(n_rounds, baseline_hit_round * 2)
                print(f"Baseline matched at round {baseline_hit_round}; planned stop at round {planned_stop_round}")

        kept_candidates = proposed_candidates if accepted else previous_candidates

        round_rows.append(
            {
                "round": round_number,
                "round_kind": "proposal",
                "n_changed_genes": len(proposed_candidates),
                "pocket_seed_targetId": "|".join(chosen_pocket_ids),
                "previous_gene": serialize_candidates(previous_candidates, "candidate_label"),
                "proposed_gene": serialize_candidates(proposed_candidates, "candidate_label"),
                "proposed_gene_matrix_id": serialize_candidates(proposed_candidates, "matrix_gene_id"),
                "proposed_gene_type": serialize_candidates(proposed_candidates, "candidate_type"),
                "proposed_neighbor_direction": serialize_candidates(proposed_candidates, "neighbor_direction"),
                "proposed_neighbor_rank": serialize_candidates(proposed_candidates, "neighbor_rank"),
                "proposal_avg_similarity": proposal_similarity,
                "proposal_avg_similarity_display": proposal_metrics["avg_similarity_display"],
                "proposal_n_pairs_used": proposal_metrics["n_pairs_used"],
                "proposal_n_pairs_missing": proposal_metrics["n_pairs_missing"],
                "proposal_missing_pairs_example": proposal_metrics["missing_pairs_example"],
                "accepted": "yes" if accepted else "no",
                "kept_gene_after_round": serialize_candidates(kept_candidates, "candidate_label"),
                "kept_avg_similarity": best_metrics["avg_similarity"],
                "kept_avg_similarity_display": best_metrics["avg_similarity_display"],
                "kept_n_pairs_used": best_metrics["n_pairs_used"],
                "kept_n_pairs_missing": best_metrics["n_pairs_missing"],
                "kept_missing_pairs_example": best_metrics["missing_pairs_example"],
                "template_round_after_round": best_round,
                "baseline_hit_round": baseline_hit_round or "",
                "planned_stop_round": planned_stop_round or "",
                "causal_baseline_similarity": baseline,
            }
        )

        if planned_stop_round is not None and round_number >= planned_stop_round:
            print(f"Early stop at round {round_number}")
            break

        if round_number % 10 == 0:
            print(
                "Round "
                f"{round_number} | Proposal: {format_similarity(proposal_similarity)} | "
                f"Kept: {best_metrics['avg_similarity_display']}"
            )

    print("PIPELINE DONE")
    print("Total rounds recorded:", len(round_rows))

    create_similarity_plot(
        round_rows,
        baseline,
        output_path,
        disease_id,
    )

    return round_rows


def write_round_rows(round_rows, output_path: Path):
    fieldnames = [
        "round",
        "round_kind",
        "n_changed_genes",
        "pocket_seed_targetId",
        "previous_gene",
        "proposed_gene",
        "proposed_gene_matrix_id",
        "proposed_gene_type",
        "proposed_neighbor_direction",
        "proposed_neighbor_rank",
        "proposal_avg_similarity",
        "proposal_avg_similarity_display",
        "proposal_n_pairs_used",
        "proposal_n_pairs_missing",
        "proposal_missing_pairs_example",
        "accepted",
        "kept_gene_after_round",
        "kept_avg_similarity",
        "kept_avg_similarity_display",
        "kept_n_pairs_used",
        "kept_n_pairs_missing",
        "kept_missing_pairs_example",
        "template_round_after_round",
        "baseline_hit_round",
        "planned_stop_round",
        "causal_baseline_similarity",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(round_rows)


def load_round_plot_data(round_csv_path: Path):
    rows = read_rows(round_csv_path)
    if not rows:
        raise ValueError("round_rows csv is empty")

    kept_rounds = []
    kept_values = []
    proposal_rounds = []
    proposal_values = []
    baseline_value = None

    for row in rows:
        round_number = int(row["round"])

        if baseline_value is None and row.get("causal_baseline_similarity"):
            baseline_value = float(row["causal_baseline_similarity"])

        kept_value = (row.get("kept_avg_similarity") or "").strip()
        if kept_value:
            kept_rounds.append(round_number)
            kept_values.append(float(kept_value))

        proposal_value = (row.get("proposal_avg_similarity") or "").strip()
        if proposal_value:
            proposal_rounds.append(round_number)
            proposal_values.append(float(proposal_value))

    if baseline_value is None:
        raise ValueError("causal baseline similarity is missing from the csv")

    return SimpleNamespace(
        rows=rows,
        baseline_value=baseline_value,
        kept_rounds=kept_rounds,
        kept_values=kept_values,
        proposal_rounds=proposal_rounds,
        proposal_values=proposal_values,
    )


def create_similarity_plot_simple(round_csv_path: Path, output_path: Path, disease_id: str):
    data = load_round_plot_data(round_csv_path)
    y_values = [data.baseline_value, *data.kept_values, *data.proposal_values]
    y_min = min(y_values)
    y_max = max(y_values)

    if math.isclose(y_min, y_max):
        y_min -= 0.05
        y_max += 0.05
    else:
        pad = (y_max - y_min) * 0.12
        y_min -= pad
        y_max += pad

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        data.proposal_rounds,
        data.proposal_values,
        color="#94a3b8",
        linewidth=1.4,
        alpha=0.95,
        label="Proposal score",
    )
    ax.plot(
        data.kept_rounds,
        data.kept_values,
        color="#1e3a8a",
        linewidth=3.0,
        label="Kept template score",
    )
    ax.axhline(
        data.baseline_value,
        color="#b91c1c",
        linestyle="--",
        linewidth=2.0,
        label="Causal gene score",
    )

    ax.set_xlim(1, max(data.kept_rounds))
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{disease_id} score across iterations: simple comparison view", fontsize=18, pad=12)
    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("Average pairwise score", fontsize=14)
    ax.grid(True, axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Simple comparison graph saved to: {output_path}")


def create_similarity_plot_sketch(round_csv_path: Path, output_path: Path, disease_id: str):
    data = load_round_plot_data(round_csv_path)
    y_values = [data.baseline_value, *data.kept_values, *data.proposal_values]
    y_min = min(y_values)
    y_max = max(y_values)

    if math.isclose(y_min, y_max):
        y_min -= 0.05
        y_max += 0.05
    else:
        pad = (y_max - y_min) * 0.15
        y_min -= pad
        y_max += pad

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    proposal_line = ax.plot(
        data.proposal_rounds,
        data.proposal_values,
        color="#8fa1b8",
        linewidth=1.8,
        alpha=0.75,
    )[0]
    kept_line = ax.plot(
        data.kept_rounds,
        data.kept_values,
        color="#243c8f",
        linewidth=3.2,
    )[0]
    baseline_line = ax.axhline(
        data.baseline_value,
        color="#b22222",
        linestyle="--",
        linewidth=2.2,
    )

    proposal_line.set_sketch_params(scale=1, length=90, randomness=2)
    kept_line.set_sketch_params(scale=1, length=110, randomness=2)
    baseline_line.set_sketch_params(scale=1, length=80, randomness=2)

    for spine in ax.spines.values():
        spine.set_sketch_params(scale=1, length=100, randomness=2)

    ax.set_xlim(1, max(data.kept_rounds))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("Pairwise score", fontsize=14)
    ax.set_title(f"{disease_id} sketch-style optimisation view", fontsize=18, pad=16)

    ax.annotate(
        "Kept template",
        xy=(data.kept_rounds[-1], data.kept_values[-1]),
        xytext=(max(data.kept_rounds) * 0.73, data.kept_values[-1] + 0.07),
        fontsize=12,
        color="#243c8f",
        arrowprops=dict(arrowstyle="->", color="#243c8f", lw=1.5),
    )
    ax.annotate(
        "Causal score",
        xy=(max(data.kept_rounds) * 0.6, data.baseline_value),
        xytext=(max(data.kept_rounds) * 0.73, data.baseline_value - 0.08),
        fontsize=12,
        color="#b22222",
        arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.4),
    )
    ax.annotate(
        "Proposal path",
        xy=(
            data.proposal_rounds[int(len(data.proposal_rounds) * 0.55)],
            data.proposal_values[int(len(data.proposal_values) * 0.55)],
        ),
        xytext=(max(data.kept_rounds) * 0.2, y_min + (y_max - y_min) * 0.18),
        fontsize=12,
        color="#62748a",
        arrowprops=dict(arrowstyle="->", color="#62748a", lw=1.2),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Sketch-style graph saved to: {output_path}")


def create_similarity_plot_kept_only(round_csv_path: Path, output_path: Path, disease_id: str):
    data = load_round_plot_data(round_csv_path)
    y_values = [data.baseline_value, *data.kept_values]
    y_min = min(y_values)
    y_max = max(y_values)

    if math.isclose(y_min, y_max):
        y_min -= 0.05
        y_max += 0.05
    else:
        pad = (y_max - y_min) * 0.12
        y_min -= pad
        y_max += pad

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        data.kept_rounds,
        data.kept_values,
        color="#1e3a8a",
        linewidth=3.1,
        label="Kept template score",
    )
    ax.axhline(
        data.baseline_value,
        color="#b91c1c",
        linestyle="--",
        linewidth=2.0,
        label="Causal gene score",
    )

    ax.set_xlim(1, max(data.kept_rounds))
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{disease_id} kept-template score only", fontsize=18, pad=12)
    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("Average pairwise score", fontsize=14)
    ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Kept-only graph saved to: {output_path}")


def create_optional_plots(config):
    disease_id = getattr(config, "effective_disease_id", config.disease_id)
    round_csv_path = resolve_input_file(config.round_output_csv, "Rounds CSV")

    create_similarity_plot_simple(round_csv_path, config.graph_output_png_simple, disease_id)
    create_similarity_plot_sketch(round_csv_path, config.graph_output_png_sketch, disease_id)
    create_similarity_plot_kept_only(round_csv_path, config.graph_output_png_kept_only, disease_id)

    print(f"Original graph: {config.graph_output_png}")
    print(f"Simple graph: {config.graph_output_png_simple}")
    print(f"Sketch graph: {config.graph_output_png_sketch}")
    print(f"Kept-only graph: {config.graph_output_png_kept_only}")


def run_from_config(config):
    causal_csv_path = resolve_input_file(config.causal_csv, "Causal CSV")
    summary_csv_path = resolve_input_file(config.summary_csv, "Summary CSV")
    matrix_csv_path = resolve_input_file(config.matrix_path, "Matrix CSV", allow_checkpoint=True)

    if not config.neighbor_csv.exists():
        print("Neighbor CSV not found. Generating it now...")
        generate_neighbor_csv(config)

    neighbor_csv_path = resolve_input_file(config.neighbor_csv, "Neighbor CSV")
    causal_rows = read_rows(causal_csv_path)
    neighbor_rows = read_rows(neighbor_csv_path)
    matrix_gene_ids = load_matrix_gene_ids(config.source_matrix)
    effective_disease_id = infer_disease_id(causal_rows, config.disease_id, causal_csv_path)
    config.effective_disease_id = effective_disease_id
    baseline_similarity = load_baseline_similarity(summary_csv_path, effective_disease_id)
    pockets = build_pockets(causal_rows, neighbor_rows, matrix_gene_ids=matrix_gene_ids)
    write_pocket_table(pockets, config.pocket_output_csv)

    required_gene_ids = collect_matrix_gene_ids(pockets)
    similarity_matrix = load_similarity_subset(matrix_csv_path, required_gene_ids)

    random.seed(config.random_seed)
    round_rows = run_ga(
        pockets,
        similarity_matrix,
        config.n_rounds,
        config.graph_output_png,
        baseline_similarity,
        effective_disease_id,
        config.max_genes_to_change,
    )
    write_round_rows(round_rows, config.round_output_csv)

    print(f"Pocket table saved to: {config.pocket_output_csv}")
    print(f"Rounds CSV saved to: {config.round_output_csv}")
    print(f"Graph saved to: {config.graph_output_png}")

    return round_rows
