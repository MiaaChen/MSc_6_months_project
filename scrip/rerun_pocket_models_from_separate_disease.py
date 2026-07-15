#!/usr/bin/env python3
"""Rerun pocket models from per-disease `separate_disease` folders.

This script uses the existing pocket-model inputs already saved under:

    separate_disease/<n>_gene_related_disease/<disease_id>/

For each disease it reads:

1. `<disease_id>.csv`
2. `<disease_id>_pocket_candidates.csv`
3. `<disease_id>_gene_gene_correlation_TPM_updated_subset.csv`
4. `<disease_id>_pocket_similarity_rounds.csv` (optional, for comparison)

It then reruns the pocket search with one or more random restarts and writes:

1. One summary row per disease
2. A ranked cross-disease swap table
3. Optional rerun round histories
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(10**9)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEPARATE_DISEASE_ROOT = PROJECT_ROOT / "data" / "separate_disease"
RANDOM_SUMMARY_DEFAULT = PROJECT_ROOT / "data" / "Similariy" / "random_similarity" / "TPM_disease_vs_random_summary.csv"
RANDOM_DISTRIBUTION_DEFAULT = PROJECT_ROOT / "data" / "Similariy" / "random_similarity" / "TPM_random_similarity_distribution.csv"
GLOBAL_SUMMARY_DEFAULT = PROJECT_ROOT / "data" / "Similariy" / "TPM_disease_avg_similarity_summary.csv"


@dataclass(frozen=True)
class Candidate:
    seed_target_id: str
    candidate_type: str
    candidate_label: str
    matrix_gene_id: str
    neighbor_direction: str
    neighbor_rank: int
    neighbor_symbol: str
    neighbor_ncbi_gene_id: str


@dataclass
class DiseaseInput:
    folder_name: str
    disease_id: str
    disease_dir: Path
    causal_csv: Path
    pocket_candidates_csv: Path
    matrix_subset_csv: Path
    saved_rounds_csv: Optional[Path]
    seed_target_ids: List[str]
    pockets: Dict[str, List[Candidate]]


@dataclass
class SearchResult:
    restart_index: int
    initial_selection: Dict[str, Candidate]
    best_selection: Dict[str, Candidate]
    initial_score: Optional[float]
    best_score: Optional[float]
    initial_metrics: dict
    best_metrics: dict
    rounds_run: int
    accepted_rounds: int
    baseline_hit_round: Optional[int]
    planned_stop_round: Optional[int]
    stop_reason: str
    history_rows: Optional[List[dict]]


@dataclass
class RerunCollection:
    best_result: SearchResult
    restart_results: List[SearchResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun per-disease pocket models from the separate_disease folder "
            "structure and rank the resulting swaps."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help=f"Project root (default: {PROJECT_ROOT})",
    )
    parser.add_argument(
        "--folder-name",
        action="append",
        dest="folder_names",
        default=[],
        help="Limit to one or more *_gene_related_disease folders. Repeat the flag to pass multiple values.",
    )
    parser.add_argument(
        "--disease-id",
        action="append",
        dest="disease_ids",
        default=[],
        help="Limit to one or more disease IDs. Repeat the flag to pass multiple values.",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=5000,
        help="Rounds per stochastic rerun (default: 5000)",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=10,
        help="Number of rerun restarts per disease (default: 10)",
    )
    parser.add_argument(
        "--max-genes-to-change",
        type=int,
        default=9,
        help="Maximum number of pockets to perturb in one proposal round (default: 9)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.8,
        help=(
            "Fraction of restart endpoints that must match the best final set "
            "to mark a disease as stable across reruns (default: 0.8)"
        ),
    )
    parser.add_argument(
        "--require-unique-genes",
        action="store_true",
        help="Reject proposals that reuse the same matrix gene in more than one pocket.",
    )
    parser.add_argument(
        "--write-round-history",
        action="store_true",
        help="Write the best rerun round history for each disease.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to pocket_model_reruns/<folder_name> when one "
            "source folder is selected, otherwise pocket_model_reruns under the project root."
        ),
    )
    parser.add_argument(
        "--random-summary-file",
        type=Path,
        default=RANDOM_SUMMARY_DEFAULT,
        help=f"Random-summary CSV (default: {RANDOM_SUMMARY_DEFAULT})",
    )
    parser.add_argument(
        "--random-distribution-file",
        type=Path,
        default=RANDOM_DISTRIBUTION_DEFAULT,
        help=f"Random-distribution CSV (default: {RANDOM_DISTRIBUTION_DEFAULT})",
    )
    parser.add_argument(
        "--global-summary-file",
        type=Path,
        default=GLOBAL_SUMMARY_DEFAULT,
        help=f"TPM disease summary CSV (default: {GLOBAL_SUMMARY_DEFAULT})",
    )
    return parser.parse_args()


def default_output_dir(project_root: Path, folder_names: Sequence[str]) -> Path:
    output_root = project_root / "data" / "pocket_model_reruns"
    unique_folder_names = []
    for value in folder_names:
        folder_name = value.strip()
        if folder_name and folder_name not in unique_folder_names:
            unique_folder_names.append(folder_name)
    if len(unique_folder_names) == 1:
        return output_root / unique_folder_names[0]
    return output_root


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def read_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_seed_target_ids(causal_csv: Path) -> List[str]:
    seed_target_ids = []
    with causal_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target_id = (row.get("targetId") or "").strip()
            if target_id:
                seed_target_ids.append(target_id)
    return ordered_unique(seed_target_ids)


def discover_disease_inputs(
    project_root: Path,
    wanted_folder_names: Sequence[str],
    wanted_disease_ids: Sequence[str],
) -> List[DiseaseInput]:
    root = project_root / "data" / "separate_disease"
    if not root.exists():
        raise FileNotFoundError(f"separate_disease root not found: {root}")

    wanted_folders = {value.strip() for value in wanted_folder_names if value.strip()}
    wanted_diseases = {value.strip() for value in wanted_disease_ids if value.strip()}
    results = []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.endswith("_gene_related_disease"):
            continue
        if wanted_folders and folder.name not in wanted_folders:
            continue

        for disease_dir in sorted(folder.iterdir()):
            if not disease_dir.is_dir():
                continue
            disease_id = disease_dir.name
            if wanted_diseases and disease_id not in wanted_diseases:
                continue

            causal_csv = disease_dir / f"{disease_id}.csv"
            pocket_csv = disease_dir / f"{disease_id}_pocket_candidates.csv"
            matrix_csv = disease_dir / f"{disease_id}_gene_gene_correlation_TPM_updated_subset.csv"
            rounds_csv = disease_dir / f"{disease_id}_pocket_similarity_rounds.csv"

            if not causal_csv.exists() or not pocket_csv.exists() or not matrix_csv.exists():
                continue

            seed_target_ids = load_seed_target_ids(causal_csv)
            pockets = load_pockets_from_csv(pocket_csv, seed_target_ids)

            results.append(
                DiseaseInput(
                    folder_name=folder.name,
                    disease_id=disease_id,
                    disease_dir=disease_dir,
                    causal_csv=causal_csv,
                    pocket_candidates_csv=pocket_csv,
                    matrix_subset_csv=matrix_csv,
                    saved_rounds_csv=rounds_csv if rounds_csv.exists() else None,
                    seed_target_ids=seed_target_ids,
                    pockets=pockets,
                )
            )

    if wanted_folders:
        found_folders = {item.folder_name for item in results}
        missing_folders = sorted(wanted_folders - found_folders)
        if missing_folders:
            raise FileNotFoundError("Folder(s) not found or missing required inputs: " + ", ".join(missing_folders))

    if wanted_diseases:
        found_diseases = {item.disease_id for item in results}
        missing_diseases = sorted(wanted_diseases - found_diseases)
        if missing_diseases:
            raise FileNotFoundError("Disease(s) not found or missing required inputs: " + ", ".join(missing_diseases))

    return results


def load_pockets_from_csv(pocket_csv: Path, seed_target_ids: Sequence[str]) -> Dict[str, List[Candidate]]:
    pockets: Dict[str, List[Candidate]] = {seed_target_id: [] for seed_target_id in seed_target_ids}
    rows = read_csv_rows(pocket_csv)
    for row in rows:
        seed_target_id = (row.get("seed_targetId") or "").strip()
        if seed_target_id not in pockets:
            continue
        pockets[seed_target_id].append(
            Candidate(
                seed_target_id=seed_target_id,
                candidate_type=(row.get("candidate_type") or "").strip(),
                candidate_label=(row.get("candidate_label") or "").strip(),
                matrix_gene_id=(row.get("matrix_gene_id") or "").strip(),
                neighbor_direction=(row.get("neighbor_direction") or "").strip(),
                neighbor_rank=int((row.get("neighbor_rank") or "0").strip() or 0),
                neighbor_symbol=(row.get("neighbor_symbol") or "").strip(),
                neighbor_ncbi_gene_id=(row.get("neighbor_ncbi_gene_id") or "").strip(),
            )
        )

    for seed_target_id in seed_target_ids:
        if not pockets[seed_target_id]:
            raise ValueError(f"No pocket candidates found for {seed_target_id} in {pocket_csv}")

    return pockets


def load_subset_matrix(matrix_csv: Path) -> Dict[str, Dict[str, float]]:
    with matrix_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        column_gene_ids = [value.strip() for value in header[1:]]

        matrix: Dict[str, Dict[str, float]] = {}
        for row in reader:
            if not row:
                continue
            row_gene = row[0].strip()
            values = {}
            for column_gene_id, value_text in zip(column_gene_ids, row[1:]):
                value_text = value_text.strip()
                if not value_text:
                    continue
                try:
                    values[column_gene_id] = float(value_text)
                except ValueError:
                    continue
            matrix[row_gene] = values
    return matrix


def compute_average_similarity(selected_candidates: Sequence[Candidate], similarity_matrix: Dict[str, Dict[str, float]]) -> dict:
    pair_values = []
    missing_pairs = []
    for left_candidate in range(len(selected_candidates)):
        for right_candidate in range(left_candidate + 1, len(selected_candidates)):
            left_gene = selected_candidates[left_candidate].matrix_gene_id
            right_gene = selected_candidates[right_candidate].matrix_gene_id
            value = similarity_matrix.get(left_gene, {}).get(right_gene)
            if value is None:
                value = similarity_matrix.get(right_gene, {}).get(left_gene)
            if value is None:
                missing_pairs.append(f"{left_gene}::{right_gene}")
                continue
            pair_values.append(value)

    if not pair_values:
        avg_similarity = None
    else:
        avg_similarity = sum(pair_values) / len(pair_values)

    return {
        "avg_similarity": avg_similarity,
        "n_pairs_used": len(pair_values),
        "n_pairs_missing": len(missing_pairs),
        "missing_pairs_example": " | ".join(missing_pairs[:5]),
    }


def selection_has_unique_genes(selection: Dict[str, Candidate]) -> bool:
    gene_ids = [candidate.matrix_gene_id for candidate in selection.values() if candidate.matrix_gene_id]
    return len(gene_ids) == len(set(gene_ids))


def choose_initial_random_selection(
    seed_target_ids: Sequence[str],
    pockets: Dict[str, List[Candidate]],
    rng: random.Random,
    require_unique_genes: bool,
    max_attempts: int = 500,
) -> Dict[str, Candidate]:
    for _ in range(max_attempts):
        selection = {
            seed_target_id: rng.choice(pockets[seed_target_id])
            for seed_target_id in seed_target_ids
        }
        if not require_unique_genes or selection_has_unique_genes(selection):
            return selection

    selection = {}
    used_gene_ids = set()
    for seed_target_id in seed_target_ids:
        selected = None
        for candidate in pockets[seed_target_id]:
            if not require_unique_genes or candidate.matrix_gene_id not in used_gene_ids:
                selected = candidate
                break
        if selected is None:
            selected = pockets[seed_target_id][0]
        selection[seed_target_id] = selected
        used_gene_ids.add(selected.matrix_gene_id)
    return selection


def propose_selection(
    seed_target_ids: Sequence[str],
    pockets: Dict[str, List[Candidate]],
    current_selection: Dict[str, Candidate],
    chosen_seed_target_ids: Sequence[str],
    rng: random.Random,
    require_unique_genes: bool,
    max_attempts: int = 200,
) -> tuple[Optional[Dict[str, Candidate]], List[Candidate], List[Candidate]]:
    for _ in range(max_attempts):
        proposal = dict(current_selection)
        previous_candidates = []
        proposed_candidates = []

        for seed_target_id in chosen_seed_target_ids:
            current_candidate = current_selection[seed_target_id]
            alternatives = [
                candidate
                for candidate in pockets[seed_target_id]
                if candidate.matrix_gene_id != current_candidate.matrix_gene_id
            ]
            if not alternatives:
                continue
            chosen_candidate = rng.choice(alternatives)
            proposal[seed_target_id] = chosen_candidate
            previous_candidates.append(current_candidate)
            proposed_candidates.append(chosen_candidate)

        if not proposed_candidates:
            continue
        if require_unique_genes and not selection_has_unique_genes(proposal):
            continue
        return proposal, previous_candidates, proposed_candidates

    return None, [], []


def metrics_are_better(candidate_metrics: dict, current_metrics: dict) -> bool:
    candidate_pairs = int(candidate_metrics.get("n_pairs_used") or 0)
    current_pairs = int(current_metrics.get("n_pairs_used") or 0)
    if candidate_pairs != current_pairs:
        return candidate_pairs > current_pairs

    candidate_score = candidate_metrics.get("avg_similarity")
    current_score = current_metrics.get("avg_similarity")
    if candidate_score is None:
        return False
    if current_score is None:
        return True
    if candidate_score != current_score:
        return candidate_score > current_score

    candidate_missing = int(candidate_metrics.get("n_pairs_missing") or 0)
    current_missing = int(current_metrics.get("n_pairs_missing") or 0)
    return candidate_missing < current_missing


def serialize_candidates(candidates: Sequence[Candidate], field_name: str) -> str:
    values = []
    for candidate in candidates:
        values.append(str(getattr(candidate, field_name)))
    return "|".join(values)


def run_single_search(
    disease_input: DiseaseInput,
    similarity_matrix: Dict[str, Dict[str, float]],
    baseline_score: Optional[float],
    n_rounds: int,
    max_genes_to_change: int,
    rng_seed: int,
    require_unique_genes: bool,
    store_history: bool,
) -> SearchResult:
    rng = random.Random(rng_seed)
    changeable_seed_target_ids = [
        seed_target_id
        for seed_target_id in disease_input.seed_target_ids
        if len(disease_input.pockets[seed_target_id]) > 1
    ]

    initial_selection = choose_initial_random_selection(
        seed_target_ids=disease_input.seed_target_ids,
        pockets=disease_input.pockets,
        rng=rng,
        require_unique_genes=require_unique_genes,
    )
    best_selection = dict(initial_selection)
    initial_metrics = compute_average_similarity(list(initial_selection.values()), similarity_matrix)
    best_metrics = dict(initial_metrics)
    initial_score = initial_metrics["avg_similarity"]
    best_score = initial_score
    accepted_rounds = 0
    baseline_hit_round = None
    planned_stop_round = None
    stop_reason = "max_rounds_reached"
    history_rows = [] if store_history else None

    if baseline_score is not None and best_score is not None and best_score >= baseline_score:
        baseline_hit_round = 1
        planned_stop_round = min(n_rounds, 2)

    if history_rows is not None:
        history_rows.append(
            {
                "restart_index": 0,
                "round": 1,
                "round_kind": "initial_random_template",
                "n_changed_genes": 0,
                "pocket_seed_target_ids": "",
                "previous_gene": "",
                "proposed_gene": "",
                "accepted": "initial_random",
                "kept_score": best_score,
                "proposal_score": best_score,
                "kept_n_pairs_used": best_metrics["n_pairs_used"],
                "kept_n_pairs_missing": best_metrics["n_pairs_missing"],
                "kept_missing_pairs_example": best_metrics["missing_pairs_example"],
                "baseline_hit_round": baseline_hit_round or "",
                "planned_stop_round": planned_stop_round or "",
            }
        )

    if not changeable_seed_target_ids:
        return SearchResult(
            restart_index=0,
            initial_selection=initial_selection,
            best_selection=best_selection,
            initial_score=initial_score,
            best_score=best_score,
            initial_metrics=initial_metrics,
            best_metrics=best_metrics,
            rounds_run=1,
            accepted_rounds=0,
            baseline_hit_round=baseline_hit_round,
            planned_stop_round=planned_stop_round,
            stop_reason="no_changeable_pockets",
            history_rows=history_rows,
        )

    last_round = 1
    for round_number in range(2, n_rounds + 1):
        last_round = round_number
        n_to_change = rng.randint(1, min(max_genes_to_change, len(changeable_seed_target_ids)))
        chosen_seed_target_ids = rng.sample(changeable_seed_target_ids, n_to_change)

        proposal_selection, previous_candidates, proposed_candidates = propose_selection(
            seed_target_ids=disease_input.seed_target_ids,
            pockets=disease_input.pockets,
            current_selection=best_selection,
            chosen_seed_target_ids=chosen_seed_target_ids,
            rng=rng,
            require_unique_genes=require_unique_genes,
        )
        if proposal_selection is None:
            continue

        proposal_metrics = compute_average_similarity(list(proposal_selection.values()), similarity_matrix)
        proposal_score = proposal_metrics["avg_similarity"]
        accepted = metrics_are_better(proposal_metrics, best_metrics)

        if accepted:
            best_selection = proposal_selection
            best_metrics = proposal_metrics
            best_score = proposal_score
            accepted_rounds += 1
            if (
                baseline_hit_round is None
                and baseline_score is not None
                and best_score is not None
                and best_score >= baseline_score
            ):
                baseline_hit_round = round_number
                planned_stop_round = min(n_rounds, baseline_hit_round * 2)

        if history_rows is not None:
            kept_candidates = proposed_candidates if accepted else previous_candidates
            history_rows.append(
                {
                    "restart_index": 0,
                    "round": round_number,
                    "round_kind": "proposal",
                    "n_changed_genes": len(proposed_candidates),
                    "pocket_seed_target_ids": "|".join(chosen_seed_target_ids),
                    "previous_gene": serialize_candidates(previous_candidates, "candidate_label"),
                    "proposed_gene": serialize_candidates(proposed_candidates, "candidate_label"),
                    "accepted": "yes" if accepted else "no",
                    "kept_gene_after_round": serialize_candidates(kept_candidates, "candidate_label"),
                    "kept_score": best_score,
                    "proposal_score": proposal_score,
                    "kept_n_pairs_used": best_metrics["n_pairs_used"],
                    "kept_n_pairs_missing": best_metrics["n_pairs_missing"],
                    "kept_missing_pairs_example": best_metrics["missing_pairs_example"],
                    "baseline_hit_round": baseline_hit_round or "",
                    "planned_stop_round": planned_stop_round or "",
                }
            )

        if planned_stop_round is not None and round_number >= planned_stop_round:
            stop_reason = "planned_stop_after_baseline_hit"
            break

    return SearchResult(
        restart_index=0,
        initial_selection=initial_selection,
        best_selection=best_selection,
        initial_score=initial_score,
        best_score=best_score,
        initial_metrics=initial_metrics,
        best_metrics=best_metrics,
        rounds_run=last_round,
        accepted_rounds=accepted_rounds,
        baseline_hit_round=baseline_hit_round,
        planned_stop_round=planned_stop_round,
        stop_reason=stop_reason,
        history_rows=history_rows,
    )


def pick_better_result(current_best: Optional[SearchResult], candidate: SearchResult) -> SearchResult:
    if current_best is None:
        return candidate
    if metrics_are_better(candidate.best_metrics, current_best.best_metrics):
        return candidate
    return current_best


def rerun_with_restarts(
    disease_input: DiseaseInput,
    similarity_matrix: Dict[str, Dict[str, float]],
    baseline_score: Optional[float],
    n_rounds: int,
    n_restarts: int,
    max_genes_to_change: int,
    base_seed: int,
    require_unique_genes: bool,
    store_history: bool,
) -> RerunCollection:
    best_result = None
    restart_results = []
    for restart_index in range(n_restarts):
        result = run_single_search(
            disease_input=disease_input,
            similarity_matrix=similarity_matrix,
            baseline_score=baseline_score,
            n_rounds=n_rounds,
            max_genes_to_change=max_genes_to_change,
            rng_seed=base_seed + restart_index,
            require_unique_genes=require_unique_genes,
            store_history=store_history,
        )
        result.restart_index = restart_index
        if result.history_rows is not None:
            for row in result.history_rows:
                row["restart_index"] = restart_index
        restart_results.append(result)
        best_result = pick_better_result(best_result, result)
    if best_result is None:
        raise RuntimeError(f"No search result produced for {disease_input.disease_id}")
    return RerunCollection(
        best_result=best_result,
        restart_results=restart_results,
    )


def load_saved_round_summary(rounds_csv: Optional[Path]) -> dict:
    if rounds_csv is None or not rounds_csv.exists():
        return {
            "saved_rounds_recorded": None,
            "saved_best_score": None,
            "saved_baseline_score": None,
        }
    rows = read_csv_rows(rounds_csv)
    best_score = None
    baseline_score = None
    for row in rows:
        kept_score_text = (row.get("kept_avg_similarity") or "").strip()
        if kept_score_text:
            kept_score = float(kept_score_text)
            if best_score is None or kept_score > best_score:
                best_score = kept_score
        baseline_text = (row.get("causal_baseline_similarity") or "").strip()
        if baseline_text and baseline_score is None:
            baseline_score = float(baseline_text)
    return {
        "saved_rounds_recorded": len(rows),
        "saved_best_score": best_score,
        "saved_baseline_score": baseline_score,
    }


def load_random_summary(path: Path) -> Dict[str, dict]:
    summary = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            disease_id = (row.get("diseaseFromSourceMappedId") or "").strip()
            if disease_id:
                summary[disease_id] = row
    return summary


def load_global_summary(path: Path) -> Dict[str, dict]:
    summary = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            disease_id = (row.get("diseaseFromSourceMappedId") or "").strip()
            if disease_id:
                summary[disease_id] = row
    return summary


def load_random_distributions(path: Path, disease_ids: Iterable[str]) -> Dict[str, List[float]]:
    wanted = set(disease_ids)
    distributions: Dict[str, List[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            disease_id = (row.get("disease_id") or "").strip()
            if disease_id not in wanted:
                continue
            value_text = (row.get("random_avg_similarity") or "").strip()
            if not value_text:
                continue
            try:
                distributions[disease_id].append(float(value_text))
            except ValueError:
                continue
    return distributions


def summarize_random_support(
    score: Optional[float],
    random_values: Sequence[float],
    random_mean: Optional[float],
    random_sd: Optional[float],
) -> dict:
    if score is None:
        return {
            "empirical_p_random_ge_score": None,
            "z_score_vs_random": None,
            "random_n_iter_used": len(random_values),
        }

    empirical_p = None
    if random_values:
        n_ge = sum(value >= score for value in random_values)
        empirical_p = (n_ge + 1) / (len(random_values) + 1)

    z_score = None
    if random_mean is not None and random_sd not in (None, 0.0):
        z_score = (score - random_mean) / random_sd

    return {
        "empirical_p_random_ge_score": empirical_p,
        "z_score_vs_random": z_score,
        "random_n_iter_used": len(random_values),
    }


def selection_to_gene_string(seed_target_ids: Sequence[str], selection: Dict[str, Candidate]) -> str:
    return "|".join(selection[seed_target_id].matrix_gene_id for seed_target_id in seed_target_ids)


def selection_to_label_string(seed_target_ids: Sequence[str], selection: Dict[str, Candidate]) -> str:
    return "|".join(
        selection[seed_target_id].neighbor_symbol or selection[seed_target_id].matrix_gene_id
        for seed_target_id in seed_target_ids
    )


def summarize_selection_changes(
    seed_target_ids: Sequence[str],
    selection: Dict[str, Candidate],
) -> tuple[List[str], List[str], List[str], List[str]]:
    changed_seed_target_ids = []
    changed_gene_pairs = []
    replacement_genes = []
    replacement_symbols = []
    for seed_target_id in seed_target_ids:
        final_candidate = selection[seed_target_id]
        if final_candidate.matrix_gene_id == seed_target_id:
            continue
        changed_seed_target_ids.append(seed_target_id)
        changed_gene_pairs.append(f"{seed_target_id}->{final_candidate.matrix_gene_id}")
        replacement_genes.append(final_candidate.matrix_gene_id)
        replacement_symbols.append(final_candidate.neighbor_symbol or final_candidate.matrix_gene_id)
    return changed_seed_target_ids, changed_gene_pairs, replacement_genes, replacement_symbols


def summarize_restart_stability(
    disease_input: DiseaseInput,
    rerun_collection: RerunCollection,
    stability_threshold: float,
) -> dict:
    restart_gene_sets = []
    restart_label_sets = []
    restart_scores = []
    restart_pairs_used = []
    for result in rerun_collection.restart_results:
        restart_gene_sets.append(selection_to_gene_string(disease_input.seed_target_ids, result.best_selection))
        restart_label_sets.append(selection_to_label_string(disease_input.seed_target_ids, result.best_selection))
        if result.best_score is not None:
            restart_scores.append(result.best_score)
        restart_pairs_used.append(int(result.best_metrics.get("n_pairs_used") or 0))

    final_set_counts = Counter(restart_gene_sets)
    most_common_final_genes = ""
    most_common_final_gene_labels = ""
    most_common_final_genes_count = 0
    if final_set_counts:
        most_common_final_genes, most_common_final_genes_count = final_set_counts.most_common(1)[0]
        for gene_string, label_string in zip(restart_gene_sets, restart_label_sets):
            if gene_string == most_common_final_genes:
                most_common_final_gene_labels = label_string
                break

    best_final_genes = selection_to_gene_string(disease_input.seed_target_ids, rerun_collection.best_result.best_selection)
    best_final_gene_labels = selection_to_label_string(disease_input.seed_target_ids, rerun_collection.best_result.best_selection)
    best_final_genes_count = final_set_counts.get(best_final_genes, 0)
    n_restarts_run = len(rerun_collection.restart_results)
    best_final_genes_fraction = (best_final_genes_count / n_restarts_run) if n_restarts_run else 0.0
    most_common_final_genes_fraction = (most_common_final_genes_count / n_restarts_run) if n_restarts_run else 0.0

    return {
        "n_restarts_run": n_restarts_run,
        "n_unique_final_sets_across_restarts": len(final_set_counts),
        "best_final_genes": best_final_genes,
        "best_final_gene_labels": best_final_gene_labels,
        "best_final_genes_seen_in_n_restarts": best_final_genes_count,
        "best_final_genes_seen_fraction": best_final_genes_fraction,
        "most_common_final_genes_across_restarts": most_common_final_genes,
        "most_common_final_gene_labels_across_restarts": most_common_final_gene_labels,
        "most_common_final_genes_count": most_common_final_genes_count,
        "most_common_final_genes_fraction": most_common_final_genes_fraction,
        "stable_across_reruns": best_final_genes_fraction >= stability_threshold,
        "stable_across_reruns_strict": len(final_set_counts) == 1 if n_restarts_run else False,
        "stability_threshold": stability_threshold,
        "mean_best_score_across_restarts": statistics.mean(restart_scores) if restart_scores else None,
        "sd_best_score_across_restarts": statistics.stdev(restart_scores) if len(restart_scores) > 1 else None,
        "min_best_score_across_restarts": min(restart_scores) if restart_scores else None,
        "max_best_score_across_restarts": max(restart_scores) if restart_scores else None,
        "mean_n_pairs_used_across_restarts": statistics.mean(restart_pairs_used) if restart_pairs_used else None,
    }


def build_summary_and_swap_rows(
    disease_input: DiseaseInput,
    best_result: SearchResult,
    stability_summary: dict,
    saved_round_summary: dict,
    global_summary_row: Optional[dict],
    random_summary_row: Optional[dict],
    random_distribution: Sequence[float],
) -> tuple[dict, List[dict]]:
    baseline_score = None
    if global_summary_row:
        text = (global_summary_row.get("avg_pairwise_similarity") or "").strip()
        if text:
            baseline_score = float(text)
    if baseline_score is None:
        baseline_score = saved_round_summary["saved_baseline_score"]

    random_mean = None
    random_sd = None
    original_empirical_p = None
    original_z = None
    if random_summary_row:
        random_mean_text = (random_summary_row.get("random_mean_similarity") or "").strip()
        random_sd_text = (random_summary_row.get("random_sd_similarity") or "").strip()
        empirical_text = (random_summary_row.get("empirical_p_random_ge_observed") or "").strip()
        z_text = (random_summary_row.get("z_score_vs_random") or "").strip()
        if random_mean_text:
            random_mean = float(random_mean_text)
        if random_sd_text:
            random_sd = float(random_sd_text)
        if empirical_text:
            original_empirical_p = float(empirical_text)
        if z_text:
            original_z = float(z_text)

    final_support = summarize_random_support(
        score=best_result.best_score,
        random_values=random_distribution,
        random_mean=random_mean,
        random_sd=random_sd,
    )

    final_genes = selection_to_gene_string(disease_input.seed_target_ids, best_result.best_selection)
    final_gene_labels = selection_to_label_string(disease_input.seed_target_ids, best_result.best_selection)
    original_genes = "|".join(disease_input.seed_target_ids)
    (
        changed_seed_target_ids,
        changed_gene_pairs,
        replacement_genes,
        replacement_symbols,
    ) = summarize_selection_changes(
        disease_input.seed_target_ids,
        best_result.best_selection,
    )
    final_beats_random_null = (
        final_support["empirical_p_random_ge_score"] is not None
        and final_support["empirical_p_random_ge_score"] <= 0.05
    )
    original_beats_random_null = (
        original_empirical_p is not None
        and original_empirical_p <= 0.05
    )

    swap_rows = []
    for seed_target_id in disease_input.seed_target_ids:
        final_candidate = best_result.best_selection[seed_target_id]
        if final_candidate.matrix_gene_id == seed_target_id:
            continue
        swap_rows.append(
            {
                "folder_name": disease_input.folder_name,
                "disease_id": disease_input.disease_id,
                "seed_target_id": seed_target_id,
                "original_gene": seed_target_id,
                "replacement_gene": final_candidate.matrix_gene_id,
                "replacement_symbol": final_candidate.neighbor_symbol,
                "replacement_direction": final_candidate.neighbor_direction,
                "replacement_neighbor_rank": final_candidate.neighbor_rank,
                "original_score": baseline_score,
                "saved_best_score": saved_round_summary["saved_best_score"],
                "rerun_best_score": best_result.best_score,
                "rerun_initial_score": best_result.initial_score,
                "score_gain_vs_original": (
                    best_result.best_score - baseline_score
                    if best_result.best_score is not None and baseline_score is not None
                    else None
                ),
                "score_gain_vs_saved_best": (
                    best_result.best_score - saved_round_summary["saved_best_score"]
                    if best_result.best_score is not None and saved_round_summary["saved_best_score"] is not None
                    else None
                ),
                "final_empirical_p_random_ge_score": final_support["empirical_p_random_ge_score"],
                "final_z_score_vs_random": final_support["z_score_vs_random"],
                "final_beats_random_null_p_le_0_05": final_beats_random_null,
                "saved_original_empirical_p_random_ge_score": original_empirical_p,
                "saved_original_z_score_vs_random": original_z,
                "saved_original_beats_random_null_p_le_0_05": original_beats_random_null,
            }
        )

    summary_row = {
        "folder_name": disease_input.folder_name,
        "disease_id": disease_input.disease_id,
        "disease_dir": disease_input.disease_dir.as_posix(),
        "causal_csv": disease_input.causal_csv.as_posix(),
        "pocket_candidates_csv": disease_input.pocket_candidates_csv.as_posix(),
        "matrix_subset_csv": disease_input.matrix_subset_csv.as_posix(),
        "saved_rounds_csv": disease_input.saved_rounds_csv.as_posix() if disease_input.saved_rounds_csv else "",
        "n_seed_genes": len(disease_input.seed_target_ids),
        "pocket_sizes": "|".join(str(len(disease_input.pockets[seed_target_id])) for seed_target_id in disease_input.seed_target_ids),
        "n_changeable_pockets": sum(len(disease_input.pockets[seed_target_id]) > 1 for seed_target_id in disease_input.seed_target_ids),
        "original_genes": original_genes,
        "saved_original_score": baseline_score,
        "saved_best_score": saved_round_summary["saved_best_score"],
        "saved_rounds_recorded": saved_round_summary["saved_rounds_recorded"],
        "n_restarts_run": stability_summary["n_restarts_run"],
        "rerun_restart_index": best_result.restart_index,
        "rerun_initial_score": best_result.initial_score,
        "rerun_best_score": best_result.best_score,
        "rerun_rounds_run": best_result.rounds_run,
        "rerun_accepted_rounds": best_result.accepted_rounds,
        "rerun_stop_reason": best_result.stop_reason,
        "rerun_baseline_hit_round": best_result.baseline_hit_round,
        "rerun_planned_stop_round": best_result.planned_stop_round,
        "final_genes": final_genes,
        "final_gene_labels": final_gene_labels,
        "changed_seed_count": len(changed_seed_target_ids),
        "changed_seed_target_ids": "|".join(changed_seed_target_ids),
        "changed_gene_pairs": "|".join(changed_gene_pairs),
        "replacement_genes": "|".join(replacement_genes),
        "replacement_gene_symbols": "|".join(replacement_symbols),
        "score_gain_vs_original": (
            best_result.best_score - baseline_score
            if best_result.best_score is not None and baseline_score is not None
            else None
        ),
        "score_gain_vs_saved_best": (
            best_result.best_score - saved_round_summary["saved_best_score"]
            if best_result.best_score is not None and saved_round_summary["saved_best_score"] is not None
            else None
        ),
        "score_gain_vs_initial": (
            best_result.best_score - best_result.initial_score
            if best_result.best_score is not None and best_result.initial_score is not None
            else None
        ),
        "rerun_best_n_pairs_used": best_result.best_metrics["n_pairs_used"],
        "rerun_best_n_pairs_missing": best_result.best_metrics["n_pairs_missing"],
        "rerun_best_missing_pairs_example": best_result.best_metrics["missing_pairs_example"],
        "n_unique_final_sets_across_restarts": stability_summary["n_unique_final_sets_across_restarts"],
        "best_final_genes_seen_in_n_restarts": stability_summary["best_final_genes_seen_in_n_restarts"],
        "best_final_genes_seen_fraction": stability_summary["best_final_genes_seen_fraction"],
        "most_common_final_genes_across_restarts": stability_summary["most_common_final_genes_across_restarts"],
        "most_common_final_gene_labels_across_restarts": stability_summary["most_common_final_gene_labels_across_restarts"],
        "most_common_final_genes_count": stability_summary["most_common_final_genes_count"],
        "most_common_final_genes_fraction": stability_summary["most_common_final_genes_fraction"],
        "stable_across_reruns": stability_summary["stable_across_reruns"],
        "stable_across_reruns_strict": stability_summary["stable_across_reruns_strict"],
        "stability_threshold": stability_summary["stability_threshold"],
        "mean_best_score_across_restarts": stability_summary["mean_best_score_across_restarts"],
        "sd_best_score_across_restarts": stability_summary["sd_best_score_across_restarts"],
        "min_best_score_across_restarts": stability_summary["min_best_score_across_restarts"],
        "max_best_score_across_restarts": stability_summary["max_best_score_across_restarts"],
        "mean_n_pairs_used_across_restarts": stability_summary["mean_n_pairs_used_across_restarts"],
        "final_empirical_p_random_ge_score": final_support["empirical_p_random_ge_score"],
        "final_z_score_vs_random": final_support["z_score_vs_random"],
        "final_beats_random_null_p_le_0_05": final_beats_random_null,
        "saved_original_empirical_p_random_ge_score": original_empirical_p,
        "saved_original_z_score_vs_random": original_z,
        "saved_original_beats_random_null_p_le_0_05": original_beats_random_null,
    }
    return summary_row, swap_rows


def build_restart_endpoint_rows(
    disease_input: DiseaseInput,
    rerun_collection: RerunCollection,
) -> List[dict]:
    rows = []
    best_restart_index = rerun_collection.best_result.restart_index
    for result in rerun_collection.restart_results:
        changed_seed_target_ids, changed_gene_pairs, replacement_genes, replacement_symbols = summarize_selection_changes(
            disease_input.seed_target_ids,
            result.best_selection,
        )
        rows.append(
            {
                "folder_name": disease_input.folder_name,
                "disease_id": disease_input.disease_id,
                "restart_index": result.restart_index,
                "is_best_overall_restart": result.restart_index == best_restart_index,
                "final_genes": selection_to_gene_string(disease_input.seed_target_ids, result.best_selection),
                "final_gene_labels": selection_to_label_string(disease_input.seed_target_ids, result.best_selection),
                "changed_seed_count": len(changed_seed_target_ids),
                "changed_seed_target_ids": "|".join(changed_seed_target_ids),
                "changed_gene_pairs": "|".join(changed_gene_pairs),
                "replacement_genes": "|".join(replacement_genes),
                "replacement_gene_symbols": "|".join(replacement_symbols),
                "best_score": result.best_score,
                "best_n_pairs_used": result.best_metrics["n_pairs_used"],
                "best_n_pairs_missing": result.best_metrics["n_pairs_missing"],
                "accepted_rounds": result.accepted_rounds,
                "rounds_run": result.rounds_run,
                "stop_reason": result.stop_reason,
                "baseline_hit_round": result.baseline_hit_round,
            }
        )
    return rows


def build_replacement_frequency_rows(swap_rows: Sequence[dict]) -> List[dict]:
    disease_sets: Dict[str, set] = defaultdict(set)
    folder_sets: Dict[str, set] = defaultdict(set)
    original_gene_sets: Dict[str, set] = defaultdict(set)
    score_gains: Dict[str, List[float]] = defaultdict(list)
    symbols: Dict[str, str] = {}

    for row in swap_rows:
        replacement_gene = row["replacement_gene"]
        disease_sets[replacement_gene].add(row["disease_id"])
        folder_sets[replacement_gene].add(row["folder_name"])
        original_gene_sets[replacement_gene].add(row["original_gene"])
        if row.get("replacement_symbol"):
            symbols[replacement_gene] = row["replacement_symbol"]
        score_gain = row.get("score_gain_vs_original")
        if score_gain is not None:
            score_gains[replacement_gene].append(float(score_gain))

    total_swap_counts = Counter(row["replacement_gene"] for row in swap_rows)
    rows = []
    for replacement_gene, total_swap_count in total_swap_counts.items():
        gains = score_gains.get(replacement_gene, [])
        rows.append(
            {
                "replacement_gene": replacement_gene,
                "replacement_symbol": symbols.get(replacement_gene, ""),
                "n_replacement_swaps_total": total_swap_count,
                "n_diseases_with_replacement_gene": len(disease_sets[replacement_gene]),
                "disease_ids": "|".join(sorted(disease_sets[replacement_gene])),
                "folder_names": "|".join(sorted(folder_sets[replacement_gene])),
                "n_unique_original_genes_replaced": len(original_gene_sets[replacement_gene]),
                "original_genes_replaced": "|".join(sorted(original_gene_sets[replacement_gene])),
                "mean_score_gain_vs_original": statistics.mean(gains) if gains else None,
                "max_score_gain_vs_original": max(gains) if gains else None,
            }
        )

    rows.sort(
        key=lambda row: (
            -int(row["n_diseases_with_replacement_gene"]),
            -int(row["n_replacement_swaps_total"]),
            row["replacement_gene"],
        )
    )
    return rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_history(output_dir: Path, disease_input: DiseaseInput, result: SearchResult) -> None:
    if result.history_rows is None:
        return
    history_dir = output_dir / "rerun_histories"
    if output_dir.name != disease_input.folder_name:
        history_dir = history_dir / disease_input.folder_name
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{disease_input.disease_id}_rerun_history.csv"
    fieldnames = [
        "restart_index",
        "round",
        "round_kind",
        "n_changed_genes",
        "pocket_seed_target_ids",
        "previous_gene",
        "proposed_gene",
        "accepted",
        "kept_gene_after_round",
        "kept_score",
        "proposal_score",
        "kept_n_pairs_used",
        "kept_n_pairs_missing",
        "kept_missing_pairs_example",
        "baseline_hit_round",
        "planned_stop_round",
    ]
    write_csv(history_path, fieldnames, result.history_rows)


def build_run_summary(summary_rows: Sequence[dict]) -> dict:
    gain_vs_original = [
        float(row["score_gain_vs_original"])
        for row in summary_rows
        if row.get("score_gain_vs_original") not in (None, "", "nan")
    ]
    gain_vs_saved_best = [
        float(row["score_gain_vs_saved_best"])
        for row in summary_rows
        if row.get("score_gain_vs_saved_best") not in (None, "", "nan")
    ]
    return {
        "n_diseases": len(summary_rows),
        "n_changed_final_sets": sum(int(row["changed_seed_count"]) > 0 for row in summary_rows),
        "n_stable_across_reruns": sum(bool(row.get("stable_across_reruns")) for row in summary_rows),
        "n_final_sets_empirical_p_le_0_05": sum(
            row.get("final_empirical_p_random_ge_score") not in (None, "")
            and float(row["final_empirical_p_random_ge_score"]) <= 0.05
            for row in summary_rows
        ),
        "n_final_sets_beating_random_null": sum(
            bool(row.get("final_beats_random_null_p_le_0_05"))
            for row in summary_rows
        ),
        "mean_gain_vs_original": statistics.mean(gain_vs_original) if gain_vs_original else None,
        "median_gain_vs_original": statistics.median(gain_vs_original) if gain_vs_original else None,
        "mean_gain_vs_saved_best": statistics.mean(gain_vs_saved_best) if gain_vs_saved_best else None,
        "median_gain_vs_saved_best": statistics.median(gain_vs_saved_best) if gain_vs_saved_best else None,
    }


def format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.12f}"


def main() -> int:
    args = parse_args()
    if args.n_rounds < 1:
        print("--n-rounds must be at least 1", file=sys.stderr)
        return 2
    if args.n_restarts < 1:
        print("--n-restarts must be at least 1", file=sys.stderr)
        return 2
    if args.max_genes_to_change < 1:
        print("--max-genes-to-change must be at least 1", file=sys.stderr)
        return 2
    if not 0.0 < args.stability_threshold <= 1.0:
        print("--stability-threshold must be > 0 and <= 1", file=sys.stderr)
        return 2

    project_root = args.project_root.expanduser().resolve()

    disease_inputs = discover_disease_inputs(
        project_root=project_root,
        wanted_folder_names=args.folder_names,
        wanted_disease_ids=args.disease_ids,
    )
    if not disease_inputs:
        print("No matching per-disease pocket model inputs were found.", file=sys.stderr)
        return 2

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else default_output_dir(project_root, [item.folder_name for item in disease_inputs])
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    random_summary = load_random_summary(args.random_summary_file.expanduser().resolve())
    global_summary = load_global_summary(args.global_summary_file.expanduser().resolve())
    random_distributions = load_random_distributions(
        args.random_distribution_file.expanduser().resolve(),
        disease_ids=[item.disease_id for item in disease_inputs],
    )

    summary_rows = []
    swap_rows = []
    restart_endpoint_rows = []

    for disease_input in disease_inputs:
        subset_matrix = load_subset_matrix(disease_input.matrix_subset_csv)
        saved_round_summary = load_saved_round_summary(disease_input.saved_rounds_csv)

        baseline_score = None
        global_summary_row = global_summary.get(disease_input.disease_id)
        if global_summary_row:
            text = (global_summary_row.get("avg_pairwise_similarity") or "").strip()
            if text:
                baseline_score = float(text)
        if baseline_score is None:
            baseline_score = saved_round_summary["saved_baseline_score"]

        rerun_collection = rerun_with_restarts(
            disease_input=disease_input,
            similarity_matrix=subset_matrix,
            baseline_score=baseline_score,
            n_rounds=args.n_rounds,
            n_restarts=args.n_restarts,
            max_genes_to_change=args.max_genes_to_change,
            base_seed=args.random_seed,
            require_unique_genes=args.require_unique_genes,
            store_history=args.write_round_history,
        )
        stability_summary = summarize_restart_stability(
            disease_input=disease_input,
            rerun_collection=rerun_collection,
            stability_threshold=args.stability_threshold,
        )
        if args.write_round_history:
            write_history(output_dir, disease_input, rerun_collection.best_result)

        summary_row, disease_swap_rows = build_summary_and_swap_rows(
            disease_input=disease_input,
            best_result=rerun_collection.best_result,
            stability_summary=stability_summary,
            saved_round_summary=saved_round_summary,
            global_summary_row=global_summary_row,
            random_summary_row=random_summary.get(disease_input.disease_id),
            random_distribution=random_distributions.get(disease_input.disease_id, []),
        )
        summary_rows.append(summary_row)
        swap_rows.extend(disease_swap_rows)
        restart_endpoint_rows.extend(
            build_restart_endpoint_rows(
                disease_input=disease_input,
                rerun_collection=rerun_collection,
            )
        )

    replacement_counts = Counter(row["replacement_gene"] for row in swap_rows)
    replacement_disease_ids: Dict[str, set] = defaultdict(set)
    for row in swap_rows:
        replacement_disease_ids[row["replacement_gene"]].add(row["disease_id"])
    for row in swap_rows:
        row["replacement_gene_repeat_count_across_diseases"] = len(replacement_disease_ids[row["replacement_gene"]])
        row["replacement_gene_total_swap_count"] = replacement_counts[row["replacement_gene"]]

    swap_rows_by_disease: Dict[tuple[str, str], List[dict]] = defaultdict(list)
    for row in swap_rows:
        swap_rows_by_disease[(row["folder_name"], row["disease_id"])].append(row)
    for summary_row in summary_rows:
        disease_swap_rows = swap_rows_by_disease[(summary_row["folder_name"], summary_row["disease_id"])]
        repeat_counts = [
            str(len(replacement_disease_ids[row["replacement_gene"]]))
            for row in disease_swap_rows
        ]
        summary_row["replacement_gene_repeat_counts_across_diseases"] = "|".join(repeat_counts)
        summary_row["max_replacement_gene_repeat_count_across_diseases"] = max(
            (len(replacement_disease_ids[row["replacement_gene"]]) for row in disease_swap_rows),
            default=0,
        )
        summary_row["n_replacement_genes_seen_in_multiple_diseases"] = sum(
            len(replacement_disease_ids[row["replacement_gene"]]) > 1
            for row in disease_swap_rows
        )

    replacement_frequency_rows = build_replacement_frequency_rows(swap_rows)

    summary_rows.sort(key=lambda row: (row["folder_name"], row["disease_id"]))
    restart_endpoint_rows.sort(key=lambda row: (row["folder_name"], row["disease_id"], row["restart_index"]))
    swap_rows.sort(
        key=lambda row: (
            row["score_gain_vs_original"] is None,
            -(row["score_gain_vs_original"] if row["score_gain_vs_original"] is not None else 0.0),
            row["final_empirical_p_random_ge_score"] if row["final_empirical_p_random_ge_score"] is not None else 1.0,
            row["folder_name"],
            row["disease_id"],
            row["seed_target_id"],
        )
    )
    for rank, row in enumerate(swap_rows, start=1):
        row["overall_rank"] = rank

    summary_fieldnames = [
        "folder_name",
        "disease_id",
        "disease_dir",
        "causal_csv",
        "pocket_candidates_csv",
        "matrix_subset_csv",
        "saved_rounds_csv",
        "n_seed_genes",
        "pocket_sizes",
        "n_changeable_pockets",
        "original_genes",
        "saved_original_score",
        "saved_best_score",
        "saved_rounds_recorded",
        "n_restarts_run",
        "rerun_restart_index",
        "rerun_initial_score",
        "rerun_best_score",
        "rerun_rounds_run",
        "rerun_accepted_rounds",
        "rerun_stop_reason",
        "rerun_baseline_hit_round",
        "rerun_planned_stop_round",
        "final_genes",
        "final_gene_labels",
        "changed_seed_count",
        "changed_seed_target_ids",
        "changed_gene_pairs",
        "replacement_genes",
        "replacement_gene_symbols",
        "replacement_gene_repeat_counts_across_diseases",
        "max_replacement_gene_repeat_count_across_diseases",
        "n_replacement_genes_seen_in_multiple_diseases",
        "score_gain_vs_original",
        "score_gain_vs_saved_best",
        "score_gain_vs_initial",
        "rerun_best_n_pairs_used",
        "rerun_best_n_pairs_missing",
        "rerun_best_missing_pairs_example",
        "n_unique_final_sets_across_restarts",
        "best_final_genes_seen_in_n_restarts",
        "best_final_genes_seen_fraction",
        "most_common_final_genes_across_restarts",
        "most_common_final_gene_labels_across_restarts",
        "most_common_final_genes_count",
        "most_common_final_genes_fraction",
        "stable_across_reruns",
        "stable_across_reruns_strict",
        "stability_threshold",
        "mean_best_score_across_restarts",
        "sd_best_score_across_restarts",
        "min_best_score_across_restarts",
        "max_best_score_across_restarts",
        "mean_n_pairs_used_across_restarts",
        "final_empirical_p_random_ge_score",
        "final_z_score_vs_random",
        "final_beats_random_null_p_le_0_05",
        "saved_original_empirical_p_random_ge_score",
        "saved_original_z_score_vs_random",
        "saved_original_beats_random_null_p_le_0_05",
    ]
    evaluation_fieldnames = [
        "folder_name",
        "disease_id",
        "n_seed_genes",
        "original_genes",
        "final_genes",
        "final_gene_labels",
        "changed_gene_pairs",
        "replacement_genes",
        "replacement_gene_symbols",
        "saved_original_score",
        "rerun_best_score",
        "score_gain_vs_original",
        "n_restarts_run",
        "n_unique_final_sets_across_restarts",
        "best_final_genes_seen_in_n_restarts",
        "best_final_genes_seen_fraction",
        "most_common_final_genes_across_restarts",
        "most_common_final_genes_fraction",
        "stable_across_reruns",
        "stability_threshold",
        "final_empirical_p_random_ge_score",
        "final_z_score_vs_random",
        "final_beats_random_null_p_le_0_05",
        "replacement_gene_repeat_counts_across_diseases",
        "max_replacement_gene_repeat_count_across_diseases",
        "n_replacement_genes_seen_in_multiple_diseases",
    ]
    swap_fieldnames = [
        "overall_rank",
        "folder_name",
        "disease_id",
        "seed_target_id",
        "original_gene",
        "replacement_gene",
        "replacement_symbol",
        "replacement_direction",
        "replacement_neighbor_rank",
        "replacement_gene_repeat_count_across_diseases",
        "replacement_gene_total_swap_count",
        "original_score",
        "saved_best_score",
        "rerun_initial_score",
        "rerun_best_score",
        "score_gain_vs_original",
        "score_gain_vs_saved_best",
        "final_empirical_p_random_ge_score",
        "final_z_score_vs_random",
        "final_beats_random_null_p_le_0_05",
        "saved_original_empirical_p_random_ge_score",
        "saved_original_z_score_vs_random",
        "saved_original_beats_random_null_p_le_0_05",
    ]
    restart_endpoint_fieldnames = [
        "folder_name",
        "disease_id",
        "restart_index",
        "is_best_overall_restart",
        "final_genes",
        "final_gene_labels",
        "changed_seed_count",
        "changed_seed_target_ids",
        "changed_gene_pairs",
        "replacement_genes",
        "replacement_gene_symbols",
        "best_score",
        "best_n_pairs_used",
        "best_n_pairs_missing",
        "accepted_rounds",
        "rounds_run",
        "stop_reason",
        "baseline_hit_round",
    ]
    replacement_frequency_fieldnames = [
        "replacement_gene",
        "replacement_symbol",
        "n_replacement_swaps_total",
        "n_diseases_with_replacement_gene",
        "disease_ids",
        "folder_names",
        "n_unique_original_genes_replaced",
        "original_genes_replaced",
        "mean_score_gain_vs_original",
        "max_score_gain_vs_original",
    ]

    write_csv(output_dir / "rerun_summary.csv", summary_fieldnames, summary_rows)
    write_csv(output_dir / "disease_evaluation_table.csv", evaluation_fieldnames, summary_rows)
    write_csv(output_dir / "ranked_candidate_swaps.csv", swap_fieldnames, swap_rows)
    write_csv(output_dir / "restart_endpoint_summary.csv", restart_endpoint_fieldnames, restart_endpoint_rows)
    write_csv(output_dir / "replacement_gene_frequency.csv", replacement_frequency_fieldnames, replacement_frequency_rows)

    run_summary = build_run_summary(summary_rows)
    write_csv(output_dir / "run_summary.csv", list(run_summary.keys()), [run_summary])

    print(f"Output directory: {output_dir}")
    print(f"Diseases processed: {len(summary_rows)}")
    print(f"Changed final sets: {run_summary['n_changed_final_sets']}")
    print(f"Stable across reruns: {run_summary['n_stable_across_reruns']}")
    print(f"Final sets with empirical p <= 0.05: {run_summary['n_final_sets_empirical_p_le_0_05']}")
    print(f"Mean gain vs original: {format_float(run_summary['mean_gain_vs_original'])}")
    print(f"Mean gain vs saved best: {format_float(run_summary['mean_gain_vs_saved_best'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
