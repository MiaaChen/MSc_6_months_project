#!/usr/bin/env python3
"""Enrich disease gene CSVs with nearby genomic neighbors from NCBI.

The script expects disease CSVs with at least:
    diseaseFromSourceMappedId,targetId

`targetId` is assumed to be a human Ensembl gene ID (for example ENSG...).
It downloads and caches:
    1. A human NCBI gene package containing `data_report.jsonl`
    2. The GRCh38.p14 genome annotation package containing `genomic.gff`

The NCBI gene report is used to map Ensembl IDs to NCBI Gene IDs and symbols.
The genome annotation GFF3 is used as the source of genomic coordinates and
neighbor ordering.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "separate_disease"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "separate_disease_ncbi_neighbors"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "database" / "ncbi_cache"
DEFAULT_MATRIX_FILE = PROJECT_ROOT / "database" / "Matrix" / "gene_gene_correlation_TPM_updated.csv"
DEFAULT_ASSEMBLY = "GCF_000001405.40"
DEFAULT_NEIGHBOR_COUNT = 2
HUMAN_TAXON = "human"
HUMAN_TAXON_ID = "9606"
PRIMARY_CHROMOSOMES = {str(i) for i in range(1, 23)} | {"X", "Y", "MT", "M"}
USER_AGENT = "KCL-ncbi-neighbor-enrichment/1.0"


class DownloadError(RuntimeError):
    """Raised when NCBI download attempts fail."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate each disease CSV with the nearby genomic neighbors of each "
            "seed gene using NCBI gene metadata and genome annotation."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Root containing *_gene_related_disease folders (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root for enriched CSVs (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for NCBI downloads (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--matrix-file",
        type=Path,
        default=DEFAULT_MATRIX_FILE,
        help=(
            "Gene-gene matrix CSV. Neighbor rows are emitted only when at least "
            f"one neighbor Ensembl ID is present in this matrix (default: {DEFAULT_MATRIX_FILE})"
        ),
    )
    parser.add_argument(
        "--assembly",
        default=DEFAULT_ASSEMBLY,
        help=f"NCBI assembly accession to use (default: {DEFAULT_ASSEMBLY})",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=DEFAULT_NEIGHBOR_COUNT,
        help=f"Number of upstream and downstream neighbors to emit (default: {DEFAULT_NEIGHBOR_COUNT})",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force re-download of cached NCBI files.",
    )
    return parser.parse_args()


def build_headers() -> Dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
    }
    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    if api_key:
        headers["api-key"] = api_key
    return headers


def fetch_binary_with_retries(
    url: str,
    dest_path: Path,
    headers: Dict[str, str],
    attempts: int = 3,
    timeout: int = 120,
) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response, dest_path.open("wb") as out:
                shutil.copyfileobj(response, out)
            return dest_path
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if dest_path.exists():
                dest_path.unlink()
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise DownloadError(f"Failed to download {url}: {last_error}")


def fetch_json_with_retries(
    url: str,
    headers: Dict[str, str],
    attempts: int = 3,
    timeout: int = 120,
) -> dict:
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise DownloadError(f"Failed to fetch JSON {url}: {last_error}")


def ensure_zip_download(
    cache_dir: Path,
    filename: str,
    candidate_urls: Sequence[str],
    headers: Dict[str, str],
    refresh_cache: bool,
) -> Path:
    zip_path = cache_dir / filename
    if zip_path.exists() and not refresh_cache:
        return zip_path

    errors: List[str] = []
    for url in candidate_urls:
        try:
            return fetch_binary_with_retries(url, zip_path, headers=headers)
        except DownloadError as exc:
            errors.append(str(exc))
    raise DownloadError(" | ".join(errors))


def gene_dataset_report_urls(page_size: int = 1000) -> List[str]:
    encoded_taxon = urllib.parse.quote(HUMAN_TAXON_ID)
    return [
        (
            "https://api.ncbi.nlm.nih.gov/datasets/v2/gene/taxon/"
            f"{encoded_taxon}/dataset_report?returned_content=COMPLETE&page_size={page_size}"
        ),
        (
            "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/taxon/"
            f"{encoded_taxon}/dataset_report?returned_content=COMPLETE&page_size={page_size}"
        ),
    ]


def genome_package_urls(assembly: str, filename: str) -> List[str]:
    encoded_filename = urllib.parse.quote(filename)
    return [
        (
            "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
            f"{assembly}/download?include_annotation_type=GENOME_GFF"
            f"&include_annotation_type=SEQUENCE_REPORT&filename={encoded_filename}"
        ),
        (
            "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/"
            f"{assembly}/download?include_annotation_type=GENOME_GFF"
            f"&include_annotation_type=SEQUENCE_REPORT&filename={encoded_filename}"
        ),
    ]


def ensure_gene_report_cache(
    cache_dir: Path,
    filename: str,
    headers: Dict[str, str],
    refresh_cache: bool,
) -> Path:
    cache_path = cache_dir / filename
    if cache_path.exists() and not refresh_cache:
        return cache_path

    errors: List[str] = []
    for base_url in gene_dataset_report_urls():
        page_token = ""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as out:
                while True:
                    url = base_url
                    if page_token:
                        url = f"{base_url}&page_token={urllib.parse.quote(page_token)}"
                    response = fetch_json_with_retries(
                        url,
                        headers={**headers, "Accept": "application/json"},
                    )
                    reports = response.get("reports") or []
                    for report in reports:
                        gene = report.get("gene") if isinstance(report, dict) else None
                        if isinstance(gene, dict):
                            out.write(json.dumps(gene, separators=(",", ":")))
                            out.write("\n")
                    page_token = str(response.get("next_page_token") or "").strip()
                    if not page_token:
                        return cache_path
        except DownloadError as exc:
            errors.append(str(exc))
            if cache_path.exists():
                cache_path.unlink()
    raise DownloadError(" | ".join(errors))


def iter_zip_jsonl(zip_path: Path, suffix: str) -> Iterator[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        member = find_single_member(zf, suffix=suffix)
        with zf.open(member) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict) and "gene" in payload and isinstance(payload["gene"], dict):
                    yield payload["gene"]
                elif isinstance(payload, dict) and "reports" in payload:
                    for report in payload.get("reports", []):
                        if isinstance(report, dict):
                            if "gene" in report and isinstance(report["gene"], dict):
                                yield report["gene"]
                            else:
                                yield report
                else:
                    yield payload


def iter_zip_text(zip_path: Path, suffix: str) -> Iterator[str]:
    with zipfile.ZipFile(zip_path) as zf:
        member = find_single_member(zf, suffix=suffix)
        with zf.open(member) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                yield line


def find_single_member(zf: zipfile.ZipFile, suffix: str) -> str:
    members = [name for name in zf.namelist() if name.endswith(suffix)]
    if not members:
        raise FileNotFoundError(f"No zip member ends with {suffix}")
    if len(members) > 1:
        exact = [name for name in members if name.count("/") == members[0].count("/")]
        if len(exact) == 1:
            return exact[0]
    return members[0]


def iter_jsonl_file(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_attributes(raw_attributes: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for item in raw_attributes.strip().split(";"):
        if not item:
            continue
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        attrs[key] = urllib.parse.unquote(value)
    return attrs


def parse_dbxref_gene_id(dbxref: str) -> Optional[str]:
    for item in dbxref.split(","):
        if item.startswith("GeneID:"):
            return item.split(":", 1)[1]
    return None


def normalize_orientation(value: Optional[str]) -> str:
    if not value:
        return ""
    value = value.lower()
    if value == "plus":
        return "+"
    if value == "minus":
        return "-"
    if value in {"+", "-"}:
        return value
    return value


def normalize_seq_name(name: str) -> str:
    value = name.strip().upper()
    if value in {"M", "MT", "MITOCHONDRION", "MITO"}:
        return "MT"
    return value


def is_primary_chromosome(name: str) -> bool:
    return normalize_seq_name(name) in PRIMARY_CHROMOSOMES


def parse_seq_report(zip_path: Path, assembly: str) -> Dict[str, str]:
    seqid_to_chr: Dict[str, str] = {}
    try:
        for record in iter_zip_jsonl(zip_path, suffix=f"{assembly}/sequence_report.jsonl"):
            refseq_accession = str(
                record.get("refseqAccession")
                or record.get("refseq_accession")
                or ""
            ).strip()
            genbank_accession = str(
                record.get("genbankAccession")
                or record.get("genbank_accession")
                or ""
            ).strip()
            accession = str(record.get("accession") or "").strip()
            chr_name = (
                str(
                    record.get("assignedMolecule")
                    or record.get("chrName")
                    or record.get("chromosome")
                    or record.get("sequenceName")
                    or ""
                ).strip()
            )
            role = str(record.get("role") or "").strip().lower()
            if not chr_name:
                continue
            if role and role != "assembled-molecule":
                continue
            if is_primary_chromosome(chr_name):
                normalized_chr = normalize_seq_name(chr_name)
                for seqid in (refseq_accession, genbank_accession, accession):
                    if seqid:
                        seqid_to_chr[seqid] = normalized_chr
    except FileNotFoundError:
        pass
    return seqid_to_chr


def build_gene_indexes(
    gene_report_path: Path,
    assembly: str,
) -> Tuple[Dict[str, List[str]], Dict[str, dict], Dict[str, List[str]]]:
    ensembl_to_gene_ids: Dict[str, List[str]] = defaultdict(list)
    gene_records: Dict[str, dict] = {}
    symbol_to_gene_ids: Dict[str, List[str]] = defaultdict(list)

    for record in iter_jsonl_file(gene_report_path):
        gene_id = str(record.get("gene_id") or record.get("geneId") or "").strip()
        if not gene_id:
            continue

        annotations = record.get("annotations") or []
        assembly_annotation = None
        for annotation in annotations:
            if str(annotation.get("assembly_accession") or annotation.get("assemblyAccession") or "").strip() == assembly:
                assembly_annotation = annotation
                break

        gene_records[gene_id] = {
            "gene_id": gene_id,
            "symbol": str(record.get("symbol") or "").strip(),
            "biotype": str(record.get("gene_type") or record.get("type") or "").strip(),
            "orientation": normalize_orientation(record.get("orientation")),
            "ensembl_ids": [
                str(x).strip()
                for x in (record.get("ensembl_gene_ids") or record.get("ensemblGeneIds") or [])
                if str(x).strip()
            ],
            "assembly_annotation": assembly_annotation,
        }

        for ensembl_id in gene_records[gene_id]["ensembl_ids"]:
            ensembl_to_gene_ids[ensembl_id].append(gene_id)

        symbol = gene_records[gene_id]["symbol"]
        if symbol:
            symbol_to_gene_ids[symbol].append(gene_id)

    for mapping in (ensembl_to_gene_ids, symbol_to_gene_ids):
        for key, values in list(mapping.items()):
            mapping[key] = sorted(set(values))

    return ensembl_to_gene_ids, gene_records, symbol_to_gene_ids


def build_local_symbol_map(raw_data_path: Path) -> Dict[str, List[str]]:
    symbol_map: Dict[str, set] = defaultdict(set)
    if not raw_data_path.exists():
        return {}

    with raw_data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            target_id = (row.get("targetId") or "").strip()
            symbol = (row.get("targetFromSourceId") or "").strip()
            if not target_id or not symbol or symbol.startswith("ENSG"):
                continue
            symbol_map[target_id].add(symbol)

    return {key: sorted(values) for key, values in symbol_map.items()}


def build_gff_index(
    genome_zip: Path,
    assembly: str,
    gene_records: Dict[str, dict],
) -> Tuple[Dict[str, List[dict]], Dict[str, dict]]:
    seqid_to_chr = parse_seq_report(genome_zip, assembly)
    genes_by_chr: Dict[str, List[dict]] = defaultdict(list)
    gene_index: Dict[str, dict] = {}

    for line in iter_zip_text(genome_zip, suffix=f"{assembly}/genomic.gff"):
        if not line or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 9:
            continue
        seqid, source, feature_type, start, end, score, strand, phase, attrs = parts

        attributes = parse_attributes(attrs)
        if attributes.get("gbkey", "").strip() != "Gene":
            continue
        if feature_type not in {"gene", "pseudogene"}:
            continue
        gene_id = (
            parse_dbxref_gene_id(attributes.get("Dbxref", ""))
            or attributes.get("GeneID")
            or attributes.get("gene_id")
            or ""
        ).strip()
        if not gene_id:
            continue

        chromosome = attributes.get("chromosome", "").strip()
        if chromosome:
            chromosome = normalize_seq_name(chromosome)
        elif seqid in seqid_to_chr:
            chromosome = seqid_to_chr[seqid]
        elif is_primary_chromosome(seqid):
            chromosome = normalize_seq_name(seqid)
        else:
            continue

        if not is_primary_chromosome(chromosome):
            continue

        symbol = (
            attributes.get("gene", "").strip()
            or attributes.get("Name", "").strip()
            or gene_records.get(gene_id, {}).get("symbol", "")
        )
        biotype = (
            attributes.get("gene_biotype", "").strip()
            or gene_records.get(gene_id, {}).get("biotype", "")
        )
        record = {
            "gene_id": gene_id,
            "chr": chromosome,
            "start": int(start),
            "end": int(end),
            "strand": strand.strip(),
            "symbol": symbol,
            "biotype": biotype,
            "seqid": seqid,
            "ensembl_ids": list(gene_records.get(gene_id, {}).get("ensembl_ids", [])),
        }
        gene_index[gene_id] = record
        genes_by_chr[chromosome].append(record)

    for chromosome, items in genes_by_chr.items():
        items.sort(key=lambda item: (item["start"], item["end"], item["gene_id"]))
        for idx, item in enumerate(items):
            item["position_index"] = idx

    return genes_by_chr, gene_index


def choose_seed_gene_id(
    target_id: str,
    ensembl_to_gene_ids: Dict[str, List[str]],
    gene_index: Dict[str, dict],
    local_symbol_map: Dict[str, List[str]],
    symbol_to_gene_ids: Dict[str, List[str]],
) -> Tuple[Optional[str], Optional[str]]:
    direct_candidates = [
        gene_id for gene_id in ensembl_to_gene_ids.get(target_id, []) if gene_id in gene_index
    ]
    direct_candidates = sorted(set(direct_candidates))
    if len(direct_candidates) == 1:
        return direct_candidates[0], None
    if len(direct_candidates) > 1:
        return None, f"ambiguous Ensembl mapping ({','.join(direct_candidates)})"

    symbols = local_symbol_map.get(target_id, [])
    symbol_candidates: List[str] = []
    for symbol in symbols:
        for gene_id in symbol_to_gene_ids.get(symbol, []):
            if gene_id in gene_index:
                symbol_candidates.append(gene_id)
    symbol_candidates = sorted(set(symbol_candidates))
    if len(symbol_candidates) == 1:
        return symbol_candidates[0], None
    if len(symbol_candidates) > 1:
        return None, f"ambiguous symbol fallback ({','.join(symbol_candidates)})"
    if symbols:
        return None, f"symbol fallback not found ({','.join(symbols)})"
    return None, "Ensembl ID not found in NCBI gene report"


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def load_matrix_gene_ids(matrix_path: Path) -> set[str]:
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
    with matrix_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return set()
    return {value.strip() for value in header[1:] if value.strip()}


def matching_matrix_ensembl_ids(ensembl_ids: Sequence[str], matrix_gene_ids: set[str]) -> List[str]:
    return [
        ensembl_id
        for ensembl_id in ensembl_ids
        if ensembl_id and ensembl_id in matrix_gene_ids
    ]


def distance_bp(seed: dict, neighbor: dict) -> int:
    if neighbor["end"] < seed["start"]:
        return seed["start"] - neighbor["end"] - 1
    if seed["end"] < neighbor["start"]:
        return neighbor["start"] - seed["end"] - 1
    return 0


def iter_neighbor_rows(seed: dict, genes_by_chr: Dict[str, List[dict]], neighbor_count: int) -> Iterator[dict]:
    chromosome_genes = genes_by_chr[seed["chr"]]
    idx = seed["position_index"]

    upstream = chromosome_genes[max(0, idx - neighbor_count):idx]
    downstream = chromosome_genes[idx + 1: idx + 1 + neighbor_count]

    upstream = list(reversed(upstream))
    for rank, neighbor in enumerate(upstream, start=1):
        yield {
            "neighbor_direction": "upstream",
            "neighbor_rank": rank,
            "neighbor": neighbor,
        }
    for rank, neighbor in enumerate(downstream, start=1):
        yield {
            "neighbor_direction": "downstream",
            "neighbor_rank": rank,
            "neighbor": neighbor,
        }


def iter_matrix_neighbor_rows(
    seed: dict,
    genes_by_chr: Dict[str, List[dict]],
    neighbor_count: int,
    matrix_gene_ids: set[str],
) -> Iterator[dict]:
    counts = {"upstream": 0, "downstream": 0}
    chromosome_genes = genes_by_chr[seed["chr"]]
    for neighbor_info in iter_neighbor_rows(seed, genes_by_chr, len(chromosome_genes)):
        neighbor = neighbor_info["neighbor"]
        matrix_ensembl_ids = matching_matrix_ensembl_ids(neighbor["ensembl_ids"], matrix_gene_ids)
        if not matrix_ensembl_ids:
            continue

        direction = neighbor_info["neighbor_direction"]
        if counts[direction] >= neighbor_count:
            continue

        counts[direction] += 1
        yield {
            **neighbor_info,
            "neighbor_rank": counts[direction],
            "matrix_ensembl_ids": matrix_ensembl_ids,
        }

        if all(count >= neighbor_count for count in counts.values()):
            break


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def gather_input_files(input_root: Path) -> List[Path]:
    return sorted(
        path
        for path in input_root.glob("*_gene_related_disease/*.csv")
        if path.name != "disease_file_grouping_summary.csv"
    )


def process_file(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    genes_by_chr: Dict[str, List[dict]],
    gene_index: Dict[str, dict],
    ensembl_to_gene_ids: Dict[str, List[str]],
    local_symbol_map: Dict[str, List[str]],
    symbol_to_gene_ids: Dict[str, List[str]],
    neighbor_count: int,
    matrix_gene_ids: set[str],
    missing_rows: List[dict],
) -> dict:
    with file_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        disease_ids: List[str] = []
        seed_target_ids: List[str] = []
        for row in reader:
            disease_id = (row.get("diseaseFromSourceMappedId") or "").strip()
            target_id = (row.get("targetId") or "").strip()
            if disease_id:
                disease_ids.append(disease_id)
            if target_id:
                seed_target_ids.append(target_id)

    disease_id = disease_ids[0] if disease_ids else file_path.stem
    seed_target_ids = ordered_unique(seed_target_ids)

    output_rows: List[dict] = []
    resolved_seed_count = 0
    for seed_target_id in seed_target_ids:
        seed_gene_id, reason = choose_seed_gene_id(
            seed_target_id,
            ensembl_to_gene_ids=ensembl_to_gene_ids,
            gene_index=gene_index,
            local_symbol_map=local_symbol_map,
            symbol_to_gene_ids=symbol_to_gene_ids,
        )
        if not seed_gene_id:
            missing_rows.append(
                {
                    "file_name": file_path.name,
                    "relative_input_path": file_path.relative_to(input_root).as_posix(),
                    "diseaseFromSourceMappedId": disease_id,
                    "seed_targetId": seed_target_id,
                    "reason": reason or "unresolved",
                }
            )
            continue

        seed = gene_index[seed_gene_id]
        resolved_seed_count += 1
        for neighbor_info in iter_matrix_neighbor_rows(seed, genes_by_chr, neighbor_count, matrix_gene_ids):
            neighbor = neighbor_info["neighbor"]
            matrix_ensembl_ids = neighbor_info["matrix_ensembl_ids"]
            output_rows.append(
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
                    "distance_bp": distance_bp(seed, neighbor),
                }
            )

    relative_path = file_path.relative_to(input_root)
    output_path = output_root / relative_path
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
    output_row_count = write_csv(output_path, fieldnames, output_rows)

    return {
        "file_name": file_path.name,
        "relative_input_path": relative_path.as_posix(),
        "relative_output_path": output_path.relative_to(output_root).as_posix(),
        "diseaseFromSourceMappedId": disease_id,
        "n_seed_genes": len(seed_target_ids),
        "n_resolved_seed_genes": resolved_seed_count,
        "n_unresolved_seed_genes": len(seed_target_ids) - resolved_seed_count,
        "n_neighbor_rows_written": output_row_count,
    }


def main() -> int:
    args = parse_args()
    if args.neighbors < 1:
        print("--neighbors must be at least 1", file=sys.stderr)
        return 2

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    cache_dir = args.cache_dir.resolve()
    matrix_file = args.matrix_file.resolve()
    if not input_root.exists():
        print(f"Input root does not exist: {input_root}", file=sys.stderr)
        return 2
    if not matrix_file.exists():
        print(f"Matrix file does not exist: {matrix_file}", file=sys.stderr)
        return 2

    headers = build_headers()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    gene_report_name = "ncbi_human_gene_report.jsonl"
    genome_zip_name = f"{args.assembly}_annotation.zip"
    gene_report_path = ensure_gene_report_cache(
        cache_dir=cache_dir,
        filename=gene_report_name,
        headers=headers,
        refresh_cache=args.refresh_cache,
    )
    genome_zip = ensure_zip_download(
        cache_dir=cache_dir,
        filename=genome_zip_name,
        candidate_urls=genome_package_urls(args.assembly, genome_zip_name),
        headers=headers,
        refresh_cache=args.refresh_cache,
    )

    print(f"Using gene report cache: {gene_report_path}")
    print(f"Using genome package: {genome_zip}")
    print(f"Using matrix file: {matrix_file}")

    ensembl_to_gene_ids, gene_records, symbol_to_gene_ids = build_gene_indexes(
        gene_report_path=gene_report_path,
        assembly=args.assembly,
    )
    genes_by_chr, gene_index = build_gff_index(
        genome_zip=genome_zip,
        assembly=args.assembly,
        gene_records=gene_records,
    )
    local_symbol_map = build_local_symbol_map(PROJECT_ROOT / "database" / "combined.tsv")
    matrix_gene_ids = load_matrix_gene_ids(matrix_file)

    print(f"Indexed {len(gene_records)} gene records from NCBI gene report")
    print(f"Indexed {len(gene_index)} primary-chromosome genes from GFF3")
    print(f"Loaded {len(matrix_gene_ids)} matrix gene IDs")

    input_files = gather_input_files(input_root)
    print(f"Found {len(input_files)} disease CSV files")

    summary_rows: List[dict] = []
    missing_rows: List[dict] = []
    for idx, file_path in enumerate(input_files, start=1):
        summary = process_file(
            file_path=file_path,
            input_root=input_root,
            output_root=output_root,
            genes_by_chr=genes_by_chr,
            gene_index=gene_index,
            ensembl_to_gene_ids=ensembl_to_gene_ids,
            local_symbol_map=local_symbol_map,
            symbol_to_gene_ids=symbol_to_gene_ids,
            neighbor_count=args.neighbors,
            matrix_gene_ids=matrix_gene_ids,
            missing_rows=missing_rows,
        )
        summary_rows.append(summary)
        print(
            f"[{idx}/{len(input_files)}] {summary['relative_input_path']} -> "
            f"{summary['n_neighbor_rows_written']} rows "
            f"({summary['n_resolved_seed_genes']}/{summary['n_seed_genes']} seeds resolved)"
        )

    write_csv(
        output_root / "run_summary.csv",
        [
            "file_name",
            "relative_input_path",
            "relative_output_path",
            "diseaseFromSourceMappedId",
            "n_seed_genes",
            "n_resolved_seed_genes",
            "n_unresolved_seed_genes",
            "n_neighbor_rows_written",
        ],
        summary_rows,
    )
    write_csv(
        output_root / "missing_seed_genes.csv",
        [
            "file_name",
            "relative_input_path",
            "diseaseFromSourceMappedId",
            "seed_targetId",
            "reason",
        ],
        missing_rows,
    )

    resolved = sum(row["n_resolved_seed_genes"] for row in summary_rows)
    unresolved = sum(row["n_unresolved_seed_genes"] for row in summary_rows)
    print("Finished.")
    print(f"Resolved seeds: {resolved}")
    print(f"Unresolved seeds: {unresolved}")
    print(f"Summary written to: {output_root / 'run_summary.csv'}")
    print(f"Missing seeds written to: {output_root / 'missing_seed_genes.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
