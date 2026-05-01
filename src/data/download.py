"""PrimeKG download utilities for temporal snapshots.

Downloads PrimeKG knowledge graph data from multiple sources:
- t0 (June 2021): Original PrimeKG from Harvard Dataverse or TDC
- t1 (July 2023): Rebuilt from updated source databases
- t2 (Feb 2026): PrimeKG-U from CellAwareGNN (optional)

Expected KG format: DataFrame with columns [relation, display_relation,
x_index, x_id, x_type, x_name, x_source, y_index, y_id, y_type, y_name, y_source].

Usage:
    from src.data.download import download_primekg_t0
    kg_t0 = download_primekg_t0(save_dir='data/benchmark/snapshots')
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Expected columns in PrimeKG
PRIMEKG_COLUMNS = [
    "relation", "display_relation",
    "x_index", "x_id", "x_type", "x_name", "x_source",
    "y_index", "y_id", "y_type", "y_name", "y_source",
]

# Expected node types in PrimeKG t0
PRIMEKG_NODE_TYPES = [
    "gene/protein", "drug", "effect/phenotype", "disease",
    "biological_process", "molecular_function", "cellular_component",
    "exposure", "pathway", "anatomy",
]

# Harvard Dataverse direct download URL for kg.csv
DATAVERSE_KG_URL = (
    "https://dataverse.harvard.edu/api/access/datafile/6180620"
)


def download_primekg_t0(
    save_dir: str = "data/benchmark/snapshots",
    method: str = "tdc",
) -> pd.DataFrame:
    """Download original PrimeKG snapshot (June 2021).

    Args:
        save_dir: Directory to save the downloaded kg.csv.
        method: Download method - 'tdc' (preferred) or 'dataverse'.

    Returns:
        DataFrame with ~8.1M rows and 12 columns.

    Raises:
        RuntimeError: If download fails from all sources.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    output_file = save_path / "kg_t0.csv"

    # If already downloaded, just load it
    if output_file.exists():
        logger.info(f"PrimeKG t0 already exists at {output_file}, loading...")
        return load_primekg(str(output_file))

    if method == "tdc":
        try:
            kg = download_primekg_tdc(str(save_path))
        except Exception as e:
            logger.warning(f"TDC download failed: {e}. Falling back to Dataverse.")
            kg = download_primekg_dataverse(str(save_path))
    elif method == "dataverse":
        kg = download_primekg_dataverse(str(save_path))
    else:
        raise ValueError(f"Unknown download method: {method}. Use 'tdc' or 'dataverse'.")

    # Save to standard location
    kg.to_csv(output_file, index=False)
    logger.info(f"Saved PrimeKG t0 to {output_file} ({len(kg):,} rows)")

    return kg


def download_primekg_tdc(save_dir: str) -> pd.DataFrame:
    """Download PrimeKG via Therapeutics Data Commons (TDC).

    Uses tdc.resource.PrimeKG to download the knowledge graph
    and drug features.

    Args:
        save_dir: Directory to save output files.

    Returns:
        PrimeKG DataFrame.
    """
    from tdc.resource import PrimeKG

    logger.info("Downloading PrimeKG via TDC...")
    data = PrimeKG(path=save_dir)
    kg = data.get_data()

    logger.info(f"TDC download complete: {kg.shape[0]:,} rows, {kg.shape[1]} columns")

    # Also get drug features if available
    try:
        drug_features = data.get_features(feature_type="drug")
        features_path = Path(save_dir) / "drug_features_t0.csv"
        drug_features.to_csv(features_path, index=False)
        logger.info(f"Saved drug features to {features_path}")
    except Exception as e:
        logger.warning(f"Could not get drug features: {e}")

    return kg


def download_primekg_dataverse(save_dir: str) -> pd.DataFrame:
    """Download PrimeKG directly from Harvard Dataverse.

    Downloads kg.csv from https://doi.org/10.7910/DVN/IXA7BM.

    Args:
        save_dir: Directory to save output files.

    Returns:
        PrimeKG DataFrame.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    raw_file = save_path / "kg_raw_dataverse.csv"

    logger.info(f"Downloading PrimeKG from Harvard Dataverse to {raw_file}...")

    def _progress_hook(count, block_size, total_size):
        percent = count * block_size * 100 / total_size if total_size > 0 else 0
        if int(percent) % 10 == 0:
            logger.info(f"Download progress: {percent:.0f}%")

    urllib.request.urlretrieve(DATAVERSE_KG_URL, str(raw_file), reporthook=_progress_hook)
    logger.info("Download complete. Loading into DataFrame...")

    kg = load_primekg(str(raw_file))
    return kg


def load_primekg(path: str, chunksize: Optional[int] = None) -> pd.DataFrame:
    """Load a PrimeKG CSV file, optionally using chunked reading.

    Args:
        path: Path to kg.csv file.
        chunksize: If set, read in chunks of this size (for large files).

    Returns:
        PrimeKG DataFrame.
    """
    logger.info(f"Loading PrimeKG from {path}...")

    if chunksize:
        chunks = []
        for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):
            chunks.append(chunk)
        kg = pd.concat(chunks, ignore_index=True)
    else:
        kg = pd.read_csv(path, low_memory=False)

    logger.info(f"Loaded PrimeKG: {kg.shape[0]:,} rows, {kg.shape[1]} columns")
    return kg


def verify_primekg(kg: pd.DataFrame, expected_snapshot: str = "t0") -> dict:
    """Verify PrimeKG data integrity.

    Checks row count, column names, node types, and edge types
    against expected values for the given snapshot.

    Args:
        kg: PrimeKG DataFrame.
        expected_snapshot: One of 't0', 't1', 't2'.

    Returns:
        Dictionary with verification results (counts, types, status).
    """
    results = {
        "snapshot": expected_snapshot,
        "num_rows": len(kg),
        "num_columns": len(kg.columns),
        "columns": list(kg.columns),
        "issues": [],
    }

    # Check columns
    missing_cols = set(PRIMEKG_COLUMNS) - set(kg.columns)
    if missing_cols:
        results["issues"].append(f"Missing columns: {missing_cols}")

    # Count unique node types (from both x_type and y_type)
    if "x_type" in kg.columns and "y_type" in kg.columns:
        node_types = set(kg["x_type"].unique()) | set(kg["y_type"].unique())
        results["node_types"] = sorted(node_types)
        results["num_node_types"] = len(node_types)

    # Count unique relation types
    if "relation" in kg.columns:
        results["num_relation_types"] = kg["relation"].nunique()
        results["relation_types"] = sorted(kg["relation"].unique().tolist())

    # Count unique nodes
    if "x_index" in kg.columns and "y_index" in kg.columns:
        all_nodes = set(kg["x_index"].unique()) | set(kg["y_index"].unique())
        results["num_unique_nodes"] = len(all_nodes)

    # Node type counts
    if "x_type" in kg.columns:
        x_counts = kg["x_type"].value_counts().to_dict()
        y_counts = kg["y_type"].value_counts().to_dict()
        results["x_type_counts"] = x_counts
        results["y_type_counts"] = y_counts

    # Snapshot-specific checks
    if expected_snapshot == "t0":
        if len(kg) < 8_000_000:
            results["issues"].append(
                f"Expected ~8.1M rows for t0, got {len(kg):,}"
            )
        if results.get("num_node_types", 0) != 10:
            results["issues"].append(
                f"Expected 10 node types for t0, got {results.get('num_node_types', '?')}"
            )

    results["status"] = "OK" if not results["issues"] else "ISSUES_FOUND"
    return results
