"""Multimodal feature extraction for PrimeKG nodes.

Extracts three types of features for knowledge graph nodes:
1. Textual: Clinical descriptions from DrugBank (drugs) and Mayo Clinic/Orphanet (diseases)
2. Molecular: Morgan fingerprints from SMILES strings (drug nodes only)
3. Structural: Derived from graph topology via GNN encoders (computed at training time)

Drug features include: description, indication, pharmacodynamics, mechanism of action,
toxicity, protein binding, metabolism, half-life, route of elimination.
Disease features include: MONDO definition, UMLS description, Mayo Clinic clinical features.

Usage:
    from src.data.features import extract_multimodal_features
    drug_feat, disease_feat = extract_multimodal_features(kg_path, drug_path, disease_path)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Text columns to concatenate for drug descriptions
DRUG_TEXT_COLUMNS = [
    "description", "indication", "pharmacodynamics",
    "mechanism_of_action",
]

# Disease feature text columns (from PrimeKG feature files)
DISEASE_TEXT_COLUMNS = [
    "mondo_definition", "umls_description", "mayo_clinical_features",
]


def extract_multimodal_features(
    kg_path: str,
    drug_features_path: str,
    disease_features_path: str | None = None,
    output_dir: str = "data/benchmark/features",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract text and structural features for multimodal nodes.

    Args:
        kg_path: Path to PrimeKG CSV.
        drug_features_path: Path to drug features CSV from PrimeKG/TDC.
        disease_features_path: Path to disease features CSV (optional,
            extracted from PrimeKG feature construction scripts).
        output_dir: Directory to save processed features.

    Returns:
        Tuple of (drug_features_df, disease_features_df).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    kg = pd.read_csv(kg_path, low_memory=False)

    # Process drug features
    drug_feat = _process_drug_features(drug_features_path, kg)
    drug_feat.to_csv(out / "drug_features.csv", index=False)
    logger.info(f"Drug features: {len(drug_feat)} drugs, columns={list(drug_feat.columns)}")

    # Process disease features
    disease_feat = _process_disease_features(disease_features_path, kg)
    disease_feat.to_csv(out / "disease_features.csv", index=False)
    logger.info(f"Disease features: {len(disease_feat)} diseases, columns={list(disease_feat.columns)}")

    # Create node index mapping (all node types)
    node_map = build_node_index_map(kg)
    node_map.to_csv(out / "node_index_map.csv", index=False)
    logger.info(f"Node index map: {len(node_map)} nodes")

    # Create modality masks
    masks = get_node_modality_masks(kg, drug_feat, disease_feat)
    logger.info(
        f"Modality coverage: {masks['has_text'].sum():,} nodes with text, "
        f"{masks['has_mol'].sum():,} nodes with molecular features"
    )

    return drug_feat, disease_feat


def _process_drug_features(
    drug_features_path: str,
    kg: pd.DataFrame,
) -> pd.DataFrame:
    """Process raw drug features into a unified format.

    Concatenates text columns into a single 'text' field and extracts
    numeric features (molecular_weight, tpsa, clogp).

    Args:
        drug_features_path: Path to drug features CSV.
        kg: PrimeKG DataFrame (for mapping drug IDs).

    Returns:
        DataFrame with drug_id, text, molecular_weight, tpsa, clogp columns.
    """
    raw = pd.read_csv(drug_features_path, low_memory=False)

    # Build unified text description
    text_parts = []
    for col in DRUG_TEXT_COLUMNS:
        if col in raw.columns:
            text_parts.append(raw[col].fillna(""))

    if text_parts:
        raw["text"] = pd.concat(text_parts, axis=1).apply(
            lambda row: " ".join(s for s in row if s.strip()), axis=1
        )
    else:
        raw["text"] = ""

    # Select output columns
    out_cols = ["node_index", "text"]
    for col in ["molecular_weight", "tpsa", "clogp", "group", "state"]:
        if col in raw.columns:
            out_cols.append(col)

    result = raw[out_cols].copy()
    result = result.rename(columns={"node_index": "drug_id"})
    result["drug_id"] = result["drug_id"].astype(str)

    # Mark which drugs have meaningful text
    result["has_text"] = result["text"].str.len() > 10

    logger.info(
        f"Drug features: {result['has_text'].sum()}/{len(result)} have text descriptions"
    )

    return result


def _process_disease_features(
    disease_features_path: str | None,
    kg: pd.DataFrame,
) -> pd.DataFrame:
    """Process disease features.

    If no disease features file is provided, creates a minimal DataFrame
    from disease nodes in the KG with names as text.

    Args:
        disease_features_path: Path to disease features CSV, or None.
        kg: PrimeKG DataFrame.

    Returns:
        DataFrame with disease_id, text, has_text columns.
    """
    if disease_features_path and Path(disease_features_path).exists():
        raw = pd.read_csv(disease_features_path, low_memory=False)

        text_parts = []
        for col in DISEASE_TEXT_COLUMNS:
            if col in raw.columns:
                text_parts.append(raw[col].fillna(""))

        if text_parts:
            raw["text"] = pd.concat(text_parts, axis=1).apply(
                lambda row: " ".join(s for s in row if s.strip()), axis=1
            )
        else:
            raw["text"] = ""

        # Identify the ID column
        id_col = None
        for candidate in ["node_index", "disease_id", "mondo_id"]:
            if candidate in raw.columns:
                id_col = candidate
                break

        if id_col is None:
            id_col = raw.columns[0]

        result = raw[[id_col, "text"]].copy()
        result = result.rename(columns={id_col: "disease_id"})

    else:
        # Fallback: extract disease nodes from KG and use names as text
        logger.info("No disease features file provided, using disease names from KG")
        diseases = pd.concat([
            kg[kg["x_type"] == "disease"][["x_id", "x_name"]].rename(
                columns={"x_id": "disease_id", "x_name": "text"}
            ),
            kg[kg["y_type"] == "disease"][["y_id", "y_name"]].rename(
                columns={"y_id": "disease_id", "y_name": "text"}
            ),
        ]).drop_duplicates("disease_id").reset_index(drop=True)

        result = diseases.copy()

    result["disease_id"] = result["disease_id"].astype(str)
    result["has_text"] = result["text"].str.len() > 10

    logger.info(
        f"Disease features: {result['has_text'].sum()}/{len(result)} have text"
    )

    return result


def build_node_index_map(kg: pd.DataFrame) -> pd.DataFrame:
    """Build a unified node-to-integer-index mapping for all nodes.

    The TDC version of PrimeKG doesn't include x_index/y_index columns,
    so we create our own consistent integer mapping.

    Args:
        kg: PrimeKG DataFrame.

    Returns:
        DataFrame with columns: node_id, node_type, node_name, node_index.
    """
    x_nodes = kg[["x_id", "x_type", "x_name"]].rename(
        columns={"x_id": "node_id", "x_type": "node_type", "x_name": "node_name"}
    )
    y_nodes = kg[["y_id", "y_type", "y_name"]].rename(
        columns={"y_id": "node_id", "y_type": "node_type", "y_name": "node_name"}
    )

    all_nodes = pd.concat([x_nodes, y_nodes]).drop_duplicates(
        subset=["node_id", "node_type"]
    ).reset_index(drop=True)

    # Sort by type then ID for deterministic ordering
    all_nodes = all_nodes.sort_values(
        ["node_type", "node_id"]
    ).reset_index(drop=True)

    all_nodes["node_index"] = range(len(all_nodes))
    all_nodes["node_id"] = all_nodes["node_id"].astype(str)

    return all_nodes


def compute_morgan_fingerprints(
    smiles_series: pd.Series,
    radius: int = 2,
    n_bits: int = 1024,
) -> np.ndarray:
    """Compute Morgan fingerprints from SMILES strings using RDKit.

    Args:
        smiles_series: Pandas Series of SMILES strings.
        radius: Morgan fingerprint radius.
        n_bits: Number of bits in the fingerprint.

    Returns:
        NumPy array of shape (n_molecules, n_bits).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    for smiles in smiles_series:
        if pd.isna(smiles) or not smiles.strip():
            fps.append(np.zeros(n_bits, dtype=np.float32))
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(np.array(fp, dtype=np.float32))

    return np.stack(fps)


def compute_text_embeddings(
    texts: list[str],
    model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    batch_size: int = 32,
    device: str = "cpu",
) -> "torch.Tensor":
    """Compute text embeddings using a pretrained language model.

    This is computationally expensive and should be run on GPU (IBEX).
    For local development, use a smaller model or cache results.

    Args:
        texts: List of text descriptions.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for encoding.
        device: Device to run on ('cpu', 'cuda', 'mps').

    Returns:
        Tensor of shape (n_texts, hidden_dim).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Replace empty strings with placeholder
        batch = [t if t.strip() else "unknown" for t in batch]

        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def get_node_modality_masks(
    kg: pd.DataFrame,
    drug_features: pd.DataFrame,
    disease_features: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Determine which nodes have which modality features.

    Not all nodes have all modalities: only drugs have molecular features,
    only drugs/diseases have textual descriptions.

    Args:
        kg: PrimeKG DataFrame.
        drug_features: Drug features DataFrame.
        disease_features: Disease features DataFrame.

    Returns:
        Dict with 'has_text' and 'has_mol' boolean arrays,
        plus 'node_ids' for indexing.
    """
    node_map = build_node_index_map(kg)
    n_nodes = len(node_map)

    has_text = np.zeros(n_nodes, dtype=bool)
    has_mol = np.zeros(n_nodes, dtype=bool)

    # Drug IDs with text
    drug_id_col = "drug_id" if "drug_id" in drug_features.columns else drug_features.columns[0]
    drug_ids_with_text = set(
        drug_features[drug_features.get("has_text", pd.Series([True] * len(drug_features)))][drug_id_col].astype(str)
    )

    # Disease IDs with text
    disease_id_col = "disease_id" if "disease_id" in disease_features.columns else disease_features.columns[0]
    disease_ids_with_text = set(
        disease_features[disease_features.get("has_text", pd.Series([True] * len(disease_features)))][disease_id_col].astype(str)
    )

    # All drug IDs have potential molecular features
    all_drug_ids = set(drug_features[drug_id_col].astype(str))

    for idx, row in node_map.iterrows():
        nid = str(row["node_id"])
        if nid in drug_ids_with_text or nid in disease_ids_with_text:
            has_text[idx] = True
        if nid in all_drug_ids:
            has_mol[idx] = True

    return {
        "has_text": has_text,
        "has_mol": has_mol,
        "node_ids": node_map["node_id"].values,
        "node_types": node_map["node_type"].values,
    }
