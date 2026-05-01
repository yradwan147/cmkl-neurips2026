"""Pre-compute multimodal features for CMKL training.

Builds edge_index/edge_type, BiomedBERT text embeddings, and molecular
numeric features from the benchmark data. Saves as .pt files to
data/benchmark/features/ for use by run_cmkl.py, run_nc.py, run_ablations.py.

Memory-safe: streams task files line by line, batches BiomedBERT encoding
with CPU offload, avoids loading large arrays into memory at once.

ID mapping note:
  Task files use x_id/y_id from PrimeKG (DrugBank IDs for drugs, MONDO IDs
  for diseases). entity_to_id maps these strings -> sequential integers.
  SMILES are loaded from data/smiles_cache.json (pre-fetched from PubChem
  via scripts/fetch_smiles.py), matched by lowercased drug name.

Usage:
    # On IBEX (GPU for BiomedBERT):
    python scripts/precompute_features.py --device cuda

    # Local (CPU, slower but works):
    python scripts/precompute_features.py --device cpu

    # Skip text embeddings (if already computed or not needed):
    python scripts/precompute_features.py --skip-text
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import resource
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _log_mem(label: str) -> None:
    """Log current RSS memory usage."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    logger.info(f"[MEM] {label}: {rss_mb:.0f} MB")


# ---------------------------------------------------------------------------
# Step 1a: Build edge_index and edge_type from task triples (streaming)
# ---------------------------------------------------------------------------

def build_edges(
    tasks_dir: Path,
    num_relations: int,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    output_dir: Path,
) -> None:
    """Build edge_index [2, E*2] and edge_type [E*2] from all task triples.

    Streams task files line by line, maps string IDs to integers using
    entity_to_id/relation_to_id, deduplicates edges, adds reverse edges.
    Reverse edges use relation IDs offset by num_relations.

    Args:
        tasks_dir: Path to benchmark/tasks directory.
        num_relations: Number of forward relation types.
        entity_to_id: Entity string -> int mapping.
        relation_to_id: Relation string -> int mapping.
        output_dir: Directory to save edge_index.pt and edge_type.pt.
    """
    _log_mem("before edge construction")

    # Collect unique (head_id, relation_id, tail_id) tuples by streaming files
    edges: set[tuple[int, int, int]] = set()
    task_dirs = sorted(d for d in tasks_dir.iterdir() if d.is_dir())

    for task_dir in task_dirs:
        for split_file in ("train.txt", "valid.txt"):
            path = task_dir / split_file
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        h_str, r_str, t_str = parts
                        if (h_str in entity_to_id
                                and r_str in relation_to_id
                                and t_str in entity_to_id):
                            h = entity_to_id[h_str]
                            r = relation_to_id[r_str]
                            t = entity_to_id[t_str]
                            edges.add((h, r, t))

    logger.info(f"Unique forward edges: {len(edges):,}")
    _log_mem("after dedup")

    # Build forward + reverse edge lists
    heads, tails, rels = [], [], []
    for h, r, t in edges:
        # Forward edge
        heads.append(h)
        tails.append(t)
        rels.append(r)
        # Reverse edge
        heads.append(t)
        tails.append(h)
        rels.append(r + num_relations)

    # Free the set
    del edges

    edge_index = torch.tensor([heads, tails], dtype=torch.long)
    edge_type = torch.tensor(rels, dtype=torch.long)

    del heads, tails, rels

    logger.info(f"edge_index shape: {edge_index.shape}")
    logger.info(f"edge_type shape: {edge_type.shape}")
    logger.info(f"Relation range: [0, {edge_type.max().item()}]")
    _log_mem("after tensor creation")

    torch.save(edge_index, output_dir / "edge_index.pt")
    torch.save(edge_type, output_dir / "edge_type.pt")
    logger.info("Saved edge_index.pt and edge_type.pt")


# ---------------------------------------------------------------------------
# Step 1b: Pre-compute BiomedBERT text embeddings (batched, CPU offload)
# ---------------------------------------------------------------------------

def _build_entity_text_mapping(
    kg_csv_path: Path,
    raw_drug_features_path: Path | None,
    entity_to_id: dict[str, int],
) -> list[tuple[int, str]]:
    """Build (entity_index, text) pairs for text encoding.

    Strategy:
    1. Stream KG CSV to build entity_id -> (name, type) mapping
    2. For drugs: try to get rich descriptions from raw drug features file
       by matching drug names. Falls back to drug name if no match.
    3. For diseases: use disease name from KG CSV
    4. For other types (proteins, pathways, etc.): use node name

    Args:
        kg_csv_path: Path to PrimeKG CSV (e.g. kg_t0.csv).
        raw_drug_features_path: Path to raw drug features file with
            descriptions (e.g. drug_features_t0.csv). Optional.
        entity_to_id: Entity string -> int mapping from task loading.

    Returns:
        List of (entity_index, text_string) pairs for encoding.
    """
    # Step 1: Stream KG CSV to build entity_id -> (name, type)
    entity_info: dict[str, tuple[str, str]] = {}  # entity_id -> (name, type)

    logger.info("Streaming KG CSV to build entity info...")
    with open(kg_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_id = row["x_id"]
            y_id = row["y_id"]
            if x_id in entity_to_id and x_id not in entity_info:
                entity_info[x_id] = (row["x_name"], row["x_type"])
            if y_id in entity_to_id and y_id not in entity_info:
                entity_info[y_id] = (row["y_name"], row["y_type"])

    logger.info(f"Entity info: {len(entity_info):,} entities with names "
                f"(out of {len(entity_to_id):,} total)")

    # Step 2: Build drug name -> rich text mapping from raw drug features
    drug_name_to_text: dict[str, str] = {}
    if raw_drug_features_path and raw_drug_features_path.exists():
        logger.info(f"Loading drug descriptions from {raw_drug_features_path}...")
        with open(raw_drug_features_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract drug name from state column: "DrugName is a solid."
                state = row.get("state", "")
                if " is " in state:
                    drug_name = state.split(" is ")[0].strip().lower()
                else:
                    continue

                # Build rich text from description columns
                text_parts = []
                for col in ["description", "indication", "pharmacodynamics",
                            "mechanism_of_action"]:
                    val = row.get(col, "")
                    if val and val.strip() and val.strip() != '""':
                        # Strip surrounding quotes
                        val = val.strip().strip('"').strip()
                        if len(val) > 5:
                            text_parts.append(val)

                if text_parts:
                    drug_name_to_text[drug_name] = " ".join(text_parts)

        logger.info(f"Drug descriptions: {len(drug_name_to_text):,} drugs with text")

    # Step 3: Build final text list
    # Encode drugs, diseases, AND gene/protein nodes.
    # Following BioMedKG/PrimeKG++ (Ngo et al., 2025) which encodes all
    # entity types. We take a middle ground: drugs + diseases + proteins.
    texts_to_encode: list[tuple[int, str]] = []
    n_rich_drug = 0
    n_name_only = 0
    n_disease = 0
    n_protein = 0

    for entity_id, (name, node_type) in entity_info.items():
        idx = entity_to_id[entity_id]

        if node_type == "drug":
            # Try to get rich description via name matching
            name_lower = name.lower().strip()
            if name_lower in drug_name_to_text:
                texts_to_encode.append((idx, drug_name_to_text[name_lower]))
                n_rich_drug += 1
            else:
                # Fall back to drug name as text
                texts_to_encode.append((idx, f"Drug: {name}"))
                n_name_only += 1

        elif node_type == "disease":
            texts_to_encode.append((idx, f"Disease: {name}"))
            n_disease += 1

        elif node_type == "gene/protein":
            texts_to_encode.append((idx, f"Protein: {name}"))
            n_protein += 1

        # Skip other types (biological_process, anatomy, etc.)
        # — names alone are not meaningful enough for BiomedBERT

    logger.info(f"Text features: {n_rich_drug} drugs with descriptions, "
                f"{n_name_only} drugs with name only, "
                f"{n_disease} diseases, {n_protein} proteins")

    return texts_to_encode


def build_text_embeddings(
    kg_csv_path: Path,
    raw_drug_features_path: Path | None,
    entity_to_id: dict[str, int],
    output_dir: Path,
    num_entities: int,
    device: str = "cpu",
    batch_size: int = 32,
) -> None:
    """Pre-compute BiomedBERT [CLS] embeddings for drug and disease nodes.

    Saves raw 768-dim embeddings (projection to embedding_dim happens
    in TextualEncoder at training time).

    Args:
        kg_csv_path: Path to PrimeKG CSV.
        raw_drug_features_path: Path to raw drug features with descriptions.
        entity_to_id: Entity string -> int mapping.
        output_dir: Directory to save text_embeddings.pt, node_has_text.pt.
        num_entities: Total number of entities.
        device: Device for BiomedBERT encoding.
        batch_size: Encoding batch size.
    """
    _log_mem("before text embedding")

    texts_to_encode = _build_entity_text_mapping(
        kg_csv_path, raw_drug_features_path, entity_to_id
    )

    # Pre-allocate output tensor on CPU
    hidden_size = 768
    text_embeddings = torch.zeros(num_entities, hidden_size, dtype=torch.float32)
    node_has_text = torch.zeros(num_entities, dtype=torch.bool)

    if not texts_to_encode:
        logger.warning("No texts found to encode!")
        torch.save(text_embeddings, output_dir / "text_embeddings.pt")
        torch.save(node_has_text, output_dir / "node_has_text.pt")
        return

    logger.info(f"Total texts to encode: {len(texts_to_encode)}")
    _log_mem("after collecting texts")

    # Load BiomedBERT
    from transformers import AutoModel, AutoTokenizer

    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    _log_mem("after loading BiomedBERT")

    # Encode in batches with CPU offload
    for i in range(0, len(texts_to_encode), batch_size):
        batch = texts_to_encode[i:i + batch_size]
        indices = [idx for idx, _ in batch]
        texts = [text[:2000] for _, text in batch]

        tokens = tokenizer(
            texts, padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu()  # [B, 768]

        # Write directly to pre-allocated tensor
        for j, idx in enumerate(indices):
            text_embeddings[idx] = cls_emb[j]
            node_has_text[idx] = True

        if (i // batch_size) % 100 == 0:
            logger.info(f"  Encoded {i + len(batch)}/{len(texts_to_encode)} texts")

    # Free model
    del model, tokenizer
    if device != "cpu":
        torch.cuda.empty_cache()

    _log_mem("after text encoding")

    logger.info(f"Nodes with text: {node_has_text.sum().item():,}")
    torch.save(text_embeddings, output_dir / "text_embeddings.pt")
    torch.save(node_has_text, output_dir / "node_has_text.pt")
    logger.info("Saved text_embeddings.pt and node_has_text.pt")


# ---------------------------------------------------------------------------
# Step 1c: Molecular features (Morgan fingerprints via RDKit + TDC SMILES)
# ---------------------------------------------------------------------------

def _compute_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 1024) -> list[int] | None:
    """Compute Morgan fingerprint from SMILES string.

    Args:
        smiles: SMILES representation of molecule.
        radius: Morgan fingerprint radius (2 = ECFP4 equivalent).
        n_bits: Number of fingerprint bits.

    Returns:
        List of 0/1 bits, or None if SMILES is invalid.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp)


def build_mol_features(
    kg_csv_path: Path,
    raw_drug_features_path: Path | None,
    entity_to_id: dict[str, int],
    output_dir: Path,
    num_entities: int,
) -> None:
    """Build molecular feature tensor using Morgan fingerprints from RDKit.

    Strategy:
    1. Load SMILES from PubChem cache (data/smiles_cache.json, pre-fetched
       via scripts/fetch_smiles.py)
    2. Match drug names to our DrugBank entity IDs via KG CSV name matching
    3. Compute 1024-bit Morgan fingerprints (ECFP4) via RDKit
    4. Fallback: use 3-scalar numeric features (MW, TPSA, CLogP) from raw
       drug features file for drugs without SMILES

    The project guide (Section 4.3.6) explicitly specifies RDKit Morgan FPs
    with nBits=1024.

    Args:
        kg_csv_path: Path to PrimeKG CSV.
        raw_drug_features_path: Path to raw drug features with numeric columns.
        entity_to_id: Entity string -> int mapping.
        output_dir: Directory to save mol_features.pt, node_has_mol.pt.
        num_entities: Total number of entities.
    """
    _log_mem("before mol features")

    morgan_bits = 1024
    mol_dim = morgan_bits

    # Step 1: Build DrugBank_ID -> drug_name from KG CSV
    logger.info("Building DrugBank ID -> drug name mapping from KG...")
    dbid_to_name: dict[str, str] = {}
    with open(kg_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["x_type"] == "drug":
                dbid_to_name[row["x_id"]] = row["x_name"].strip().lower()
            if row["y_type"] == "drug":
                dbid_to_name[row["y_id"]] = row["y_name"].strip().lower()

    name_to_dbid: dict[str, str] = {}
    for dbid, name in dbid_to_name.items():
        if name not in name_to_dbid:
            name_to_dbid[name] = dbid

    logger.info(f"DrugBank drugs: {len(dbid_to_name)}, unique names: {len(name_to_dbid)}")

    # Step 2: Load SMILES from cache (pre-fetched via scripts/fetch_smiles.py)
    smiles_map: dict[str, str] = {}  # drug_name_lower -> SMILES
    smiles_cache_path = Path("data/smiles_cache.json")
    if smiles_cache_path.exists():
        import json as _json
        cache_data = _json.loads(smiles_cache_path.read_text())
        # Match cache keys (drug_name_lower) to our drug names
        for drug_name_lower in name_to_dbid:
            if drug_name_lower in cache_data:
                smiles_map[drug_name_lower] = cache_data[drug_name_lower]
        logger.info(f"SMILES from cache: {len(smiles_map)} drugs "
                    f"(cache has {len(cache_data)} total entries)")
    else:
        logger.warning("No SMILES cache found at data/smiles_cache.json")
        logger.info("Run 'python scripts/fetch_smiles.py' first to fetch SMILES from PubChem")

    # Step 3: Compute Morgan FPs for matched drugs
    mol_features = torch.zeros(num_entities, mol_dim, dtype=torch.float32)
    node_has_mol = torch.zeros(num_entities, dtype=torch.bool)
    n_morgan = 0
    n_failed_smiles = 0

    if smiles_map:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")  # suppress RDKit warnings

        for drug_name, smiles in smiles_map.items():
            if drug_name not in name_to_dbid:
                continue
            dbid = name_to_dbid[drug_name]
            if dbid not in entity_to_id:
                continue
            idx = entity_to_id[dbid]

            fp = _compute_morgan_fp(smiles, radius=2, n_bits=morgan_bits)
            if fp is not None:
                mol_features[idx] = torch.tensor(fp, dtype=torch.float32)
                node_has_mol[idx] = True
                n_morgan += 1
            else:
                n_failed_smiles += 1

        logger.info(f"Morgan FPs: {n_morgan} drugs computed, "
                    f"{n_failed_smiles} failed SMILES parsing")

    # Step 4: Fallback — use 3-scalar numeric features for unmatched drugs
    n_fallback = 0
    if raw_drug_features_path and raw_drug_features_path.exists():
        mol_cols = ["molecular_weight", "tpsa", "clogp"]
        with open(raw_drug_features_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                state = row.get("state", "")
                if " is " not in state:
                    continue
                drug_name = state.split(" is ")[0].strip().lower()

                if drug_name not in name_to_dbid:
                    continue
                dbid = name_to_dbid[drug_name]
                if dbid not in entity_to_id:
                    continue
                idx = entity_to_id[dbid]

                # Skip if already has Morgan FP
                if node_has_mol[idx]:
                    continue

                vals = []
                has_any = False
                for col in mol_cols:
                    v = row.get(col, "")
                    if v and v.strip() and v.strip() != '""':
                        numbers = re.findall(r"[-+]?\d*\.?\d+", str(v))
                        if numbers:
                            vals.append(float(numbers[0]))
                            has_any = True
                        else:
                            vals.append(0.0)
                    else:
                        vals.append(0.0)

                if has_any:
                    # Pad 3 scalars to mol_dim with zeros
                    padded = vals + [0.0] * (mol_dim - len(vals))
                    mol_features[idx] = torch.tensor(padded, dtype=torch.float32)
                    node_has_mol[idx] = True
                    n_fallback += 1

        if n_fallback > 0:
            logger.info(f"Fallback numeric features: {n_fallback} additional drugs "
                        f"(padded to {mol_dim}-dim)")

    total_mol = node_has_mol.sum().item()
    logger.info(f"Total mol features: {total_mol} drugs "
                f"({n_morgan} Morgan + {n_fallback} fallback)")

    torch.save(mol_features, output_dir / "mol_features.pt")
    torch.save(node_has_mol, output_dir / "node_has_mol.pt")

    with open(output_dir / "mol_dim.txt", "w") as f:
        f.write(str(mol_dim))

    logger.info("Saved mol_features.pt, node_has_mol.pt, mol_dim.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute features for CMKL")
    parser.add_argument("--tasks-dir", default="data/benchmark/tasks",
                        help="Path to benchmark tasks directory")
    parser.add_argument("--kg-csv", default="data/benchmark/snapshots/kg_t0.csv",
                        help="Path to PrimeKG CSV")
    parser.add_argument("--drug-features",
                        default="data/benchmark/snapshots/drug_features_t0.csv",
                        help="Path to raw drug features CSV")
    parser.add_argument("--output-dir", default="data/benchmark/features",
                        help="Output directory for .pt files")
    parser.add_argument("--device", default="cpu",
                        help="Device for BiomedBERT encoding (cpu or cuda)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for text encoding")
    parser.add_argument("--skip-text", action="store_true",
                        help="Skip text embedding computation")
    parser.add_argument("--skip-edges", action="store_true",
                        help="Skip edge construction")
    parser.add_argument("--skip-mol", action="store_true",
                        help="Skip molecular feature computation")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    kg_csv_path = Path(args.kg_csv)
    drug_features_path = Path(args.drug_features)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get entity/relation mappings from task files
    from src.baselines._base import load_task_sequence
    logger.info("Scanning vocabulary to get entity/relation mappings...")
    task_seq, entity_to_id, relation_to_id = load_task_sequence(tasks_dir)
    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)
    logger.info(f"Entities: {num_entities:,}, Relations: {num_relations}")

    # Free task data — we only need the mappings
    del task_seq

    # Save vocab sizes so run scripts can use them even with task subsets
    import json
    with open(output_dir / "vocab_sizes.json", "w") as f:
        json.dump({"num_entities": num_entities, "num_relations": num_relations}, f)
    logger.info(f"Saved vocab_sizes.json (entities={num_entities}, relations={num_relations})")

    _log_mem("after vocab scan")

    # Build edges
    if not args.skip_edges:
        logger.info("\n=== Building edge_index and edge_type ===")
        build_edges(tasks_dir, num_relations, entity_to_id, relation_to_id,
                    output_dir)
    else:
        logger.info("Skipping edge construction")

    # Build text embeddings
    if not args.skip_text:
        logger.info("\n=== Building BiomedBERT text embeddings ===")
        build_text_embeddings(
            kg_csv_path,
            drug_features_path if drug_features_path.exists() else None,
            entity_to_id, output_dir, num_entities,
            device=args.device, batch_size=args.batch_size,
        )
    else:
        logger.info("Skipping text embeddings")

    # Build mol features
    if not args.skip_mol:
        logger.info("\n=== Building molecular features ===")
        build_mol_features(
            kg_csv_path,
            drug_features_path if drug_features_path.exists() else None,
            entity_to_id, output_dir, num_entities,
        )
    else:
        logger.info("Skipping molecular features")

    _log_mem("final")
    logger.info("\nDone! Feature files saved to: %s", output_dir)


if __name__ == "__main__":
    main()
