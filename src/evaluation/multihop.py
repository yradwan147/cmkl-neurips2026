"""Multi-hop path extraction and evaluation for biomedical KGs.

Extracts biologically meaningful 2-hop paths from the KG and evaluates
whether models can predict multi-hop targets — demonstrating the advantage
of graph structure (R-GCN message passing) over flat approaches (RAG).

Usage:
    from src.evaluation.multihop import (
        BIOMEDICAL_PATH_TYPES,
        build_adjacency_by_relation,
        extract_multihop_paths,
        evaluate_multihop,
    )
    adj = build_adjacency_by_relation(train_triples, relation_to_id)
    paths = extract_multihop_paths(adj, rel1_id, rel2_id, direct_pairs)
    metrics = evaluate_multihop(score_fn, paths, num_entities)
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Biomedically meaningful 2-hop path patterns
# Format: (relation_1, relation_2, description)
# Path: source --[rel1]--> intermediate --[rel2]--> target
# ---------------------------------------------------------------------------
BIOMEDICAL_PATH_TYPES: list[tuple[str, str, str]] = [
    # Drug repurposing via shared protein targets
    ("drug_protein", "disease_protein", "drug -> protein -> disease"),
    # Drug mechanism via pathway
    ("drug_protein", "pathway_protein", "drug -> protein -> pathway"),
    # Drug mechanism via biological process
    ("drug_protein", "bioprocess_protein", "drug -> protein -> bioprocess"),
    # Disease mechanism via protein-pathway link
    ("disease_protein", "pathway_protein", "disease -> protein -> pathway"),
    # Disease mechanism via biological process
    ("disease_protein", "bioprocess_protein", "disease -> protein -> bioprocess"),
    # Drug interaction chain (drug -> protein -> protein)
    ("drug_protein", "protein_protein", "drug -> protein -> protein"),
    # Disease interaction chain (disease -> protein -> protein)
    ("disease_protein", "protein_protein", "disease -> protein -> protein"),
]

# Multi-hop question templates for RAG evaluation
# Maps (rel1, rel2) -> question template with {source}, {intermediate}, {target}
MULTIHOP_QUESTION_TEMPLATES: dict[tuple[str, str], str] = {
    ("drug_protein", "disease_protein"): (
        "Drug {source} targets protein {intermediate}. "
        "What disease might be treated by {source}?"
    ),
    ("drug_protein", "pathway_protein"): (
        "Drug {source} targets protein {intermediate}. "
        "What pathway involves this protein?"
    ),
    ("drug_protein", "bioprocess_protein"): (
        "Drug {source} targets protein {intermediate}. "
        "What biological process involves this protein?"
    ),
    ("disease_protein", "pathway_protein"): (
        "Disease {source} involves protein {intermediate}. "
        "What pathway does this protein participate in?"
    ),
    ("disease_protein", "bioprocess_protein"): (
        "Disease {source} involves protein {intermediate}. "
        "What biological process does this protein participate in?"
    ),
    ("drug_protein", "protein_protein"): (
        "Drug {source} targets protein {intermediate}. "
        "What other protein interacts with {intermediate}?"
    ),
    ("disease_protein", "protein_protein"): (
        "Disease {source} involves protein {intermediate}. "
        "What other protein interacts with {intermediate}?"
    ),
}


# ---------------------------------------------------------------------------
# Adjacency construction
# ---------------------------------------------------------------------------

def build_adjacency_by_relation(
    triples: np.ndarray,
    relation_to_id: dict[str, int],
) -> dict[int, dict[int, set[int]]]:
    """Build per-relation adjacency lists from int triples.

    Args:
        triples: [N, 3] int64 array with (head_id, rel_id, tail_id).
        relation_to_id: Relation string -> int mapping (for logging only).

    Returns:
        Dict mapping rel_id -> {head_id -> set(tail_ids)}.
    """
    adj: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))

    for i in range(len(triples)):
        h, r, t = int(triples[i, 0]), int(triples[i, 1]), int(triples[i, 2])
        adj[r][h].add(t)

    id_to_rel = {v: k for k, v in relation_to_id.items()}
    for rel_id, heads in adj.items():
        rel_name = id_to_rel.get(rel_id, str(rel_id))
        n_edges = sum(len(tails) for tails in heads.values())
        logger.debug(f"  Relation '{rel_name}' (id={rel_id}): "
                     f"{len(heads)} sources, {n_edges} edges")

    return dict(adj)


def build_direct_pair_set(triples: np.ndarray) -> set[tuple[int, int]]:
    """Build set of all directly-connected (head, tail) entity pairs.

    Used to filter multi-hop paths where source and target also have
    a direct edge, which would make prediction trivially easy.

    Args:
        triples: [N, 3] int64 array.

    Returns:
        Set of (head_id, tail_id) tuples.
    """
    pairs: set[tuple[int, int]] = set()
    for i in range(len(triples)):
        pairs.add((int(triples[i, 0]), int(triples[i, 2])))
    return pairs


# ---------------------------------------------------------------------------
# Path extraction
# ---------------------------------------------------------------------------

def extract_multihop_paths(
    adj_by_rel: dict[int, dict[int, set[int]]],
    rel1_id: int,
    rel2_id: int,
    direct_pairs: set[tuple[int, int]] | None = None,
    max_paths: int = 10000,
    seed: int = 42,
) -> list[tuple[int, int, int, int, int]]:
    """Extract 2-hop paths: source --[rel1]--> mid --[rel2]--> target.

    Only keeps (source, target) pairs that are NOT directly connected
    (to test genuine multi-hop reasoning, not shortcut memorization).

    Args:
        adj_by_rel: Per-relation adjacency from build_adjacency_by_relation.
        rel1_id: First hop relation ID.
        rel2_id: Second hop relation ID.
        direct_pairs: Directly-connected pairs to exclude. If None, no filter.
        max_paths: Maximum paths to return (samples if more found).
        seed: Random seed for sampling.

    Returns:
        List of (source_id, rel1_id, mid_id, rel2_id, target_id) tuples.
    """
    adj1 = adj_by_rel.get(rel1_id, {})
    adj2 = adj_by_rel.get(rel2_id, {})

    if not adj1 or not adj2:
        return []

    paths: list[tuple[int, int, int, int, int]] = []

    for source, intermediates in adj1.items():
        for mid in intermediates:
            if mid not in adj2:
                continue
            for target in adj2[mid]:
                if target == source:
                    continue  # skip self-loops
                if direct_pairs and (source, target) in direct_pairs:
                    continue  # skip if directly connected
                paths.append((source, rel1_id, mid, rel2_id, target))

    if len(paths) > max_paths:
        rng = random.Random(seed)
        paths = rng.sample(paths, max_paths)

    return paths


def extract_all_path_types(
    triples: np.ndarray,
    relation_to_id: dict[str, int],
    max_paths_per_type: int = 10000,
    seed: int = 42,
) -> dict[str, list[tuple[int, int, int, int, int]]]:
    """Extract multi-hop paths for all biomedical path types.

    Convenience function that builds adjacency, computes direct pairs,
    and extracts paths for each type in BIOMEDICAL_PATH_TYPES.

    Args:
        triples: [N, 3] int64 training triples.
        relation_to_id: Relation name -> int mapping.
        max_paths_per_type: Max paths per path type.
        seed: Random seed.

    Returns:
        Dict mapping path description -> list of path tuples.
    """
    logger.info(f"Building adjacency from {len(triples):,} triples...")
    adj = build_adjacency_by_relation(triples, relation_to_id)
    direct_pairs = build_direct_pair_set(triples)
    logger.info(f"Direct pairs: {len(direct_pairs):,}")

    all_paths: dict[str, list[tuple[int, int, int, int, int]]] = {}

    for rel1_name, rel2_name, description in BIOMEDICAL_PATH_TYPES:
        rel1_id = relation_to_id.get(rel1_name)
        rel2_id = relation_to_id.get(rel2_name)

        if rel1_id is None or rel2_id is None:
            logger.warning(f"Skipping '{description}': "
                           f"rel1='{rel1_name}' (id={rel1_id}), "
                           f"rel2='{rel2_name}' (id={rel2_id})")
            continue

        paths = extract_multihop_paths(
            adj, rel1_id, rel2_id, direct_pairs,
            max_paths=max_paths_per_type, seed=seed,
        )
        all_paths[description] = paths
        logger.info(f"  {description}: {len(paths):,} paths")

    return all_paths


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_multihop(
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    paths: list[tuple[int, int, int, int, int]],
    num_entities: int,
    batch_size: int = 256,
) -> dict[str, float]:
    """Evaluate multi-hop prediction: given source, can model predict target?

    For each path (src, r1, mid, r2, tgt), we query score_fn(src, r2)
    to get scores for all entities, then rank tgt. This tests whether
    the model's embeddings capture 2-hop structural information.

    R-GCN (2 layers) propagates 2-hop neighborhood info into embeddings,
    so src's embedding should encode information about 2-hop neighbors.
    Flat KGE methods and RAG don't have this structural encoding.

    Args:
        score_fn: Takes (head_ids [B], rel_ids [B]) and returns
            scores [B, num_entities] for all tail entities.
        paths: Multi-hop paths from extract_multihop_paths.
        num_entities: Total entity count for ranking.
        batch_size: Evaluation batch size.

    Returns:
        Dict with multihop_MRR, multihop_Hits@1/3/10, num_paths.
    """
    if not paths:
        return {"multihop_MRR": 0.0, "multihop_Hits@1": 0.0,
                "multihop_Hits@3": 0.0, "multihop_Hits@10": 0.0,
                "num_paths": 0}

    sources = np.array([p[0] for p in paths], dtype=np.int64)
    rel2s = np.array([p[3] for p in paths], dtype=np.int64)
    targets = np.array([p[4] for p in paths], dtype=np.int64)

    all_ranks = []

    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        batch_src = sources[start:end]
        batch_rel = rel2s[start:end]
        batch_tgt = targets[start:end]

        # score_fn returns [B, num_entities]
        scores = score_fn(batch_src, batch_rel)

        # Rank target entity for each path
        for i in range(len(batch_src)):
            tgt_score = scores[i, batch_tgt[i]]
            # Count how many entities score higher (1-based rank)
            rank = int((scores[i] > tgt_score).sum()) + 1
            all_ranks.append(rank)

    ranks = np.array(all_ranks, dtype=np.float64)
    mrr = float(np.mean(1.0 / ranks))
    hits_1 = float(np.mean(ranks <= 1))
    hits_3 = float(np.mean(ranks <= 3))
    hits_10 = float(np.mean(ranks <= 10))

    return {
        "multihop_MRR": mrr,
        "multihop_Hits@1": hits_1,
        "multihop_Hits@3": hits_3,
        "multihop_Hits@10": hits_10,
        "multihop_mean_rank": float(np.mean(ranks)),
        "num_paths": len(ranks),
    }


def evaluate_multihop_rag(
    agent,
    paths: list[tuple[int, int, int, int, int]],
    id_to_entity: dict[int, str],
    id_to_relation: dict[int, str],
    max_questions: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate RAG agent on multi-hop reasoning questions.

    Generates NL questions requiring 2-hop reasoning and evaluates
    whether the RAG agent can answer them correctly. RAG must retrieve
    BOTH triples in the chain to answer, which is harder than 1-hop.

    Args:
        agent: Initialized BiomedicalRAGAgent with indexed KG.
        paths: Multi-hop paths.
        id_to_entity: Int ID -> entity name.
        id_to_relation: Int ID -> relation name.
        max_questions: Max questions to evaluate.
        seed: Random seed for sampling.

    Returns:
        Dict with multihop_EM, multihop_F1, num_questions.
    """
    if not paths:
        return {"multihop_EM": 0.0, "multihop_F1": 0.0, "num_questions": 0}

    # Sample paths if too many
    if len(paths) > max_questions:
        rng = random.Random(seed)
        paths = rng.sample(paths, max_questions)

    questions = []
    answers = []

    for src, r1, mid, r2, tgt in paths:
        src_name = id_to_entity.get(src, str(src))
        mid_name = id_to_entity.get(mid, str(mid))
        tgt_name = id_to_entity.get(tgt, str(tgt))
        r1_name = id_to_relation.get(r1, str(r1))
        r2_name = id_to_relation.get(r2, str(r2))

        # Find matching template
        template = MULTIHOP_QUESTION_TEMPLATES.get((r1_name, r2_name))
        if template is None:
            template = (
                f"{src_name} is connected to {mid_name} via {r1_name}. "
                f"What entity is connected to {mid_name} via {r2_name}?"
            )
        else:
            template = template.format(
                source=src_name, intermediate=mid_name, target=tgt_name
            )

        questions.append(template)
        answers.append(tgt_name)

    # Evaluate using RAG agent
    em_scores = []
    f1_scores = []

    for q, gold in zip(questions, answers):
        try:
            pred = agent.answer_question(q)
            # Compute EM and token F1
            pred_norm = pred.lower().strip()
            gold_norm = gold.lower().strip().replace("_", " ")
            em = 1.0 if pred_norm == gold_norm else 0.0
            # Token F1
            pred_tokens = set(pred_norm.split())
            gold_tokens = set(gold_norm.split())
            if not pred_tokens or not gold_tokens:
                f1 = 0.0
            else:
                common = pred_tokens & gold_tokens
                precision = len(common) / len(pred_tokens) if pred_tokens else 0
                recall = len(common) / len(gold_tokens) if gold_tokens else 0
                f1 = (2 * precision * recall / (precision + recall)
                       if (precision + recall) > 0 else 0.0)
            em_scores.append(em)
            f1_scores.append(f1)
        except Exception as e:
            logger.warning(f"RAG evaluation error: {e}")
            em_scores.append(0.0)
            f1_scores.append(0.0)

    return {
        "multihop_EM": float(np.mean(em_scores)) if em_scores else 0.0,
        "multihop_F1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "num_questions": len(questions),
    }


# ---------------------------------------------------------------------------
# Score function factories for model-specific multi-hop evaluation
# ---------------------------------------------------------------------------

def make_pykeen_score_fn(
    model: "torch.nn.Module",
    num_entities: int,
    device: str = "cpu",
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Create a score function from a PyKEEN model.

    Uses direct embedding extraction and manual scoring to bypass
    PyKEEN's score_hrt() which causes CUDA segfaults with large entity
    sets. Same approach as the custom evaluate_link_prediction() fix.

    Returns a callable that takes (head_ids [B], rel_ids [B]) and returns
    scores [B, num_entities] for all tail entities.

    Args:
        model: Trained PyKEEN model (TransE or DistMult).
        num_entities: Total number of entities.
        device: Device for computation.

    Returns:
        Score function compatible with evaluate_multihop().
    """
    import torch
    from pykeen.models import TransE as TransEModel

    model = model.to(device)
    model.eval()

    # Extract embeddings directly (same as custom eval fix)
    with torch.no_grad():
        entity_emb = model.entity_representations[0](indices=None).to(device)
        relation_emb = model.relation_representations[0](indices=None).to(device)

    is_transe = isinstance(model, TransEModel)
    p_norm = 1
    if is_transe and hasattr(model.interaction, "p"):
        p_norm = model.interaction.p

    def score_fn(head_ids: np.ndarray, rel_ids: np.ndarray) -> np.ndarray:
        B = len(head_ids)
        heads = torch.tensor(head_ids, dtype=torch.long, device=device)
        rels = torch.tensor(rel_ids, dtype=torch.long, device=device)

        scores = np.zeros((B, num_entities), dtype=np.float32)
        with torch.no_grad():
            for i in range(B):
                h_emb = entity_emb[heads[i]]   # [D]
                r_emb = relation_emb[rels[i]]   # [D]

                if is_transe:
                    # TransE: -||h + r - t||_p (higher = better)
                    target = h_emb + r_emb  # [D]
                    dists = torch.cdist(
                        target.unsqueeze(0), entity_emb, p=p_norm
                    ).squeeze(0)  # [num_entities]
                    scores[i] = -dists.cpu().numpy()
                else:
                    # DistMult: h * r * t (dot product)
                    hr = h_emb * r_emb  # [D]
                    s = (hr.unsqueeze(0) * entity_emb).sum(dim=-1)  # [num_entities]
                    scores[i] = s.cpu().numpy()

        return scores

    return score_fn


def make_cmkl_score_fn(
    cmkl_model: "torch.nn.Module",
    node_embeddings: "torch.Tensor",
    device: str = "cpu",
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Create a score function from a CMKL model.

    Returns a callable that takes (head_ids [B], rel_ids [B]) and returns
    scores [B, num_entities] for all tail entities.

    Args:
        cmkl_model: Trained CMKL model.
        node_embeddings: Fused node embeddings [num_entities, dim].
        device: Device for computation.

    Returns:
        Score function compatible with evaluate_multihop().
    """
    import torch

    cmkl_model.eval()
    num_entities = node_embeddings.shape[0]
    emb = node_embeddings.to(device)

    def score_fn(head_ids: np.ndarray, rel_ids: np.ndarray) -> np.ndarray:
        B = len(head_ids)
        heads = torch.tensor(head_ids, dtype=torch.long, device=device)
        rels = torch.tensor(rel_ids, dtype=torch.long, device=device)
        all_tails = torch.arange(num_entities, device=device)

        scores = np.zeros((B, num_entities), dtype=np.float32)
        with torch.no_grad():
            for i in range(B):
                s = cmkl_model.score_triples(
                    emb,
                    heads[i].expand(num_entities),
                    rels[i].expand(num_entities),
                    all_tails,
                ).cpu().numpy().flatten()
                scores[i] = s

        return scores

    return score_fn
