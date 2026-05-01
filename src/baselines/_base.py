"""Shared infrastructure for continual KGE baselines.

Provides:
- Memory-efficient task loading (int IDs, not string arrays)
- Global entity/relation mapping across all tasks
- PyKEEN TriplesFactory creation from pre-mapped triples
- Model creation (TransE, ComplEx, DistMult, RotatE)
- Evaluation (MRR, Hits@K) on arbitrary test sets
- Results matrix tracking

All concrete baselines extend this module's functions.
"""

from __future__ import annotations

import logging
import resource
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from pykeen.models import TransE, ComplEx, DistMult, RotatE
from pykeen.triples import TriplesFactory

logger = logging.getLogger(__name__)


def _log_mem(label: str) -> None:
    """Log current RSS memory usage."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    logger.info(f"[MEM] {label}: {rss_mb:.0f} MB")


MODEL_REGISTRY = {
    "TransE": TransE,
    "ComplEx": ComplEx,
    "DistMult": DistMult,
    "RotatE": RotatE,
}


# ---------------------------------------------------------------------------
# Memory-efficient data loading: stream files → int ID arrays directly
# Never creates numpy string arrays (which use fixed-width dtype and
# consume ~50 GB for 8M triples).
# ---------------------------------------------------------------------------

def _scan_vocab(
    tasks_dir: Path,
    task_names: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Stream all task files to build entity/relation vocabularies.

    Reads files line by line without storing strings in memory.

    Returns:
        (entity_to_id, relation_to_id) dicts.
    """
    entities: set[str] = set()
    relations: set[str] = set()

    for name in task_names:
        for split_file in ("train.txt", "valid.txt", "test.txt"):
            path = tasks_dir / name / split_file
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        entities.add(parts[0])
                        entities.add(parts[2])
                        relations.add(parts[1])

    entity_to_id = {e: i for i, e in enumerate(sorted(entities))}
    relation_to_id = {r: i for i, r in enumerate(sorted(relations))}
    logger.info(f"Global vocab: {len(entity_to_id):,} entities, "
                f"{len(relation_to_id)} relations")
    return entity_to_id, relation_to_id


def _load_mapped_triples(
    path: Path,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> np.ndarray:
    """Load a triples file directly as int64 array using pre-built mappings.

    Args:
        path: Path to tab-separated triples file.
        entity_to_id: Entity string → int mapping.
        relation_to_id: Relation string → int mapping.

    Returns:
        int64 numpy array of shape (n, 3) with columns [head_id, relation_id, tail_id].
    """
    ids: list[list[int]] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                h, r, t = parts
                if h in entity_to_id and r in relation_to_id and t in entity_to_id:
                    ids.append([entity_to_id[h], relation_to_id[r], entity_to_id[t]])
    if not ids:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(ids, dtype=np.int64)


def load_task_sequence(
    tasks_dir: str | Path,
    task_names: list[str] | None = None,
) -> tuple[OrderedDict[str, dict[str, np.ndarray]], dict[str, int], dict[str, int]]:
    """Load task sequence as memory-efficient int arrays.

    Two-pass approach:
    1. Stream all files to build entity/relation vocabularies
    2. Stream again to convert triples directly to int64 arrays

    Memory usage: ~200 MB for 8M triples (vs ~50 GB with numpy string arrays).

    Args:
        tasks_dir: Path to benchmark/tasks directory.
        task_names: Specific task names to load. If None, loads all sorted.

    Returns:
        Tuple of (tasks, entity_to_id, relation_to_id) where tasks is
        OrderedDict mapping task_name → {'train': int64_array, 'val': ..., 'test': ...}.
    """
    tasks_dir = Path(tasks_dir)

    if task_names is None:
        task_names = sorted([
            d.name for d in tasks_dir.iterdir()
            if d.is_dir() and (d / "train.txt").exists()
        ])

    _log_mem("before loading tasks")

    # Pass 1: build vocab by streaming files
    entity_to_id, relation_to_id = _scan_vocab(tasks_dir, task_names)
    _log_mem("after vocab scan")

    # Pass 2: load triples as int arrays
    split_map = {"train": "train.txt", "val": "valid.txt", "test": "test.txt"}
    tasks: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    for name in task_names:
        task_dir = tasks_dir / name
        task_data: dict[str, np.ndarray] = {}
        for split_key, filename in split_map.items():
            task_data[split_key] = _load_mapped_triples(
                task_dir / filename, entity_to_id, relation_to_id
            )
        tasks[name] = task_data

        total = sum(len(v) for v in task_data.values())
        logger.info(f"Loaded {name}: {total:,} triples "
                    f"(train={len(task_data['train']):,}, "
                    f"val={len(task_data['val']):,}, "
                    f"test={len(task_data['test']):,})")
        _log_mem(f"after loading {name}")

    return tasks, entity_to_id, relation_to_id


def make_triples_factory(
    mapped_triples: np.ndarray,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> TriplesFactory:
    """Create a PyKEEN TriplesFactory from pre-mapped int triples.

    Args:
        mapped_triples: int64 array of shape (n, 3) with [head_id, rel_id, tail_id].
        entity_to_id: Entity string → int mapping.
        relation_to_id: Relation string → int mapping.

    Returns:
        TriplesFactory ready for PyKEEN training/evaluation.
    """
    tensor = torch.as_tensor(mapped_triples, dtype=torch.long)
    return TriplesFactory(
        mapped_triples=tensor,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )


def create_model(
    model_name: str,
    triples_factory: TriplesFactory,
    embedding_dim: int = 256,
    random_seed: int = 42,
) -> torch.nn.Module:
    """Create a PyKEEN KGE model.

    Args:
        model_name: One of 'TransE', 'ComplEx', 'DistMult', 'RotatE'.
        triples_factory: TriplesFactory defining the entity/relation vocab.
        embedding_dim: Embedding dimension.
        random_seed: Random seed for reproducibility.

    Returns:
        Initialized PyKEEN model.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY)}")

    torch.manual_seed(random_seed)
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(
        triples_factory=triples_factory,
        embedding_dim=embedding_dim,
        random_seed=random_seed,
    )
    return model


def evaluate_link_prediction(
    model: torch.nn.Module,
    test_factory: TriplesFactory,
    device: str = "cpu",
    batch_size: int = 64,
    all_known_mapped_triples: torch.Tensor | None = None,
    max_test_triples: int = 50_000,
) -> dict[str, float]:
    """Evaluate model on a test set using rank-based metrics.

    Computes MRR, Hits@1, Hits@3, Hits@10 with filtered ranking.
    Uses direct embedding extraction and manual scoring (bypasses
    PyKEEN's RankBasedEvaluator to avoid CUDA segfaults with large
    entity sets).

    For each test triple (h, r, t):
    - Score all entities as potential tails: score(h, r, e) for all e
    - Filter out known true tails (except the test triple itself)
    - Rank the true tail among remaining entities

    Args:
        model: Trained PyKEEN model (TransE or DistMult).
        test_factory: TriplesFactory for test data.
        device: Device for evaluation.
        batch_size: Evaluation batch size (default 64 for safety).
        all_known_mapped_triples: Concatenated train+val+test triples from
            all tasks for filtered ranking. If None, uses raw ranking.
        max_test_triples: Maximum test triples to evaluate. Larger sets
            are randomly subsampled. Default 50K.

    Returns:
        Dict with MRR, Hits@1, Hits@3, Hits@10.
    """
    from pykeen.models import TransE as TransEModel

    mapped_triples = test_factory.mapped_triples
    if mapped_triples.shape[0] > max_test_triples:
        logger.warning(
            f"Test set has {mapped_triples.shape[0]:,} triples, "
            f"sampling {max_test_triples:,} for evaluation"
        )
        indices = torch.randperm(mapped_triples.shape[0])[:max_test_triples]
        mapped_triples = mapped_triples[indices]

    _log_mem(f"before eval ({mapped_triples.shape[0]:,} test triples)")
    model = model.to(device)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Extract embeddings directly from PyKEEN model
    with torch.no_grad():
        entity_emb = model.entity_representations[0](indices=None).to(device)
        relation_emb = model.relation_representations[0](indices=None).to(device)

    # Detect model type for scoring
    from pykeen.models import RotatE as RotatEModel, ComplEx as ComplExModel
    is_transe = isinstance(model, TransEModel)
    is_rotate = isinstance(model, RotatEModel)
    is_complex = isinstance(model, ComplExModel)
    p_norm = 1
    if is_transe and hasattr(model.interaction, "p"):
        p_norm = model.interaction.p

    # RotatE and ComplEx use complex embeddings — convert to real for scoring
    if entity_emb.is_complex():
        if is_rotate:
            # RotatE: score(h,r,t) = -||h ∘ r - t|| in complex space
            # For all-tail scoring, compute per-query and take real norm
            pass  # handled in scoring loop below
        elif is_complex:
            # ComplEx: score = Re(sum(h * r * conj(t)))
            pass  # handled in scoring loop below
        else:
            # Fallback: take real part
            entity_emb = entity_emb.real
            relation_emb = relation_emb.real

    # Build filter: for each (h, r), collect known tail entities
    hr_to_tails: dict[tuple[int, int], set[int]] = {}
    if all_known_mapped_triples is not None:
        known_np = all_known_mapped_triples.cpu().numpy()
        for i in range(known_np.shape[0]):
            key = (int(known_np[i, 0]), int(known_np[i, 1]))
            if key not in hr_to_tails:
                hr_to_tails[key] = set()
            hr_to_tails[key].add(int(known_np[i, 2]))

    ranks = []
    test_t = mapped_triples.to(device)

    with torch.no_grad():
        for start in range(0, len(test_t), batch_size):
            batch = test_t[start:start + batch_size]
            heads = batch[:, 0]
            rels = batch[:, 1]
            tails = batch[:, 2]
            B = heads.shape[0]

            h_emb = entity_emb[heads]   # [B, D]
            r_emb = relation_emb[rels]  # [B, D]

            # Score all entities as tails
            if is_rotate:
                # RotatE: score = -||h ∘ r - t||_p in complex space (p=2 default)
                # Score in entity sub-batches to avoid OOM on [B, N, D]
                query = h_emb * r_emb  # [B, D_complex]
                rot_p = getattr(model.interaction, 'p', 2)
                N = entity_emb.shape[0]
                all_scores = torch.empty(B, N, device=device)
                ent_chunk = 8192
                for ei in range(0, N, ent_chunk):
                    ej = min(ei + ent_chunk, N)
                    diff = query.unsqueeze(1) - entity_emb[ei:ej].unsqueeze(0)
                    all_scores[:, ei:ej] = -(diff.abs() ** rot_p).sum(dim=-1).float() ** (1 / rot_p)
            elif is_complex:
                # ComplEx: score = Re(sum(h * r * conj(t)))
                query = h_emb * r_emb  # [B, D_complex]
                N = entity_emb.shape[0]
                all_scores = torch.empty(B, N, device=device)
                ent_chunk = 8192
                for ei in range(0, N, ent_chunk):
                    ej = min(ei + ent_chunk, N)
                    all_scores[:, ei:ej] = (query.unsqueeze(1) * entity_emb[ei:ej].conj().unsqueeze(0)).sum(dim=-1).real.float()
            elif is_transe:
                query = h_emb + r_emb  # [B, D]
                all_scores = -torch.cdist(query, entity_emb, p=p_norm)  # [B, N]
            else:
                # DistMult / default: element-wise product
                query = h_emb * r_emb  # [B, D]
                all_scores = query @ entity_emb.T  # [B, N]

            # Filtered ranking: mask known tails except the true one
            if hr_to_tails:
                for b_idx in range(B):
                    h_val = heads[b_idx].item()
                    r_val = rels[b_idx].item()
                    t_val = tails[b_idx].item()
                    known = hr_to_tails.get((h_val, r_val), set())
                    if known:
                        mask_ids = [t for t in known if t != t_val]
                        if mask_ids:
                            all_scores[b_idx, mask_ids] = float("-inf")

            # Compute rank of true tail (pessimistic: ties count against)
            true_scores = all_scores[torch.arange(B, device=device), tails]
            batch_ranks = (
                (all_scores >= true_scores.unsqueeze(1)).sum(dim=1).float()
            )
            ranks.extend(batch_ranks.cpu().tolist())

    _log_mem("after eval")

    if not ranks:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}

    ranks_arr = np.array(ranks)
    rr = 1.0 / ranks_arr
    metrics = {
        "MRR": float(np.mean(rr)),
        "Hits@1": float(np.mean(ranks_arr <= 1)),
        "Hits@3": float(np.mean(ranks_arr <= 3)),
        "Hits@10": float(np.mean(ranks_arr <= 10)),
    }
    return metrics


def train_epoch(
    model: torch.nn.Module,
    train_factory: TriplesFactory,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    batch_size: int = 256,
    extra_loss_fn: callable | None = None,
) -> float:
    """Train model for one epoch using sLCWA (stochastic local closed-world assumption).

    Uses negative sampling with the model's built-in loss function.

    Args:
        model: PyKEEN KGE model.
        train_factory: Training TriplesFactory.
        optimizer: PyTorch optimizer.
        device: Device.
        batch_size: Training batch size.
        extra_loss_fn: Optional callable returning additional loss term
            (e.g., EWC penalty). Added to the base KGE loss.

    Returns:
        Average loss over the epoch.
    """
    from pykeen.models import TransE as TransEModel

    model = model.to(device)
    model.train()
    is_transe = isinstance(model, TransEModel)

    mapped_cpu = train_factory.mapped_triples
    n = mapped_cpu.shape[0]

    perm = torch.randperm(n)
    mapped_cpu = mapped_cpu[perm]

    total_loss = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        batch = mapped_cpu[start:start + batch_size].to(device)

        neg_batch = _generate_negatives(
            batch, model.num_entities, device=device
        )

        pos_scores = _score_triples(model, batch, is_transe)
        neg_scores = _score_triples(model, neg_batch, is_transe)

        loss = _margin_loss(pos_scores, neg_scores, margin=1.0)

        if extra_loss_fn is not None:
            loss = loss + extra_loss_fn()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Free computation graph immediately
        del pos_scores, neg_scores, loss, batch, neg_batch

    # Release GPU cache between epochs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_loss / max(n_batches, 1)


def _score_triples(
    model: torch.nn.Module,
    triples: torch.Tensor,
    is_transe: bool,
) -> torch.Tensor:
    """Score triples using direct embedding lookup.

    Bypasses PyKEEN's score_hrt and Representation.__call__ to avoid:
    1. GPU OOM from PyKEEN's internal scoring overhead
    2. Memory leak from LpRegularizer.update() accumulating computation
       graphs across batches (regularizer.regularization_term grows unboundedly)

    Supports TransE, DistMult, RotatE (complex), and ComplEx (complex).

    Args:
        model: PyKEEN model with entity_representations and relation_representations.
        triples: Tensor of shape (B, 3) with [head, relation, tail].
        is_transe: If True, use TransE scoring; otherwise auto-detect.

    Returns:
        Scores of shape (B, 1), always real-valued.
    """
    from pykeen.models import RotatE as RotatEModel, ComplEx as ComplExModel

    h_idx = triples[:, 0]
    r_idx = triples[:, 1]
    t_idx = triples[:, 2]

    # Access raw nn.Embedding directly, bypassing PyKEEN's Representation
    # layer which triggers LpRegularizer.update() and leaks GPU memory.
    entity_weight = model.entity_representations[0]._embeddings.weight
    relation_weight = model.relation_representations[0]._embeddings.weight

    h_raw = entity_weight[h_idx]
    r_raw = relation_weight[r_idx]
    t_raw = entity_weight[t_idx]

    if isinstance(model, RotatEModel):
        # RotatE: raw weights are real [B, 2D], need complex [B, D].
        # Entity: view as complex directly.
        # Relation: raw weights are PHASES (theta), convert via exp(i*theta).
        h_emb = torch.view_as_complex(h_raw.reshape(h_raw.shape[0], -1, 2))
        t_emb = torch.view_as_complex(t_raw.reshape(t_raw.shape[0], -1, 2))
        # Relation phases -> unit complex: r = exp(i * theta)
        r_phases = torch.view_as_complex(r_raw.reshape(r_raw.shape[0], -1, 2))
        r_emb = r_phases / r_phases.abs().clamp(min=1e-8)  # normalize to unit modulus
        diff = h_emb * r_emb - t_emb
        p = getattr(model.interaction, 'p', 2)
        scores = -(diff.abs() ** p).sum(dim=-1, keepdim=True) ** (1 / p)
        scores = scores.float()
    elif isinstance(model, ComplExModel):
        # ComplEx: raw weights are real [B, 2D], need complex [B, D].
        h_emb = torch.view_as_complex(h_raw.reshape(h_raw.shape[0], -1, 2))
        r_emb = torch.view_as_complex(r_raw.reshape(r_raw.shape[0], -1, 2))
        t_emb = torch.view_as_complex(t_raw.reshape(t_raw.shape[0], -1, 2))
        scores = (h_emb * r_emb * t_emb.conj()).sum(dim=-1, keepdim=True).real.float()
    elif is_transe:
        p = getattr(model.interaction, 'p', 1)
        if p == 1:
            scores = -(h_raw + r_raw - t_raw).abs().sum(dim=-1, keepdim=True)
        else:
            scores = -(h_raw + r_raw - t_raw).norm(p=p, dim=-1, keepdim=True)
    else:
        # DistMult / default: element-wise product
        scores = (h_raw * r_raw * t_raw).sum(dim=-1, keepdim=True)

    return scores


def _generate_negatives(
    pos_batch: torch.Tensor,
    num_entities: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate negative samples by corrupting head or tail.

    Randomly replaces head or tail entity with a random entity.

    Args:
        pos_batch: Positive triples of shape (batch, 3).
        num_entities: Total number of entities.
        device: Device.

    Returns:
        Negative triples of shape (batch, 3).
    """
    neg = pos_batch.clone()
    n = neg.shape[0]
    # 50% corrupt head, 50% corrupt tail
    mask = torch.rand(n, device=device) < 0.5
    random_entities = torch.randint(0, num_entities, (n,), device=device)
    neg[mask, 0] = random_entities[mask]
    neg[~mask, 2] = random_entities[~mask]
    return neg


def _margin_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Margin ranking loss for KGE training.

    L = max(0, margin - pos_score + neg_score)

    Higher score = more plausible triple.
    """
    return torch.nn.functional.relu(margin - pos_scores + neg_scores).mean()


def get_device(requested: str = "auto") -> str:
    """Get the best available device.

    Args:
        requested: 'auto', 'cuda', 'mps', or 'cpu'.

    Returns:
        Device string.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested
