"""KGQA (Knowledge Graph Question Answering) dataset generation.

Converts KG triples into natural language question-answer pairs
for evaluating continual KGQA. Each relation type maps to a
question template that produces a question from the head entity
and expects the tail entity as the answer.

Usage:
    from src.data.kgqa import generate_kgqa_questions, generate_continual_kgqa_dataset
    questions = generate_kgqa_questions(triples, n=100)
    qa_dataset = generate_continual_kgqa_dataset(task_sequence, per_task=200)
"""

from __future__ import annotations

import logging
import random
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

# Relation type -> (question template, answer source)
# Template uses {head} for the head entity name and expects tail as answer.
# Some relations are reversed so the question is more natural.
QUESTION_TEMPLATES: dict[str, str] = {
    # Drug-Disease relations
    "indication": "What drug is indicated for treating {head}?",
    "contraindication": "What drug is contraindicated for {head}?",
    "off-label use": "What drug has off-label use for {head}?",
    # Drug-Protein relations
    "drug_protein": "What protein does the drug {head} target?",
    "carrier": "What protein is a carrier of the drug {head}?",
    "enzyme": "What enzyme metabolizes the drug {head}?",
    "target": "What is a target of the drug {head}?",
    "transporter": "What transporter interacts with {head}?",
    # Drug-Drug relations
    "drug_drug": "What drug interacts with {head}?",
    # Disease relations
    "disease_phenotype_positive": "What phenotype is associated with the disease {head}?",
    "disease_phenotype_negative": "What phenotype is absent in {head}?",
    "disease_protein": "What protein is associated with {head}?",
    "disease_disease": "What disease is related to {head}?",
    # Protein/Gene relations
    "protein_protein": "What protein interacts with {head}?",
    "bioprocess_protein": "What protein is involved in {head}?",
    "molfunc_protein": "What protein has the molecular function {head}?",
    "cellcomp_protein": "What protein is found in {head}?",
    # Anatomy relations
    "anatomy_protein_present": "What protein is present in {head}?",
    "anatomy_protein_absent": "What protein is absent from {head}?",
    "anatomy_anatomy": "What anatomy is related to {head}?",
    # Exposure relations
    "exposure_protein": "What protein is affected by exposure to {head}?",
    "exposure_disease": "What disease is linked to exposure to {head}?",
    "exposure_bioprocess": "What biological process is affected by {head}?",
    "exposure_molfunc": "What molecular function is affected by {head}?",
    "exposure_cellcomp": "What cellular component is affected by {head}?",
    # Pathway
    "pathway_protein": "What protein participates in the pathway {head}?",
}

# Fallback template for unmapped relation types
FALLBACK_TEMPLATE = "What entity is related to {head} via {relation}?"


def _clean_entity_name(entity_id: str) -> str:
    """Convert entity ID to a readable name.

    Strips prefixes like 'MONDO:', 'DrugBank:', etc. and replaces
    underscores with spaces.

    Args:
        entity_id: Raw entity identifier.

    Returns:
        Cleaned entity name string.
    """
    # Remove common prefixes
    for prefix in ["MONDO:", "HP:", "GO:", "UBERON:", "DrugBank:",
                   "Reactome:", "CHEBI:", "NCBI:"]:
        if entity_id.startswith(prefix):
            entity_id = entity_id[len(prefix):]
            break

    return entity_id.replace("_", " ")


def generate_kgqa_questions(
    triples: np.ndarray,
    n: int | None = None,
    seed: int = 42,
    id_to_entity: dict[int, str] | None = None,
    id_to_relation: dict[int, str] | None = None,
) -> list[dict[str, str]]:
    """Generate QA pairs from KG triples.

    Each triple (head, relation, tail) becomes a question where the
    head entity is used to fill the question template and the tail
    entity is the expected answer.

    Args:
        triples: [N, 3] array of (head, relation, tail). Can be int IDs
            (requires id_to_entity/id_to_relation) or strings.
        n: Number of questions to generate. None = all triples.
        seed: Random seed for sampling.
        id_to_entity: Reverse mapping from int ID to entity name string.
        id_to_relation: Reverse mapping from int ID to relation name string.

    Returns:
        List of dicts with 'question', 'answer', 'head', 'relation', 'tail'.
    """
    rng = random.Random(seed)

    if n is not None and n < len(triples):
        indices = rng.sample(range(len(triples)), n)
        selected = triples[indices]
    else:
        selected = triples

    questions = []
    for triple in selected:
        h_raw, r_raw, t_raw = triple[0], triple[1], triple[2]

        # Convert int IDs to strings if mappings provided
        if id_to_entity is not None:
            head = id_to_entity.get(int(h_raw), str(h_raw))
            tail = id_to_entity.get(int(t_raw), str(t_raw))
        else:
            head, tail = str(h_raw), str(t_raw)

        if id_to_relation is not None:
            relation = id_to_relation.get(int(r_raw), str(r_raw))
        else:
            relation = str(r_raw)

        template = QUESTION_TEMPLATES.get(relation, FALLBACK_TEMPLATE)
        head_name = _clean_entity_name(head)
        question_text = template.format(head=head_name, relation=relation)
        answer_text = _clean_entity_name(tail)

        questions.append({
            "question": question_text,
            "answer": answer_text,
            "head": head,
            "relation": relation,
            "tail": tail,
        })

    return questions


def generate_continual_kgqa_dataset(
    task_sequence: OrderedDict[str, dict[str, np.ndarray]],
    questions_per_task: int = 200,
    seed: int = 42,
    id_to_entity: dict[int, str] | None = None,
    id_to_relation: dict[int, str] | None = None,
) -> OrderedDict[str, list[dict[str, str]]]:
    """Generate per-task QA sets aligned with the CL task sequence.

    For each task, generates questions from its test triples. This
    ensures evaluation measures knowledge acquired at each task.

    Args:
        task_sequence: OrderedDict of task_name -> {'train', 'val', 'test'}.
        questions_per_task: Number of QA pairs per task.
        seed: Random seed.
        id_to_entity: Reverse mapping from int ID to entity name string.
        id_to_relation: Reverse mapping from int ID to relation name string.

    Returns:
        OrderedDict mapping task_name -> list of QA dicts.
    """
    qa_dataset: OrderedDict[str, list[dict[str, str]]] = OrderedDict()

    for task_name, task_data in task_sequence.items():
        test_triples = task_data["test"]
        n = min(questions_per_task, len(test_triples))
        qa_pairs = generate_kgqa_questions(
            test_triples, n=n, seed=seed,
            id_to_entity=id_to_entity, id_to_relation=id_to_relation,
        )
        qa_dataset[task_name] = qa_pairs
        logger.info(f"Generated {len(qa_pairs)} QA pairs for {task_name}")

    total = sum(len(v) for v in qa_dataset.values())
    logger.info(f"Total KGQA dataset: {total} questions across "
                f"{len(qa_dataset)} tasks")
    return qa_dataset
