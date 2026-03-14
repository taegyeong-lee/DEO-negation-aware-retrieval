"""Dataset-specific evaluation metrics (pairwise accuracy)."""

from typing import Dict, List


def evaluate_pairwise_nevir(results: Dict[str, Dict[str, float]], pairs: List[dict]) -> dict:
    """
    NevIR pairwise accuracy: both q1->doc1 and q2->doc2 must rank correctly.
    """
    correct = 0
    total = 0
    q1_correct = 0
    q2_correct = 0

    for pair in pairs:
        q1_id = pair["q1_id"]
        q2_id = pair["q2_id"]
        doc1_id = pair["doc1_id"]
        doc2_id = pair["doc2_id"]

        if q1_id not in results or q2_id not in results:
            continue

        total += 1

        s_q1_d1 = results[q1_id].get(doc1_id, -float("inf"))
        s_q1_d2 = results[q1_id].get(doc2_id, -float("inf"))
        q1_ok = s_q1_d1 > s_q1_d2

        s_q2_d2 = results[q2_id].get(doc2_id, -float("inf"))
        s_q2_d1 = results[q2_id].get(doc1_id, -float("inf"))
        q2_ok = s_q2_d2 > s_q2_d1

        if q1_ok:
            q1_correct += 1
        if q2_ok:
            q2_correct += 1
        if q1_ok and q2_ok:
            correct += 1

    pairwise_acc = correct / total if total > 0 else 0.0
    q1_acc = q1_correct / total if total > 0 else 0.0
    q2_acc = q2_correct / total if total > 0 else 0.0

    return {
        "pairwise_accuracy": pairwise_acc,
        "q1_accuracy": q1_acc,
        "q2_accuracy": q2_acc,
        "total_pairs": total,
        "correct_pairs": correct,
    }


def evaluate_custom_dataset(dataset_name: str, results: Dict, extra: list) -> dict:
    """Run dataset-specific evaluation."""
    name = dataset_name.lower()
    if name == "nevir":
        return evaluate_pairwise_nevir(results, extra)
    else:
        raise ValueError(f"Unknown custom dataset: {dataset_name}")
