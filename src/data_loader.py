"""
Custom dataset loaders for negation-aware retrieval benchmarks.
Converts NevIR into BEIR-compatible format (corpus, queries, qrels).
"""

import os
from typing import Dict, Tuple, List

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from datasets import load_dataset

# BEIR-compatible type aliases
Corpus = Dict[str, Dict[str, str]]
Queries = Dict[str, str]
Qrels = Dict[str, Dict[str, int]]

CUSTOM_DATASETS = {"nevir"}


def is_custom_dataset(dataset_name: str) -> bool:
    return dataset_name.lower() in CUSTOM_DATASETS


def load_dataset_auto(dataset: str, split: str = "test",
                      data_root: str = "beir") -> Tuple[Corpus, Queries, Qrels, object]:
    """
    Unified dataset loader. Returns (corpus, queries, qrels, extra).
    extra is None for BEIR datasets, dataset-specific metadata for custom ones.
    """
    if is_custom_dataset(dataset):
        return _load_nevir(split=split)

    data_path = os.path.join(data_root, dataset)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

    if not os.path.exists(data_path):
        os.makedirs(data_root, exist_ok=True)
        print(f"[Download] Downloading {dataset} dataset...")
        util.download_and_unzip(url, data_root)

    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
    return corpus, queries, qrels, None


# ── NevIR ──

def _load_nevir(split: str = "test") -> Tuple[Corpus, Queries, Qrels, List[dict]]:
    hf_split = {"test": "test", "dev": "validation", "train": "train"}.get(split, split)

    print(f"[NevIR] Loading split={hf_split} from HuggingFace...")
    ds = load_dataset("orionweller/NevIR", split=hf_split)

    corpus: Corpus = {}
    queries: Queries = {}
    qrels: Qrels = {}
    pairs: List[dict] = []

    for row in ds:
        pair_id = str(row["id"])
        doc1_id = f"nevir_doc_{pair_id}_1"
        doc2_id = f"nevir_doc_{pair_id}_2"
        q1_id = f"nevir_q_{pair_id}_1"
        q2_id = f"nevir_q_{pair_id}_2"

        corpus[doc1_id] = {"text": row["doc1"], "title": ""}
        corpus[doc2_id] = {"text": row["doc2"], "title": ""}
        queries[q1_id] = row["q1"]
        queries[q2_id] = row["q2"]
        qrels[q1_id] = {doc1_id: 1}
        qrels[q2_id] = {doc2_id: 1}

        pairs.append({
            "pair_id": pair_id,
            "q1_id": q1_id, "q2_id": q2_id,
            "doc1_id": doc1_id, "doc2_id": doc2_id,
        })

    print(f"[NevIR] Loaded {len(corpus)} docs, {len(queries)} queries, {len(pairs)} pairs")
    return corpus, queries, qrels, pairs
