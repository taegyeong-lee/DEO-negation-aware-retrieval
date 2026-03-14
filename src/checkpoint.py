"""CheckpointManager – manages decomposition, embedding, progress, and result caches."""

import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from .utils import get_query_hash, get_weight_hash


class CheckpointManager:

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.decompose_dir = os.path.join(checkpoint_dir, "decompositions")
        self.embedding_dir = os.path.join(checkpoint_dir, "embeddings")
        self.results_dir = os.path.join(checkpoint_dir, "results")
        self.progress_dir = os.path.join(checkpoint_dir, "progress")

        for dir_path in [self.decompose_dir, self.embedding_dir,
                         self.results_dir, self.progress_dir]:
            os.makedirs(dir_path, exist_ok=True)

    # ── Decomposition save / load ──

    def save_decomposition(
        self, query: str, decomposition: Dict[str, List[str]],
        dataset: str = "default", decompose_model: str = None,
        prompt_sys: str = None, prompt_version: str = "v1"
    ):
        query_hash = get_query_hash(query)
        model_tag = decompose_model.replace("/", "_") if decompose_model else "unknownLLM"
        pv_tag = (prompt_version or "v1").replace("/", "_")

        filename = f"{dataset}_{query_hash}_{model_tag}_{pv_tag}_decompose.json"
        filepath = os.path.join(self.decompose_dir, filename)

        data = {
            "query": query,
            "query_hash": query_hash,
            "dataset": dataset,
            "decompose_model": decompose_model,
            "timestamp": datetime.now().isoformat(),
            "decomposition": decomposition,
            "prompt_system": (prompt_sys.splitlines() if prompt_sys else []),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return filepath

    def load_decomposition(
        self, query: str, dataset: str = "default",
        decompose_model: str = None, prompt_version: str = "v1"
    ) -> Optional[Dict]:
        query_hash = get_query_hash(query)
        model_tag = decompose_model.replace("/", "_") if decompose_model else "unknownLLM"
        pv_tag = (prompt_version or "v1").replace("/", "_")
        filename = f"{dataset}_{query_hash}_{model_tag}_{pv_tag}_decompose.json"
        filepath = os.path.join(self.decompose_dir, filename)

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data["decomposition"]
        return None

    # ── Embedding save / load ──

    def get_embedding_filename(
        self, query: str, dataset: str,
        reg_weight: float, pos_weight: float,
        neg_weight: float, num_steps: int, lr: float,
        embed_model: str = None,
        decompose_model: str = None,
        prompt_version: str = None
    ) -> str:
        """Build embedding cache filename (unique per weight config)."""
        query_hash = get_query_hash(query)
        weight_hash = get_weight_hash(reg_weight, pos_weight, neg_weight, num_steps, lr)

        emb_tag = embed_model.replace("/", "_") if embed_model else "unknownEmbed"
        dec_tag = decompose_model.replace("/", "_") if decompose_model else "unknownDecomp"
        pv_tag = (prompt_version or "v1").replace("/", "_")

        return f"{dataset}_{query_hash}__dec_{dec_tag}__pv_{pv_tag}__{weight_hash}__emb_{emb_tag}_embedding.npz"

    def save_optimized_embedding(
        self, query: str, embedding: np.ndarray,
        metadata: Dict, dataset: str = "default",
        decompose_model: str = None,
        prompt_version: str = None
    ):
        """Save optimized embedding (separate file per weight config)."""
        filename = self.get_embedding_filename(
            query, dataset,
            metadata.get("reg_weight", 1.0),
            metadata.get("pos_weight", 1.0),
            metadata.get("neg_weight", 1.0),
            metadata.get("num_steps", 100),
            metadata.get("lr", 0.01),
            metadata.get("embed_model"),
            decompose_model=decompose_model,
            prompt_version=prompt_version
        )

        filepath = os.path.join(self.embedding_dir, filename)

        # Separate numpy arrays for storage
        np_data = {"embedding": embedding}
        if "loss_history" in metadata:
            for key, value in metadata["loss_history"].items():
                np_data[f"loss_history_{key}"] = np.array(value)

        np.savez_compressed(filepath, **np_data)

        # Save metadata as JSON alongside
        meta_filepath = filepath.replace(".npz", "_meta.json")
        json_metadata = {
            "query": query,
            "query_hash": get_query_hash(query),
            "dataset": dataset,
            "timestamp": datetime.now().isoformat(),
            "decompose_model": decompose_model,
            "prompt_version": prompt_version,
            **{k: v for k, v in metadata.items() if not isinstance(v, np.ndarray)}
        }

        with open(meta_filepath, "w", encoding="utf-8") as f:
            json.dump(json_metadata, f, ensure_ascii=False, indent=2)

        return filepath

    def load_optimized_embedding(
        self, query: str, dataset: str = "default",
        reg_weight: float = 1.0, pos_weight: float = 1.0,
        neg_weight: float = 1.0, num_steps: int = 100,
        lr: float = 0.01, embed_model: str = None,
        decompose_model: str = None,
        prompt_version: str = None
    ) -> Optional[Tuple[np.ndarray, Dict]]:
        """Load a previously saved optimized embedding."""
        filename = self.get_embedding_filename(
            query, dataset, reg_weight, pos_weight, neg_weight, num_steps, lr,
            embed_model=embed_model,
            decompose_model=decompose_model,
            prompt_version=prompt_version
        )
        filepath = os.path.join(self.embedding_dir, filename)

        if os.path.exists(filepath):
            data = np.load(filepath, allow_pickle=True)
            embedding = data["embedding"]

            meta_filepath = filepath.replace(".npz", "_meta.json")
            metadata = {}
            if os.path.exists(meta_filepath):
                with open(meta_filepath, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

            return embedding, metadata
        return None

    # ── Progress save / load ──

    def save_progress(self, dataset: str, processed_queries: List[str],
                      total_queries: int, results: Dict = None,
                      weight_config: Dict = None):
        """Save retrieval progress (includes weight config for resume validation)."""
        filename = f"{dataset}_progress.json"
        filepath = os.path.join(self.progress_dir, filename)

        data = {
            "dataset": dataset,
            "processed_queries": processed_queries,
            "total_queries": total_queries,
            "progress_percentage": len(processed_queries) / total_queries * 100,
            "last_update": datetime.now().isoformat(),
            "weight_config": weight_config or {},
            "results": results or {}
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return filepath

    def load_progress(self, dataset: str) -> Optional[Dict]:
        """Load retrieval progress."""
        filename = f"{dataset}_progress.json"
        filepath = os.path.join(self.progress_dir, filename)

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    # ── Final results save ──

    def save_final_results(self, dataset: str, results: Dict, metrics: Dict,
                           weight_config: Dict = None, params: Dict = None):
        """Save final retrieval results and metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if weight_config:
            weight_str = get_weight_hash(
                weight_config.get("reg_weight", 1.0),
                weight_config.get("pos_weight", 1.0),
                weight_config.get("neg_weight", 1.0),
                weight_config.get("num_steps", 100),
                weight_config.get("lr", 0.01)
            )
            filename = f"{dataset}_results_{weight_str}_{timestamp}.json"
        else:
            filename = f"{dataset}_results_{timestamp}.json"

        filepath = os.path.join(self.results_dir, filename)

        data = {
            "dataset": dataset,
            "timestamp": datetime.now().isoformat(),
            "weight_config": weight_config or {},
            "metrics": metrics,
            "num_queries": len(results),
            "params": params or {},
            "results": results,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Results saved to: {filepath}")
        return filepath
