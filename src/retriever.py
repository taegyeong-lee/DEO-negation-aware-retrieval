"""OptimizedPolarityRetriever – combines LLM decomposition + embedding optimization."""

from typing import Dict

import numpy as np

from .config import ExperimentConfig
from .checkpoint import CheckpointManager
from .llm import PolarityLLM
from .embedder import OptimizedEmbedder
from .indexer import CorpusIndex


class OptimizedPolarityRetriever:

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.checkpoint_manager = CheckpointManager(cfg.checkpoint_dir)

        self.llm = PolarityLLM(cfg, checkpoint_manager=self.checkpoint_manager)
        self.embedder = OptimizedEmbedder(
            model_name=cfg.embed_model,
            checkpoint_manager=self.checkpoint_manager,
        )
        self.indexer = CorpusIndex(
            self.embedder, cache_dir=cfg.cache_dir, use_gpu=cfg.use_gpu_index,
        )
        self.dataset_name = None

    def build_index(self, dataset_name: str, corpus: Dict[str, Dict]):
        self.dataset_name = dataset_name
        self.indexer.build_or_load(dataset_name, corpus)

    def search(self, corpus: Dict[str, Dict], queries: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        assert self.dataset_name is not None, "Call build_index() first."

        cfg = self.cfg
        results = {}
        processed_queries = []
        total_queries = len(queries)
        weight_config = cfg.weight_config

        # Resume from previous progress
        if cfg.resume:
            progress = self.checkpoint_manager.load_progress(self.dataset_name)
            if progress:
                prev_weight = progress.get("weight_config", {})
                if prev_weight == weight_config:
                    processed_queries = progress["processed_queries"]
                    results = progress.get("results", {})
                    print(f"\n[Resume] {len(processed_queries)}/{total_queries} queries already done")
                else:
                    print(f"\n[Resume] Weight config changed, starting fresh")

        for idx, (qid, qtext) in enumerate(queries.items(), 1):
            if qid in processed_queries:
                continue

            try:
                print(f"\n[Progress] Processing query {idx}/{total_queries}: {qid}")

                qvec = self._get_optimized_query_embedding(qtext)
                hits = self.indexer.search(qvec, top_k=cfg.top_k)
                results[qid] = {doc_id: score for doc_id, score in hits}

                processed_queries.append(qid)

                if len(processed_queries) % 10 == 0:
                    self.checkpoint_manager.save_progress(
                        self.dataset_name, processed_queries,
                        total_queries, results, weight_config,
                    )
                    print(f"[Checkpoint] Saved progress: {len(processed_queries)}/{total_queries}")

            except Exception as e:
                print(f"[Error] Failed to process query {qid}: {e}")
                self.checkpoint_manager.save_progress(
                    self.dataset_name, processed_queries,
                    total_queries, results, weight_config,
                )
                raise

        # Final save
        self.checkpoint_manager.save_progress(
            self.dataset_name, processed_queries,
            total_queries, results, weight_config,
        )
        return results

    def _get_optimized_query_embedding(self, query: str) -> np.ndarray:
        cfg = self.cfg

        if not cfg.use_optimization:
            embedding = self.embedder.encode(query, normalize=True)
            if cfg.verbose:
                print(f"[Baseline] Query: {query}")
            return embedding.cpu().numpy().astype("float32")

        expansion = self.llm.expand(query, dataset=self.dataset_name)
        positives = expansion.get("positives", [query])
        negatives = expansion.get("negatives", [])

        if cfg.verbose:
            print(f"\n[Query]: {query}")
            print(f"[Positives]: {positives}")
            print(f"[Negatives]: {negatives}")

        optimized_embedding = self.embedder.optimize_embedding(
            query=query,
            positives=positives,
            negatives=negatives,
            dataset=self.dataset_name,
            num_steps=cfg.optimization_steps,
            lr=cfg.lr,
            reg_weight=cfg.reg_weight,
            pos_weight=cfg.pos_weight,
            neg_weight=cfg.neg_weight,
            verbose=cfg.verbose,
            use_cache=cfg.use_cache,
            decompose_model=self.llm.decompose_model,
            prompt_version=cfg.prompt_version,
        )

        return optimized_embedding.cpu().numpy().astype("float32")
