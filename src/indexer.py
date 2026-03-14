"""CorpusIndex – corpus embedding and FAISS index management."""

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import faiss

from .embedder import OptimizedEmbedder


class CorpusIndex:

    def __init__(self, embedder: OptimizedEmbedder, cache_dir: str = "./beir_cache", use_gpu: bool = True):
        self.embedder = embedder
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.index = None
        self.doc_ids: List[str] = []
        self.dim = None
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        if self.use_gpu:
            print(f"[FAISS] GPU acceleration enabled ({faiss.get_num_gpus()} GPUs)")
        else:
            print("[FAISS] Using CPU for indexing")

    def _paths(self, dataset):
        if hasattr(self.embedder, "model_name"):
            model_name = self.embedder.model_name.replace("/", "_")
        elif hasattr(self.embedder.model, "name_or_path"):
            model_name = self.embedder.model.name_or_path.replace("/", "_")
        else:
            model_name = "unknown_model"

        base = os.path.join(self.cache_dir, f"{dataset}__{model_name}")
        return base + ".npy", base + ".json", base + ".faiss"

    def build_or_load(self, dataset: str, corpus: Dict[str, Dict[str, str]]) -> None:
        emb_path, ids_path, faiss_path = self._paths(dataset)

        # Load from cache if available
        if os.path.exists(emb_path) and os.path.exists(ids_path) and os.path.exists(faiss_path):
            with open(ids_path, "r", encoding="utf-8") as f:
                self.doc_ids = json.load(f)
            embeddings = np.load(emb_path)
            self.dim = embeddings.shape[1]
            self.index = faiss.read_index(faiss_path)
            if self.use_gpu:
                self.index = faiss.index_cpu_to_all_gpus(self.index)
            print(f"[Cache] Loaded index for {dataset} with {len(self.doc_ids)} documents")
            return

        # Build new embeddings
        print(f"[Building] Creating new index for {dataset}...")
        self.doc_ids = list(corpus.keys())
        texts = [corpus[_id].get("text", "") or (corpus[_id].get("title", "") or "") for _id in self.doc_ids]

        batch_size = 64
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedder.encode(batch_texts, normalize=True)
            all_embeddings.append(batch_embeddings.cpu().numpy())

            if (i // batch_size) % 10 == 0:
                print(f"  Encoded {i + len(batch_texts)}/{len(texts)} documents")

        embeddings = np.vstack(all_embeddings).astype("float32")
        self.dim = embeddings.shape[1]

        # Save cache
        np.save(emb_path, embeddings)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f)

        # Build FAISS index (inner product for cosine similarity on normalized vectors)
        cpu_index = faiss.IndexFlatIP(self.dim)
        cpu_index.add(embeddings)
        faiss.write_index(cpu_index, faiss_path)

        self.index = cpu_index
        if self.use_gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index)

        print(f"[Complete] Built and cached index for {dataset} with {len(self.doc_ids)} documents")

    def search(self, qvec: np.ndarray, top_k: int = 100) -> List[Tuple[str, float]]:
        q = qvec.astype("float32").reshape(1, -1)
        D, I = self.index.search(q, top_k)
        hits = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1:
                continue
            hits.append((self.doc_ids[idx], float(score)))
        return hits
