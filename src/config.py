"""Experiment configuration dataclass."""

from dataclasses import dataclass, field, asdict
from typing import Optional

import torch


@dataclass
class ExperimentConfig:
    # ── Dataset ──
    dataset: str = "nsir"
    split: str = "test"

    # ── LLM (local) ──
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # ── LLM (API) ──
    use_api: bool = False
    openai_api_key: Optional[str] = None
    api_model: str = "gpt-4.1-nano"

    # ── Embedding ──
    embed_model: str = "BAAI/bge-m3"

    # ── Optimization ──
    optimization_steps: int = 500
    lr: float = 0.001
    reg_weight: float = 0.2
    pos_weight: float = 1.0
    neg_weight: float = 1.0
    use_optimization: bool = True

    # ── Retrieval ──
    top_k: int = 1000
    k_eval: int = 100

    # ── Prompt ──
    prompt_version: str = "v1"

    # ── Paths ──
    cache_dir: str = "beir_cache"
    checkpoint_dir: str = "checkpoints"

    # ── Runtime ──
    use_gpu_index: bool = field(default_factory=lambda: torch.cuda.is_available())
    verbose: bool = True
    use_cache: bool = True
    resume: bool = True

    # ── Test / Debug ──
    max_corpus: Optional[int] = None
    max_queries: Optional[int] = None
    quick_test: bool = False

    # ── Mode ──
    analysis_mode: bool = False

    # ── Derived helpers ──
    @property
    def decompose_model(self) -> str:
        return self.api_model if self.use_api else self.llm_model

    @property
    def weight_config(self) -> dict:
        return {
            "reg_weight": self.reg_weight,
            "pos_weight": self.pos_weight,
            "neg_weight": self.neg_weight,
            "num_steps": self.optimization_steps,
            "lr": self.lr,
        }

    def to_dict(self) -> dict:
        return asdict(self)
