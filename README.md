# DEO: Training-Free Direct Embedding Optimization for Negation-Aware Retrieval

[![arXiv](https://img.shields.io/badge/arXiv-2603.09185-b31b1b.svg)](https://arxiv.org/abs/2603.09185)

> **DEO: Training-Free Direct Embedding Optimization for Negation-Aware Retrieval**
>
> Taegyeong Lee<sup>1*</sup>, Jiwon Park<sup>2*</sup>, Seunghyun Hwang<sup>3*</sup>, JooYoung Jang<sup>1†</sup>
>
> <sup>*</sup>Equal contribution, <sup>†</sup>Corresponding author

LLM-driven query decomposition into positive/negative intents, followed by gradient-based embedding optimization for negation-aware document retrieval.

## Installation

**Python 3.9+** recommended.

```bash
pip install -r requirements.txt
```

## Setup

```bash
cp .env.sample .env
# Enter your OpenAI API key (optional if using local LLM only)
```

## Supported Datasets

### BEIR Datasets

Standard BEIR-format datasets are auto-downloaded when not found locally:

- `nsir`, `scifact`, `arguana`, `fiqa`, `trec-covid`, `nfcorpus`, `dbpedia-entity`, etc.

### NevIR (Negation-aware Vector IR)

- **Source**: HuggingFace (`orionweller/NevIR`)
- **Format**: Pairwise comparison (q1/q2, doc1/doc2)
- **Metric**: Pairwise Accuracy
- **Auto-download**: Yes

## Quick Start

```bash
python run_experiments.py
```

## Usage

### 1. Edit `run_experiments.py`

Defaults are defined in `ExperimentConfig` (`src/config.py`). Override only what you need:

```python
base_cfg = ExperimentConfig(
    openai_api_key=get_openai_key(),
    use_api=True,                   # True = OpenAI API, False = local LLM
    use_optimization=False,         # False = baseline (no optimization)
)

datasets = ["nsir"]                 # datasets to run (loop)
```

### 2. Choose datasets

```python
datasets = ["nsir"]                      # single dataset
datasets = ["nsir", "nevir"]             # multiple datasets (runs sequentially)
```

### 3. Configure weight experiments

Each dict in `weight_experiments` is a separate run. Add more dicts to sweep:

```python
weight_experiments = [
    # Baseline (no optimization)
    {"reg_weight": 0.0, "pos_weight": 0.0, "neg_weight": 0.0,
     "optimization_steps": 0, "use_optimization": False},

    # Optimized
    {"reg_weight": 0.2, "pos_weight": 1.0, "neg_weight": 1.0,
     "optimization_steps": 20, "use_optimization": True},
]
```

### 4. Run

```bash
python run_experiments.py
```

**Pipeline:**

1. Load / download dataset
2. Build corpus embeddings + FAISS index (cached)
3. LLM query decomposition (cached per model + prompt version)
4. Gradient-based embedding optimization (cached per weight config)
5. FAISS retrieval + BEIR evaluation
6. Save results to `checkpoints/results/`
7. Print weight config comparison table

### 5. Interrupt and resume

All intermediate results are cached. If interrupted (Ctrl+C), just run again — it picks up where it left off (`resume=True`).

## Configuration

All parameters are managed via `ExperimentConfig` dataclass in `src/config.py`.

| Category | Parameters |
|---|---|
| **LLM** | `use_api`, `llm_model`, `api_model`, `prompt_version` |
| **Optimization** | `optimization_steps`, `lr`, `reg_weight`, `pos_weight`, `neg_weight`, `use_optimization` |
| **Retrieval** | `embed_model`, `top_k`, `k_eval`, `use_gpu_index` |
| **Runtime** | `use_cache`, `resume`, `verbose`, `quick_test`, `max_queries`, `max_corpus` |

## Caching and Reproducibility

```
checkpoints/
  decompositions/   # Query decomposition results (per model/prompt)
  embeddings/       # Optimized query embeddings (per weight config)
  progress/         # Interrupt/resume progress state
  results/          # Final retrieval results and metrics
```

Experiments can be safely interrupted and resumed with `resume=True`.

## Evaluation Metrics

Standard BEIR metrics via the official evaluator:

- `NDCG@10`, `NDCG@K`
- `MAP@K`
- `Recall@K`
- `Precision@K`

Queries are also split into **neg group** (queries with negative intents) and **pos-only group** for separate evaluation.
