"""Main entry point – orchestrates data loading, retrieval, and evaluation."""

from beir.retrieval.evaluation import EvaluateRetrieval

from .config import ExperimentConfig
from .utils import set_seed, _as_int_keys, _get_metric
from .retriever import OptimizedPolarityRetriever
from .checkpoint import CheckpointManager
from .data_loader import load_dataset_auto, is_custom_dataset
from .evaluator import evaluate_custom_dataset
from .analysis import analyze_results


def main(cfg: ExperimentConfig):
    # ── Analysis mode ──
    if cfg.analysis_mode:
        return _run_analysis(cfg)

    # ── Training / Retrieval mode ──
    return _run_retrieval(cfg)


def _run_analysis(cfg: ExperimentConfig):
    results_dir = f"{cfg.checkpoint_dir}/results"
    out_dir = f"{cfg.checkpoint_dir}/analysis"

    print(f"[Analysis Mode] Dataset={cfg.dataset}")

    _, queries, qrels, _ = load_dataset_auto(
        cfg.dataset, split=cfg.split, data_root="beir"
    )

    checkpoint_manager = CheckpointManager(cfg.checkpoint_dir)

    # Lightweight stub – analysis only reads cached decompositions
    class _AnalysisStub:
        def __init__(self):
            self.llm = type("obj", (), {
                "use_api": cfg.use_api,
                "api_model": cfg.api_model,
                "model_name": cfg.llm_model,
                "checkpoint_manager": checkpoint_manager,
            })()
            self.embedder = type("obj", (), {"model_name": "dummy-embedder"})()
            self.dataset_name = cfg.dataset

    analyze_results(
        dataset=cfg.dataset,
        retriever=_AnalysisStub(),
        queries=queries,
        qrels=qrels,
        results_dir=results_dir,
        out_dir=out_dir,
        k_eval=cfg.k_eval,
    )


def _run_retrieval(cfg: ExperimentConfig):
    set_seed(42)

    if cfg.quick_test:
        cfg.optimization_steps = min(cfg.optimization_steps, 50)
        cfg.top_k = min(cfg.top_k, 50)
        if cfg.max_corpus is None:
            cfg.max_corpus = 1000
        if cfg.max_queries is None:
            cfg.max_queries = 50
        cfg.resume = False

    _print_banner(cfg)

    # 1) Load data
    corpus, queries, qrels, dataset_extra = load_dataset_auto(
        cfg.dataset, split=cfg.split, data_root="beir"
    )
    print(f"[Data] Loaded {len(corpus)} documents and {len(queries)} queries")

    if cfg.max_corpus is not None or cfg.max_queries is not None:
        corpus, queries, qrels = _apply_limits(corpus, queries, qrels, cfg)

    # 2) Build retriever & index
    retriever = OptimizedPolarityRetriever(cfg)
    retriever.build_index(cfg.dataset, corpus)

    # 3) Search & evaluate
    print("\n[Retrieval] Starting retrieval process...")
    print("[Info] Progress will be saved automatically. You can safely interrupt and resume.\n")

    try:
        results = retriever.search(corpus, queries)

        evaluator = EvaluateRetrieval(retriever, score_function="dot")
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [10, cfg.k_eval])

        ndcg = _as_int_keys(ndcg)
        _map = _as_int_keys(_map)
        recall = _as_int_keys(recall)
        precision = _as_int_keys(precision)

        metrics = {
            "ndcg@10": _get_metric(ndcg, 10),
            f"ndcg@{cfg.k_eval}": _get_metric(ndcg, cfg.k_eval),
            f"map@{cfg.k_eval}": _get_metric(_map, cfg.k_eval),
            f"recall@{cfg.k_eval}": _get_metric(recall, cfg.k_eval),
            f"precision@{cfg.k_eval}": _get_metric(precision, cfg.k_eval),
        }

        # Custom dataset evaluation
        if is_custom_dataset(cfg.dataset) and dataset_extra is not None:
            custom_metrics = evaluate_custom_dataset(
                cfg.dataset, results, dataset_extra
            )
            metrics.update(custom_metrics)

        retriever.checkpoint_manager.save_final_results(
            cfg.dataset, results, metrics, cfg.weight_config, cfg.to_dict()
        )

        _print_results(cfg, ndcg, _map, recall, precision, metrics)
        return metrics

    except KeyboardInterrupt:
        print("\n[Interrupt] Process interrupted by user.")
        print("[Info] Progress has been saved. Run again with resume=True to continue.")
        return None
    except Exception as e:
        print(f"\n[Error] An error occurred: {e}")
        print("[Info] Progress has been saved. Run again with resume=True to continue.")
        raise


# ── Helpers ──

def _print_banner(cfg: ExperimentConfig):
    print(f"\n{'='*60}")
    print(f"Optimized Polarity-Based Retrieval System")
    print(f"{'='*60}")
    print(f"Dataset: {cfg.dataset}")
    print(f"Prompt Version: {cfg.prompt_version}")
    if cfg.use_api:
        print(f"LLM: OpenAI API ({cfg.api_model})")
    else:
        print(f"LLM Model: {cfg.llm_model}")
    print(f"Embedding Model: {cfg.embed_model}")
    print(f"Optimization Steps: {cfg.optimization_steps}")
    print(f"Learning Rate: {cfg.lr}")
    print(f"Reg/Pos/Neg Weight: {cfg.reg_weight}/{cfg.pos_weight}/{cfg.neg_weight}")
    print(f"Cache: {cfg.use_cache} | Resume: {cfg.resume}")
    print(f"{'='*60}\n")


def _apply_limits(corpus, queries, qrels, cfg: ExperimentConfig):
    if cfg.max_corpus is not None:
        corpus = dict(list(corpus.items())[:cfg.max_corpus])
        print(f"[Limit] Using only {len(corpus)} documents")
    if cfg.max_queries is not None:
        queries = dict(list(queries.items())[:cfg.max_queries])
        qrels = {qid: rel for qid, rel in qrels.items() if qid in queries}
        print(f"[Limit] Using only {len(queries)} queries")
    return corpus, queries, qrels


def _print_results(cfg, ndcg, _map, recall, precision, metrics):
    k = cfg.k_eval
    print(f"\n{'='*60}")
    print(f"Results for {cfg.dataset} ({cfg.split} split)")
    print(f"Weight Config: reg={cfg.reg_weight}, pos={cfg.pos_weight}, neg={cfg.neg_weight}")
    print(f"{'='*60}")
    print(f"NDCG@10:        {ndcg[10]:.4f}")
    print(f"NDCG@{k}:      {ndcg[k]:.4f}")
    print(f"MAP@{k}:       {_map[k]:.4f}")
    print(f"Recall@{k}:    {recall[k]:.4f}")
    print(f"Precision@{k}: {precision[k]:.4f}")

    # Custom metrics
    custom_keys = [key for key in metrics if key not in {
        "ndcg@10", f"ndcg@{k}", f"map@{k}", f"recall@{k}", f"precision@{k}"
    }]
    if custom_keys:
        print(f"{'-'*60}")
        print(f"[{cfg.dataset.upper()} Custom Metrics]")
        for key in custom_keys:
            v = metrics[key]
            print(f"  {key}: {v:.4f}" if isinstance(v, float) else f"  {key}: {v}")

    print(f"{'='*60}\n")


