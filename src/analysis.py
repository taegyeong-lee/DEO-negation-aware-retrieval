"""Result analysis – weight config comparison, optimization history plots, group metrics."""

import os
import json
import csv
import glob
from datetime import datetime
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from beir.retrieval.evaluation import EvaluateRetrieval

from .utils import get_query_hash, get_weight_hash, _as_int_keys, _get_metric


def _extract_decompose_model(params: dict, sep: str = "/") -> str:
    """Extract decompose model name from saved params dict."""
    default = "unknown"
    if params.get("use_api"):
        raw = params.get("api_model", default)
    else:
        raw = params.get("llm_model", default)
    return raw.replace("/", sep) if sep != "/" else raw


def compare_weight_configs(dataset: str = "nsir", checkpoint_dir: str = "./checkpoints"):
    """Compare performance across different weight configurations (grouped by model)."""

    results_pattern = f"{checkpoint_dir}/results/{dataset}_results_*.json"
    result_files = glob.glob(results_pattern)

    if not result_files:
        print(f"No result files found for dataset: {dataset}")
        return

    # Group results by (decompose_model, prompt_version, embed_model)
    grouped_results = {}

    for file in result_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        weight_config = data.get("weight_config", {})
        metrics = data.get("metrics", {})
        params = data.get("params", {})

        embed_model = params.get("embed_model", "unknown_model")
        decompose_model = _extract_decompose_model(params)

        prompt_version = params.get("prompt_version", "v1")

        entry = {
            "file": os.path.basename(file),
            "reg_weight": weight_config.get("reg_weight", "N/A"),
            "pos_weight": weight_config.get("pos_weight", "N/A"),
            "neg_weight": weight_config.get("neg_weight", "N/A"),
            "num_steps": weight_config.get("num_steps", "N/A"),
            "lr": weight_config.get("lr", "N/A"),
            "ndcg@10": metrics.get("ndcg@10", "N/A"),
            "ndcg@100": metrics.get("ndcg@100", "N/A"),
            "map@100": metrics.get("map@100", "N/A"),
        }

        grouped_results.setdefault((decompose_model, prompt_version, embed_model), []).append(entry)

    # Print grouped output
    for (decompose_model, prompt_version, embed_model), comparisons in grouped_results.items():
        print(f"\n{'=' * 110}")
        print(f"Weight Configuration Comparison for {dataset}")
        print(f"Prompt Version: {prompt_version} | Decompose Model: {decompose_model} | Embed Model: {embed_model}")
        print(f"{'=' * 110}")
        print(
            f"{'Reg':>6} {'Pos':>6} {'Neg':>6} {'Steps':>6} {'LR':>8} | "
            f"{'NDCG@10':>8} {'NDCG@100':>9} {'MAP@100':>8} | {'ΔMAP':>14}"
        )
        print(f"{'-' * 110}")

        # Sort by NDCG@10 descending
        comparisons.sort(
            key=lambda x: x.get("ndcg@10", 0) if isinstance(x.get("ndcg@10"), (int, float)) else 0,
            reverse=True
        )

        # Find baseline MAP (steps==0)
        baseline_map = None
        for comp in comparisons:
            if comp["num_steps"] == 0:
                baseline_map = comp.get("map@100")
                break

        for comp in comparisons:
            map_val = comp.get("map@100")

            if comp["num_steps"] == 0 or baseline_map is None:
                delta_str = "baseline"
            else:
                if not isinstance(map_val, (int, float)) or not isinstance(baseline_map, (int, float)):
                    delta_str = "N/A"
                else:
                    delta = map_val - baseline_map
                    pct = (delta / baseline_map) * 100 if baseline_map > 0 else 0.0
                    delta_str = f"{delta:+.4f} ({pct:+.2f}%)"

            print(
                f"{comp['reg_weight']:>6} {comp['pos_weight']:>6} {comp['neg_weight']:>6} "
                f"{comp['num_steps']:>6} {comp['lr']:>8} | "
                f"{comp['ndcg@10']:>8.4f} {comp['ndcg@100']:>9.4f} {comp['map@100']:>8.4f} | "
                f"{delta_str:>14}"
            )

        print(f"{'=' * 110}")


def plot_optimization_history(query: str, dataset: str = "nsir",
                              weight_configs: List[Dict] = None,
                              checkpoint_dir: str = "./checkpoints"):
    """Compare optimization loss histories across multiple weight configurations."""

    if weight_configs is None:
        weight_configs = [{"reg_weight": 0.2, "pos_weight": 1.0, "neg_weight": 1.0, "num_steps": 100, "lr": 0.001}]

    query_hash = get_query_hash(query)

    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(weight_configs)))

    for idx, config in enumerate(weight_configs):
        weight_str = get_weight_hash(
            config["reg_weight"], config["pos_weight"],
            config["neg_weight"], config["num_steps"], config["lr"]
        )

        filename = f"{dataset}_{query_hash}_{weight_str}_embedding.npz"
        filepath = os.path.join(checkpoint_dir, "embeddings", filename)

        if not os.path.exists(filepath):
            print(f"File not found for config {config}: {filepath}")
            continue

        # Load loss history from npz
        data = np.load(filepath)
        history = {}
        for key in data.files:
            if key.startswith("loss_history_"):
                loss_key = key.replace("loss_history_", "")
                history[loss_key] = data[key]

        label = f"r={config['reg_weight']}, p={config['pos_weight']}, n={config['neg_weight']}"

        if "total_losses" in history:
            axes[0, 0].plot(history["total_losses"], color=colors[idx], label=label)
        if "pos_losses" in history:
            axes[0, 1].plot(history["pos_losses"], color=colors[idx], label=label)
        if "deviation_losses" in history:
            axes[1, 0].plot(history["deviation_losses"], color=colors[idx], label=label)
        if "neg_losses" in history:
            axes[1, 1].plot(history["neg_losses"], color=colors[idx], label=label)

    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title("Positive Loss")
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Deviation Loss (Regularization)")
    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Negative Loss")
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Optimization History Comparison\nQuery: {query[:60]}...")
    plt.tight_layout()
    plt.show()


def analyze_results(dataset,
                    retriever,
                    queries,
                    qrels,
                    results_dir="./checkpoints/results",
                    out_dir="./checkpoints/analysis",
                    k_eval=100):
    """
    Read result JSONs from results_dir, compute overall / neg_group / pos_only_group
    metrics per (dataset, decompose_model, embed_model) group, and save as JSON/CSV.
    """

    json_dir = os.path.join(out_dir, "json")
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    result_files = [f for f in os.listdir(results_dir) if f.startswith(dataset) and f.endswith(".json")]
    if not result_files:
        print(f"[WARN] No result files found in {results_dir}")
        return

    all_groups = {}

    for file in result_files:
        path = os.path.join(results_dir, file)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        params_used = data.get("params", {})
        embed_model = params_used.get("embed_model", "unknown").replace("/", "-")
        decompose_model = _extract_decompose_model(params_used, sep="-")

        group_key = (dataset, decompose_model, embed_model)

        weight_config = data.get("weight_config", {})
        reg = weight_config.get("reg_weight", 0.0)
        pos = weight_config.get("pos_weight", 0.0)
        neg = weight_config.get("neg_weight", 0.0)
        steps = weight_config.get("num_steps", 0)
        lr = weight_config.get("lr", 0.0)
        key = f"r{reg:.3f}_p{pos:.3f}_n{neg:.3f}_s{steps}_lr{lr:.4f}"

        results = data.get("results", {})
        evaluator = EvaluateRetrieval(retriever, score_function="dot")

        # Overall evaluation
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [10, k_eval])
        ndcg = _as_int_keys(ndcg)
        _map = _as_int_keys(_map)
        recall = _as_int_keys(recall)
        precision = _as_int_keys(precision)

        metrics_all = {
            "overall": {
                "queries": len(results),
                "ndcg@10": _get_metric(ndcg, 10),
                f"ndcg@{k_eval}": _get_metric(ndcg, k_eval),
                f"map@{k_eval}": _get_metric(_map, k_eval),
                f"recall@{k_eval}": _get_metric(recall, k_eval),
                f"precision@{k_eval}": _get_metric(precision, k_eval)
            }
        }

        # Split queries into neg group vs pos-only group
        neg_qids, pos_only_qids = [], []
        for qid, qtext in queries.items():
            decomp = retriever.llm.checkpoint_manager.load_decomposition(
                qtext, dataset,
                decompose_model=(params_used.get("api_model") if params_used.get("use_api")
                                 else params_used.get("llm_model"))
            )
            if not decomp:
                continue
            if decomp.get("negatives"):
                neg_qids.append(qid)
            else:
                pos_only_qids.append(qid)

        results_neg = {qid: results[qid] for qid in neg_qids if qid in results}
        results_posonly = {qid: results[qid] for qid in pos_only_qids if qid in results}

        # Neg group metrics
        if results_neg:
            qids_with_labels = [qid for qid in results_neg if qid in qrels and len(qrels[qid]) > 0]
            if qids_with_labels:
                qrels_neg = {qid: qrels[qid] for qid in qids_with_labels}
                results_neg_filtered = {qid: results_neg[qid] for qid in qids_with_labels}

                ndcg_neg, map_neg, recall_neg, precision_neg = evaluator.evaluate(qrels_neg, results_neg_filtered, [10, k_eval])
                metrics_all["neg_group"] = {
                    "queries": len(results_neg_filtered),
                    "ndcg@10": _get_metric(_as_int_keys(ndcg_neg), 10),
                    f"ndcg@{k_eval}": _get_metric(_as_int_keys(ndcg_neg), k_eval),
                    f"map@{k_eval}": _get_metric(_as_int_keys(map_neg), k_eval),
                    f"recall@{k_eval}": _get_metric(_as_int_keys(recall_neg), k_eval),
                    f"precision@{k_eval}": _get_metric(_as_int_keys(precision_neg), k_eval)
                }

        # Pos-only group metrics
        if results_posonly:
            qids_with_labels = [qid for qid in results_posonly if qid in qrels and len(qrels[qid]) > 0]
            if qids_with_labels:
                qrels_pos = {qid: qrels[qid] for qid in qids_with_labels}
                results_pos_filtered = {qid: results_posonly[qid] for qid in qids_with_labels}

                ndcg_pos, map_pos, recall_pos, precision_pos = evaluator.evaluate(qrels_pos, results_pos_filtered, [10, k_eval])
                metrics_all["pos_only_group"] = {
                    "queries": len(results_pos_filtered),
                    "ndcg@10": _get_metric(_as_int_keys(ndcg_pos), 10),
                    f"ndcg@{k_eval}": _get_metric(_as_int_keys(ndcg_pos), k_eval),
                    f"map@{k_eval}": _get_metric(_as_int_keys(map_pos), k_eval),
                    f"recall@{k_eval}": _get_metric(_as_int_keys(recall_pos), k_eval),
                    f"precision@{k_eval}": _get_metric(_as_int_keys(precision_pos), k_eval)
                }

        metrics_all["tag"] = None

        all_groups.setdefault(group_key, {})[key] = {
            "weights": {"reg": reg, "pos": pos, "neg": neg, "steps": steps, "lr": lr},
            "metrics": metrics_all,
        }

    # Save per-group files
    for (dataset, decompose_model, embed_model), group_metrics in all_groups.items():
        # Tag baseline and best
        for key, entry in group_metrics.items():
            w = entry["weights"]
            if w["reg"] == 0 and w["pos"] == 0 and w["neg"] == 0 and w["steps"] == 0:
                entry["metrics"]["tag"] = "baseline"

        best_key = max(
            (k for k, v in group_metrics.items() if v["metrics"]["tag"] != "baseline"),
            key=lambda k: group_metrics[k]["metrics"]["overall"]["ndcg@10"],
            default=None
        )
        if best_key:
            group_metrics[best_key]["metrics"]["tag"] = "best"

        def sort_priority(item):
            _key, entry = item
            tag = entry["metrics"].get("tag", None)
            if tag == "baseline":
                return (0, 0)
            elif tag == "best":
                return (1, -entry["metrics"]["overall"]["ndcg@10"])
            else:
                return (2, -entry["metrics"]["overall"]["ndcg@10"])

        sorted_items = sorted(group_metrics.items(), key=sort_priority)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON output
        json_out = {
            "dataset": dataset,
            "decompose_model": decompose_model,
            "embed_model": embed_model,
            "metrics": {k: v for k, v in sorted_items}
        }
        json_path = os.path.join(json_dir, f"{dataset}_{decompose_model}_{embed_model}_analysis_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_out, f, ensure_ascii=False, indent=2)

        # CSV output
        csv_path = os.path.join(csv_dir, f"{dataset}_{decompose_model}_{embed_model}_analysis_{timestamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset", "decompose_model", "embed_model",
                "reg_weight", "pos_weight", "neg_weight", "num_steps", "lr",
                "group", "queries", "ndcg@10", f"ndcg@{k_eval}",
                f"map@{k_eval}", f"recall@{k_eval}", f"precision@{k_eval}",
                "tag"
            ])
            for key, entry in sorted_items:
                w = entry["weights"]
                tag = entry["metrics"]["tag"]
                for group, vals in entry["metrics"].items():
                    if group == "tag":
                        continue
                    writer.writerow([
                        dataset, decompose_model, embed_model,
                        w["reg"], w["pos"], w["neg"], w["steps"], w["lr"],
                        group, vals.get("queries", 0),
                        vals.get("ndcg@10", "nan"),
                        vals.get(f"ndcg@{k_eval}", "nan"),
                        vals.get(f"map@{k_eval}", "nan"),
                        vals.get(f"recall@{k_eval}", "nan"),
                        vals.get(f"precision@{k_eval}", "nan"),
                        tag
                    ])

        print(f"[Saved] JSON -> {json_path}")
        print(f"[Saved] CSV  -> {csv_path}")
