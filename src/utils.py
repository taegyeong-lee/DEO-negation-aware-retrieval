import os
import re
import json
import hashlib
import random
from typing import Dict, List
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def safe_json_extract(text: str) -> Dict[str, List[str]]:
    """Extract JSON object from LLM output text."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    blob = m.group(0) if m else text
    try:
        data = json.loads(blob)
    except Exception:
        return {}
    out = {}
    for k in ("positives", "negatives"):
        vals = data.get(k, [])
        if isinstance(vals, list):
            out[k] = [str(s).strip() for s in vals if isinstance(s, (str, int, float)) and str(s).strip()]
    return out


def get_query_hash(query: str) -> str:
    """Generate a short hash of a query string (for filenames)."""
    return hashlib.md5(query.encode()).hexdigest()[:12]


def get_weight_hash(reg_weight: float, pos_weight: float, neg_weight: float,
                    num_steps: int, lr: float) -> str:
    """Generate a deterministic string key for a weight configuration."""
    return f"r{reg_weight:.3f}_p{pos_weight:.3f}_n{neg_weight:.3f}_s{num_steps}_lr{lr:.4f}"


def _as_int_keys(d):
    """Convert metric dict keys like 'NDCG@10' to integer keys {10: value}."""
    out = {}
    for k, v in d.items():
        try:
            out[int(k)] = v
        except Exception:
            m = re.search(r"\d+", str(k))
            if m:
                out[int(m.group(0))] = v
    return out


def _get_metric(d, k):
    """Safely retrieve a metric value by key."""
    return d.get(k, float("nan"))


