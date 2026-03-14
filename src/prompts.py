"""Prompt loader – reads prompt definitions from prompts/*.yaml files."""

import os
from functools import lru_cache
from typing import Tuple

import yaml

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
DEFAULT_PROMPT_VERSION = "v1"


@lru_cache(maxsize=None)
def _load_prompt_file(version: str) -> dict:
    path = os.path.join(_PROMPTS_DIR, f"{version}.yaml")
    if not os.path.exists(path):
        available = [
            f.replace(".yaml", "")
            for f in os.listdir(_PROMPTS_DIR)
            if f.endswith(".yaml")
        ]
        raise ValueError(
            f"Unknown prompt_version={version}. "
            f"Available: {available}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_prompt(prompt_version: str = None) -> Tuple[str, str]:
    """Return (system_prompt, user_template) for the given version."""
    v = prompt_version or DEFAULT_PROMPT_VERSION
    data = _load_prompt_file(v)
    return data["system"], data["user"]


def list_versions() -> list:
    """Return all available prompt version names."""
    if not os.path.isdir(_PROMPTS_DIR):
        return []
    return sorted(
        f.replace(".yaml", "")
        for f in os.listdir(_PROMPTS_DIR)
        if f.endswith(".yaml")
    )
