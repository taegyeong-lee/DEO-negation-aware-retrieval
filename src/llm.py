"""PolarityLLM – decomposes queries into positive/negative intents."""

from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from .config import ExperimentConfig
from .prompts import get_prompt
from .utils import safe_json_extract, get_query_hash
from .checkpoint import CheckpointManager

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class PolarityLLM:

    def __init__(self, cfg: ExperimentConfig, checkpoint_manager: CheckpointManager):
        self.cfg = cfg
        self.use_cache = cfg.use_cache
        self.checkpoint_manager = checkpoint_manager
        self.use_api = cfg.use_api
        self.prompt_version = cfg.prompt_version

        if self.use_api:
            if not OpenAI:
                raise ImportError("openai library not installed. pip install openai")
            if not cfg.openai_api_key:
                raise ValueError("OpenAI API key is required when use_api=True")

            self.client = OpenAI(api_key=cfg.openai_api_key)
            print(f"[PolarityLLM] Using OpenAI API: {cfg.api_model}")

            self.tokenizer = None
            self.model = None
            self.is_seq2seq = False
        else:
            print(f"[PolarityLLM] Using local model: {cfg.llm_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.llm_model, trust_remote_code=True
            )
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    cfg.llm_model, device_map="auto",
                    torch_dtype=torch.bfloat16, trust_remote_code=True
                )
                self.is_seq2seq = True
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.llm_model, device_map="auto",
                    torch_dtype=torch.bfloat16, trust_remote_code=True
                )
                self.is_seq2seq = False

        self.gen_kwargs = dict(
            max_new_tokens=192, temperature=0.0, top_p=1.0, do_sample=False
        )

    @property
    def decompose_model(self) -> str:
        return self.cfg.decompose_model

    def _call_openai_api(self, query: str) -> str:
        prompt_sys, prompt_user = get_prompt(self.prompt_version)
        try:
            response = self.client.chat.completions.create(
                model=self.cfg.api_model,
                messages=[
                    {"role": "system", "content": prompt_sys},
                    {"role": "user", "content": prompt_user.format(query=query)},
                ],
                temperature=self.gen_kwargs["temperature"],
                top_p=self.gen_kwargs["top_p"],
                max_tokens=self.gen_kwargs["max_new_tokens"],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Error] OpenAI API call failed: {e}")
            return '{"positives": ["' + query + '"], "negatives": []}'

    def _build_inputs(self, query: str):
        prompt_sys, prompt_user = get_prompt(self.prompt_version)

        if self.is_seq2seq:
            prompt = f"{prompt_sys}\n{prompt_user.format(query=query)}\nJSON:"
            return self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        msgs = [
            {"role": "system", "content": prompt_sys},
            {"role": "user", "content": prompt_user.format(query=query)},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            return self.tokenizer(text, return_tensors="pt").to(self.model.device)

        prompt = f"[SYSTEM] {prompt_sys}\n[USER] {prompt_user.format(query=query)}\n[ASSISTANT]"
        return self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

    def expand(self, query: str, dataset: str = "default") -> Dict[str, List[str]]:
        """Decompose query into positive/negative intents (cached by model, not weights)."""
        decompose_model = self.decompose_model

        # Check cache
        if self.use_cache and self.checkpoint_manager:
            cached = self.checkpoint_manager.load_decomposition(
                query, dataset, decompose_model=decompose_model,
                prompt_version=self.prompt_version,
            )
            if cached:
                print(f"  [Cache Hit] Loaded decomposition for query using {decompose_model}")
                return cached

        # Run LLM
        if self.use_api:
            text = self._call_openai_api(query)
        else:
            inputs = self._build_inputs(query)
            with torch.no_grad():
                out = self.model.generate(**inputs, **self.gen_kwargs)
            gen_tokens = out[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        data = safe_json_extract(text)
        if not data.get("positives"):
            data["positives"] = [query]
        if "negatives" not in data:
            data["negatives"] = []

        # Save to cache
        if self.checkpoint_manager:
            prompt_sys, _ = get_prompt(self.prompt_version)
            self.checkpoint_manager.save_decomposition(
                query, data, dataset,
                decompose_model=decompose_model,
                prompt_sys=prompt_sys,
                prompt_version=self.prompt_version,
            )
            print(f"  [Saved] Decomposition hash={get_query_hash(query)} model={decompose_model}")

        return data
