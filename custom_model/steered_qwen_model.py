from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm.auto import tqdm

from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import ModelResponse
from lighteval.metrics.utils.metric_utils import SamplingMethod
from lighteval.tasks.requests import Doc
from lighteval.utils.cache_management import SampleCache, cached

from src.hidden_state.config import load_runtime_config
from src.hidden_state.extraction import build_vanilla_bundle_for_question
from src.hidden_state.generation import generate_samples
from src.hidden_state.logprob import _sum_continuation_logprob, rolling_loglikelihood
from src.hidden_state.modeling import load_model_and_tokenizer
from src.hidden_state.prompting import infer_raw_question_from_query, maybe_apply_chat_template
from src.hidden_state.steering_core import load_steering_bundle, select_last_k_layers


class SteeredQwenLightevalModel(LightevalModel):
    def __init__(self, config, env_config=None):
        self.config = config
        self.runtime = load_runtime_config()
        self.loaded = load_model_and_tokenizer(config.model_name, precision=self.runtime.precision)
        self._tokenizer = self.loaded.tokenizer
        runtime_signature = hashlib.sha256(
            json.dumps(asdict(self.runtime), sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()[:16]
        cache_config = config.model_copy(
            update={
                "cache_dir": str(
                    Path(os.path.expanduser(config.cache_dir)) / f"steering_{runtime_signature}"
                )
            }
        )
        self._cache = SampleCache(cache_config)
        self._tgs_bundle = None
        if self.runtime.steering_method == "tgs" and self.runtime.tgs_vector_path:
            self._tgs_bundle = select_last_k_layers(load_steering_bundle(self.runtime.tgs_vector_path), self.runtime.k)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        return int(getattr(self.loaded.model.config, "max_position_embeddings", 4096))

    def _infer_benchmark_style(self, doc: Doc) -> str:
        if doc.specific and "benchmark_style" in doc.specific:
            return str(doc.specific["benchmark_style"])
        task_name = getattr(doc, "task_name", "") or ""
        if "math_stock_semantics" in task_name:
            return "math_stock"
        if "math_greedy_steering" in task_name:
            return "math_greedy"
        return "gsm8k"

    def _get_raw_question(self, doc: Doc) -> str:
        if doc.specific and "raw_question" in doc.specific:
            return str(doc.specific["raw_question"])
        return infer_raw_question_from_query(doc.query)

    def _select_steering_bundle(self, doc: Doc):
        method = self.runtime.steering_method
        if method == "none":
            return None
        if method == "tgs":
            return self._tgs_bundle
        if method == "vanilla":
            question = self._get_raw_question(doc)
            benchmark_style = self._infer_benchmark_style(doc)
            full_bundle = build_vanilla_bundle_for_question(
                self.loaded,
                question,
                benchmark_style=benchmark_style,
                use_chat_template=self.runtime.use_chat_template,
                system_prompt=self.runtime.system_prompt,
            )
            return select_last_k_layers(full_bundle, self.runtime.k)
        raise ValueError(f"Unsupported steering method: {method}")

    def _prepare_prompt(self, doc: Doc) -> str:
        return maybe_apply_chat_template(
            self.loaded.tokenizer,
            doc.query,
            use_chat_template=self.runtime.use_chat_template,
            system_prompt=self.runtime.system_prompt,
        )

    def _progress_desc(self, docs: list[Doc], phase: str) -> str:
        task_names = sorted({getattr(doc, "task_name", "") or "unknown" for doc in docs})
        label = task_names[0] if len(task_names) == 1 else f"{len(task_names)} tasks"
        return f"{phase} [{self.runtime.steering_method}] {label}"

    def _progress_interval(self, total: int) -> int:
        return max(1, total // 20)

    def _maybe_print_progress(self, *, phase: str, current: int, total: int) -> None:
        if total <= 1:
            return
        interval = self._progress_interval(total)
        if current == 1 or current == total or current % interval == 0:
            print(f"{phase}: {current}/{total}", flush=True)

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        responses = []
        phase = self._progress_desc(docs, "Greedy generation")
        iterator = tqdm(
            docs,
            desc=phase,
            disable=self.disable_tqdm,
        )
        total_docs = len(docs)
        for index, doc in enumerate(iterator, start=1):
            prompt_text = self._prepare_prompt(doc)
            steering_bundle = self._select_steering_bundle(doc)
            max_new_tokens = int(doc.generation_size or self.runtime.default_max_new_tokens)
            num_samples = int(getattr(doc, "num_samples", 1) or 1)
            texts, output_token_lists, input_tokens = generate_samples(
                self.loaded,
                prompt_text,
                steering_bundle=steering_bundle,
                alpha=self.runtime.alpha,
                norm_preserving=self.runtime.norm_preserving,
                max_new_tokens=max_new_tokens,
                stop_sequences=doc.stop_sequences,
                num_samples=num_samples,
                temperature=self.runtime.temperature_for_sampling,
                top_p=self.runtime.top_p_for_sampling,
            )
            responses.append(
                ModelResponse(
                    input=prompt_text,
                    input_tokens=input_tokens,
                    text=texts,
                    output_tokens=output_token_lists,
                )
            )
            self._maybe_print_progress(phase=phase, current=index, total=total_docs)
        return responses

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        responses = []
        phase = self._progress_desc(docs, "Loglikelihood")
        iterator = tqdm(
            docs,
            desc=phase,
            disable=self.disable_tqdm,
        )
        total_docs = len(docs)
        for index, doc in enumerate(iterator, start=1):
            context = self._prepare_prompt(doc)
            logprobs = []
            argmax_eq_gold = []
            output_tokens = []
            input_tokens_ref = []
            for choice in doc.choices:
                lp, input_tokens, cont_tokens, eq = _sum_continuation_logprob(self.loaded, context, choice)
                input_tokens_ref = input_tokens
                output_tokens.append(cont_tokens)
                logprobs.append(lp)
                argmax_eq_gold.append(eq)
            responses.append(
                ModelResponse(
                    input=context,
                    input_tokens=input_tokens_ref,
                    logprobs=logprobs,
                    argmax_logits_eq_gold=argmax_eq_gold,
                    output_tokens=output_tokens,
                )
            )
            self._maybe_print_progress(phase=phase, current=index, total=total_docs)
        return responses

    @cached(SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        responses = []
        phase = self._progress_desc(docs, "Rolling loglikelihood")
        iterator = tqdm(
            docs,
            desc=phase,
            disable=self.disable_tqdm,
        )
        total_docs = len(docs)
        for index, doc in enumerate(iterator, start=1):
            context = self._prepare_prompt(doc)
            lp, input_tokens = rolling_loglikelihood(self.loaded, context)
            responses.append(
                ModelResponse(
                    input=context,
                    input_tokens=input_tokens,
                    logprobs=[lp],
                )
            )
            self._maybe_print_progress(phase=phase, current=index, total=total_docs)
        return responses
