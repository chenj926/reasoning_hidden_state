from __future__ import annotations

from typing import Iterable

import torch

from .modeling import LoadedModelBundle


def _sum_continuation_logprob(bundle: LoadedModelBundle, context_text: str, continuation_text: str) -> tuple[float, list[int], list[int], bool]:
    tokenizer = bundle.tokenizer
    model = bundle.model
    full_text = context_text + continuation_text
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bundle.device)
    ctx_ids = tokenizer(context_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bundle.device)
    cont_ids = tokenizer(continuation_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bundle.device)

    with torch.inference_mode():
        logits = model(full_ids, use_cache=False, return_dict=True).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    ctx_len = int(ctx_ids.shape[1])
    cont_len = int(cont_ids.shape[1])
    if cont_len == 0:
        return 0.0, ctx_ids[0].detach().cpu().tolist(), [], True

    start = max(ctx_len - 1, 0)
    end = start + cont_len
    continuation_lp = token_log_probs[0, start:end]
    greedy_ids = shift_logits.argmax(dim=-1)[0, start:end]
    gold_ids = shift_labels[0, start:end]
    argmax_eq_gold = bool(torch.equal(greedy_ids, gold_ids))
    return (
        float(continuation_lp.sum().item()),
        ctx_ids[0].detach().cpu().tolist(),
        cont_ids[0].detach().cpu().tolist(),
        argmax_eq_gold,
    )


def rolling_loglikelihood(bundle: LoadedModelBundle, text: str) -> tuple[float, list[int]]:
    tokenizer = bundle.tokenizer
    model = bundle.model
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bundle.device)
    if ids.shape[1] <= 1:
        return 0.0, ids[0].detach().cpu().tolist()
    with torch.inference_mode():
        logits = model(ids, use_cache=False, return_dict=True).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs[0].sum().item()), ids[0].detach().cpu().tolist()
