"""Architecture-aware hooks, LoRA loading, and temporary activation capture."""
from __future__ import annotations
from contextlib import contextmanager
import gc
import os
import re

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .common import adapter_dir, resolve_base, verify_adapter_identity

def load_model(cfg, family, condition="base", checkpoint="final"):
    name, rev = resolve_base(cfg, family)
    dtype = torch.bfloat16 if cfg["runtime"]["dtype"] == "bfloat16" else torch.float32
    tok = AutoTokenizer.from_pretrained(name, revision=rev, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=rev, torch_dtype=dtype,
        attn_implementation=cfg["runtime"]["attention"], device_map="auto",
        low_cpu_mem_usage=True,
    )
    if condition != "base":
        verify_adapter_identity(cfg, family, condition)
        model = PeftModel.from_pretrained(model, str(adapter_dir(cfg, family, condition, checkpoint)))
    model.eval()
    return model, tok

def blocks(model):
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    for attribute in ("model.layers", "model.model.layers", "transformer.h"):
        obj = base
        try:
            for part in attribute.split("."):
                obj = getattr(obj, part)
            if len(obj) and hasattr(obj[0], "mlp"):
                return list(obj)
        except (AttributeError, TypeError, IndexError):
            continue
    raise TypeError("Unsupported block layout; expected model.layers[*].mlp")

def site_module(model, layer, site):
    block = blocks(model)[layer]
    if site == "residual":
        return block
    if site == "mlp":
        return block.mlp
    raise ValueError(site)

def tensor_output(value):
    return value[0] if isinstance(value, tuple) else value

def replace_tensor_output(original, tensor):
    if isinstance(original, tuple):
        return (tensor, *original[1:])
    return tensor

def encode(tokenizer, texts, device, max_length):
    return tokenizer(texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=max_length).to(device)

def pool(hidden, mask, how):
    lengths = mask.long().sum(-1)
    if how == "mean":
        return (hidden.float() * mask.unsqueeze(-1)).sum(1) / lengths.clamp_min(1).unsqueeze(-1)
    if how == "final":
        indices = mask.long().cumsum(-1).argmax(-1) if False else mask.size(1)-1
        # Left padding is required; last token is the final nonpadding token.
        if not torch.all(mask[:, -1] == 1):
            raise ValueError("Final-token extraction requires left-padded inputs")
        return hidden[:, indices].float()
    raise ValueError(how)

@contextmanager
def capture(model, layer, site, mask, pooling):
    state = {}
    def hook(_module, _inputs, output):
        state["pooled"] = pool(tensor_output(output).detach(), mask, pooling)
    handle = site_module(model, layer, site).register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()

@torch.inference_mode()
def pooled_batches(model, tokenizer, texts, layer, site, pooling, batch_size=4, max_length=512):
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        inputs = encode(tokenizer, batch, model.device, max_length)
        with capture(model, layer, site, inputs["attention_mask"], pooling) as state:
            model(**inputs, use_cache=False)
        yield state["pooled"].detach()
        del inputs, state

@contextmanager
def steer(model, layer, site, vector, strength, token_policy="prompt_last"):
    """Modify exactly the final prompt token during prefill; optionally every decode token."""
    vec = vector.float()
    def hook(_module, _inputs, output):
        hidden = tensor_output(output)
        # In left-padded prefill, index -1 corresponds to each example's last prompt token.
        # At cached decoding steps the sequence dimension is one.
        if token_policy == "prompt_last" and hidden.shape[1] == 1:
            return output
        shifted = hidden.clone()
        delta = (strength * vec).to(hidden.device, dtype=hidden.dtype)
        shifted[:, -1, :] += delta
        return replace_tensor_output(output, shifted)
    handle = site_module(model, layer, site).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

@torch.inference_mode()
def continuation_scores(model, tokenizer, prompts, supporting, opposing, layer, site,
                        direction, strength, policy, max_length=512):
    """Length-mean token log likelihood of each candidate conditioned on the prompt."""
    scores = []
    for prompt, pos, neg in zip(prompts, supporting, opposing):
        result = []
        for option in (pos, neg):
            # Score token transitions of continuation only, respecting the exact text prefix.
            ids_p = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            ids_f = tokenizer(prompt + option, add_special_tokens=True)["input_ids"]
            if ids_f[:len(ids_p)] != ids_p:
                # Prefix tokenization differs: score using explicit concatenation.
                ids_f = ids_p + tokenizer(option, add_special_tokens=False)["input_ids"]
            if len(ids_f) > max_length:
                raise ValueError("Candidate exceeds maximum scoring length")
            x = torch.tensor([ids_f], dtype=torch.long, device=model.device)
            mask = torch.ones_like(x)
            with steer(model, layer, site, direction, strength, policy):
                out = model(input_ids=x, attention_mask=mask, use_cache=False)
            start = len(ids_p)
            target = x[:, start:]
            logits = out.logits[:, start-1:-1].float().log_softmax(-1)
            score = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean().item()
            result.append(score)
        scores.append(result[0] - result[1])
    return scores

def unload(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
