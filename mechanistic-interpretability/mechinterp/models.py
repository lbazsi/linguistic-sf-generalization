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
        indices = mask.size(1) - 1
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
                        direction, strength, policy, max_length=512, batch_size=4):
    """Length-normalized conditional log-probability gaps in small GPU batches.

    A model-generated response is never sampled and no API judge is contacted.
    Candidate tokens are appended after the *fixed tokenized prefix*.  Paired
    candidates receive the same intervention; all model outputs are transient.
    """
    if not (len(prompts)==len(supporting)==len(opposing)):
        raise ValueError("Mismatched prompt and option counts")
    if policy not in ("prompt_last","generation"):
        raise ValueError(policy)
    all_scores=[]
    pad=tokenizer.pad_token_id
    if pad is None:raise ValueError("Tokenizer padding token not configured")
    for start_batch in range(0,len(prompts),batch_size):
        pairs=[]
        for prompt,pos,neg in zip(prompts[start_batch:start_batch+batch_size],
                                   supporting[start_batch:start_batch+batch_size],
                                   opposing[start_batch:start_batch+batch_size]):
            ids=tokenizer(prompt,add_special_tokens=True)["input_ids"]
            if len(ids)<1:raise ValueError("Empty evaluation prefix")
            for option in (pos,neg):
                # Separate option encoding respects the causal prefix used at inference.
                tail=tokenizer(option,add_special_tokens=False)["input_ids"]
                if not tail:raise ValueError("Empty scoring continuation")
                whole=ids+tail
                if len(whole)>max_length:
                    raise ValueError("Candidate exceeds maximum scoring length")
                pairs.append((whole,len(ids)))
        width=max(len(seq) for seq,_ in pairs)
        batch_inputs=torch.full((len(pairs),width),pad,dtype=torch.long,device=model.device)
        mask=torch.zeros_like(batch_inputs)
        offsets=[]
        scored=torch.zeros_like(batch_inputs,dtype=torch.bool)
        for i,(seq,prefix_length) in enumerate(pairs):
            left=width-len(seq)
            batch_inputs[i,left:]=torch.tensor(seq,device=model.device)
            mask[i,left:]=1
            last_prompt=left+prefix_length-1
            offsets.append(last_prompt)
            scored[i,last_prompt+1:]=True
        prompt_positions=torch.tensor(offsets,device=model.device)
        vector=(strength*direction).to(device=model.device,dtype=next(model.parameters()).dtype)

        def hook(_module,_inputs,output):
            hidden=tensor_output(output)
            changed=hidden.clone()
            row_ids=torch.arange(len(pairs),device=hidden.device)
            if policy=="prompt_last":
                changed[row_ids,prompt_positions,:]+=vector
            else:
                index=torch.arange(width,device=hidden.device)[None,:]
                region=(index>=prompt_positions[:,None]) & (mask.bool())
                changed+=region.unsqueeze(-1).to(changed.dtype)*vector
            return replace_tensor_output(output,changed)

        handle=site_module(model,layer,site).register_forward_hook(hook)
        try:
            result=model(input_ids=batch_inputs,attention_mask=mask,use_cache=False)
        finally:
            handle.remove()
        logits=result.logits[:,:-1].float()
        target=batch_inputs[:,1:]
        scores=(logits.gather(-1,target.unsqueeze(-1)).squeeze(-1) -
                torch.logsumexp(logits,dim=-1))
        token_mask=scored[:,1:]
        means=(scores*token_mask).sum(-1)/token_mask.sum(-1).clamp_min(1)
        values=means.detach().tolist()
        all_scores.extend(values[i]-values[i+1] for i in range(0,len(values),2))
        del result,logits,scores,means,batch_inputs,mask
    return all_scores

def unload(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
