"""Shared text-generation helpers with explicit reasoning-output handling."""

from __future__ import annotations

import re
from typing import Any


_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", flags=re.IGNORECASE)
_SPECIAL_RE = re.compile(r"<\|.*?\|>|<｜.*?｜>")


def clean_instruction(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    text = _SPECIAL_RE.sub("", text).strip()
    text = text.strip('"').strip("'").strip()
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if paragraphs:
        text = paragraphs[-1]
    return text.strip()


def generate_instruction(
    *,
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    reasoning_mode: str,
) -> str:
    """Generate one deterministic instruction from a chat-style causal LM."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if reasoning_mode == "closed_think_prefill":
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt_text += "<think>\n\n</think>\n\n"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(
            next(model.parameters()).device
        )
        input_ids = inputs["input_ids"]
        kwargs = inputs
    elif reasoning_mode == "postprocess":
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(next(model.parameters()).device)
        kwargs = {"input_ids": input_ids}
    else:
        raise ValueError(f"Unsupported reasoning mode: {reasoning_mode}")

    with torch.no_grad():
        output = model.generate(
            **kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(
        output[0, input_ids.shape[-1] :], skip_special_tokens=False
    )
    return clean_instruction(decoded)
