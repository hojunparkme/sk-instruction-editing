"""Prompt templates recovered from the two reported experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Method(str, Enum):
    SIMPLE = "simple"
    LLM_ONLY = "llm_only"
    SK_FILTER = "sk_filter"
    SK_LLM = "sk_llm"


@dataclass(frozen=True)
class EditorPromptConfig:
    name: str
    scene_prompt: str
    system_prompt: str
    reasoning_mode: str
    max_new_tokens: int


FLUX_PROMPTS = EditorPromptConfig(
    name="FLUX Kontext",
    scene_prompt=(
        "Describe this image in 3–5 sentences. Include:\n"
        "1. All visible people and their clothing, accessories, and actions\n"
        "2. All visible objects, vehicles, and surfaces\n"
        "3. Lighting, atmosphere, and overall mood\n"
        "Be specific and concrete. Avoid vague language."
    ),
    system_prompt=(
        "You are an expert at writing image editing instructions for FLUX Kontext.\n"
        "Output ONLY the final instruction text — no analysis, no JSON, no preamble, no reasoning."
    ),
    reasoning_mode="postprocess",
    max_new_tokens=512,
)

IP2P_PROMPTS = EditorPromptConfig(
    name="InstructPix2Pix",
    scene_prompt=(
        "Describe this image in detail. Include visible objects, people, weather, "
        "lighting, and atmosphere."
    ),
    system_prompt=(
        "You are an expert at writing image editing instructions for InstructPix2Pix.\n"
        "Output ONLY the final instruction text — no analysis, no JSON, no preamble, no reasoning."
    ),
    reasoning_mode="closed_think_prefill",
    max_new_tokens=768,
)


def build_instruction_prompt(
    method: Method,
    *,
    scene: str,
    request: str,
    cues: list[str] | tuple[str, ...] = (),
    editor: str = "flux",
) -> str:
    if method is Method.SIMPLE:
        return request

    chunks = [
        f"Image scene description:\n{scene}",
        f"User editing request:\n{request}",
    ]
    if method in {Method.SK_FILTER, Method.SK_LLM}:
        chunks.append(f"KG candidates:\n{', '.join(cues)}")

    rules = ["Write ONE image editing instruction:", "- 2–3 sentences maximum."]
    if method is Method.LLM_ONLY:
        rules.append("- Only reference objects already visible in the scene description.")
    elif method is Method.SK_FILTER:
        rules.append(
            "- Only use KG cues that match objects VISIBLE in the scene description above."
        )
        if editor.lower() == "flux":
            rules.append(
                "- Do NOT use cues referencing elements absent from the scene "
                "(e.g. no road/asphalt visible → never use 'wet asphalt', 'tire spray', "
                "'road puddles', or 'lane markings')."
            )
        else:
            rules.append("- Do NOT use cues referencing elements absent from the scene.")
    elif method is Method.SK_LLM:
        rules.append("- Use the KG cues to enrich the editing instruction.")

    rules.extend(
        [
            "- Do NOT include reasoning — just the instruction.",
            "Return only the instruction text.",
        ]
    )
    chunks.append("\n".join(rules))
    return "\n\n".join(chunks)
