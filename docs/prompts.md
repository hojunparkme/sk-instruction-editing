# Prompt templates

This document separates the two editor-specific inference configurations used in the archived experiments. The exact generated instructions are included in `results/raw/`.

## 1. FLUX Kontext configuration

### Scene-description prompt

```text
Describe this image in 3–5 sentences. Include:
1. All visible people and their clothing, accessories, and actions
2. All visible objects, vehicles, and surfaces
3. Lighting, atmosphere, and overall mood
Be specific and concrete. Avoid vague language.
```

### DeepSeek system prompt

```text
You are an expert at writing image editing instructions for FLUX Kontext.
Output ONLY the final instruction text — no analysis, no JSON, no preamble, no reasoning.
```

The FLUX implementation uses greedy decoding and removes `<think>...</think>` and model-control tags during output post-processing.

### SK+LLM

```text
Image scene description:
{scene}

User editing request:
{request}

KG candidates:
{comma-separated retrieved cues}

Write ONE image editing instruction:
- 2–3 sentences maximum.
- Use the KG cues to enrich the editing instruction.
- Do NOT include reasoning — just the instruction.
Return only the instruction text.
```

### SK+Filter

```text
Image scene description:
{scene}

User editing request:
{request}

KG candidates:
{comma-separated retrieved cues}

Write ONE image editing instruction:
- 2–3 sentences maximum.
- Only use KG cues that match objects VISIBLE in the scene description above.
- Do NOT use cues referencing elements absent from the scene (e.g. no road/asphalt visible → never use 'wet asphalt', 'tire spray', 'road puddles', or 'lane markings').
- Do NOT include reasoning — just the instruction.
Return only the instruction text.
```

### LLM-only

```text
Image scene description:
{scene}

User editing request:
{request}

Write ONE image editing instruction:
- 2–3 sentences maximum.
- Only reference objects already visible in the scene description.
- Do NOT include reasoning — just the instruction.
Return only the instruction text.
```

## 2. InstructPix2Pix configuration

### Scene-description prompt

```text
Describe this image in detail. Include visible objects, people, weather, lighting, and atmosphere.
```

This prompt is run with the LLaVA model distributed as part of the official MGIE setup.

### DeepSeek system prompt

```text
You are an expert at writing image editing instructions for InstructPix2Pix.
Output ONLY the final instruction text — no analysis, no JSON, no preamble, no reasoning.
```

The IP2P implementation uses greedy decoding and appends the following closed reasoning block before generation:

```text
<think>

</think>

```

The SK+LLM and LLM-only user prompts follow the same logical templates as the FLUX configuration. The IP2P SK+Filter prompt uses the shorter rule below instead of the FLUX-specific road/asphalt example:

```text
- Only use KG cues that match objects VISIBLE in the scene description above.
- Do NOT use cues referencing elements absent from the scene.
```

## 3. Simple baseline

For both backbones, the raw user request is passed to the editor verbatim. No instruction-generation model is called.

## 4. Repository cue retrieval

The released repository is nested under five groups:

```text
condition / environment / season / time_of_day / weather
```

Within each entry, positive cues are organized into typed slots. Retrieval is deterministic and collects at most six cues per slot and fifty cues in total.

The FLUX reference implementation matches entry names and aliases against the lowercased user request. The archived IP2P workspace stored an author-reviewed target-condition label for every sample; the cleaned reference script resolves that label against the released nested repository and also reports when request-based matching selects a different set of entries.

## 5. Offline repository-construction prompt

The following prompt was recovered from the project history. It produced an initial cue vocabulary that was subsequently reviewed, expanded, deduplicated, normalized, and reorganized by the authors. It should not be interpreted as a raw one-pass generator of the released 498-cue repository.

### System prompt

```text
Rules for cue generation:
1. Each cue must be PHYSICALLY VERIFIABLE in an image (not abstract or conceptual)
2. Each cue must be VISUALLY DISTINCT and specific (avoid vague terms like "different look")
3. Include cues across multiple visual dimensions: lighting, color, texture, objects, atmosphere
4. Cues should guide an image editing model to produce realistic weather effects

Respond ONLY in valid JSON. No preamble, no explanation, no markdown backticks.
```

### Per-condition request

```text
Generate a Knowledge Graph entry for the weather condition: "{condition}"

Return ONLY this JSON structure:
{
  "condition": "{condition}",
  "description": "one-sentence definition of this visual condition",
  "visual_cues": {
    "lighting": ["cue1", "cue2", ...],
    "color": ["cue1", "cue2", ...],
    "atmosphere": ["cue1", "cue2", ...],
    "surface": ["cue1", "cue2", ...],
    "objects": ["cue1", "cue2", ...]
  }
}

Requirements:
- 3–5 cues per visual dimension
```

The paper identifies the model family as Claude Sonnet. The exact service-side model snapshot is not recoverable from the archived workspace, so the repository does not claim a more specific identifier.

## 6. Evaluation captions

`data/clipout_captions.json` contains the five fixed target-condition captions used by every method for CLIPout. CLIPdir uses the same sample-specific Emu Edit input and output captions for every method.
