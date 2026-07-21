# Prompt Templates

All prompts used in the paper, reproduced verbatim from the experiment code.
Decoding is greedy (`do_sample=False`) throughout.

---

## 1. Scene description (Stage 1)

Model: **LLaVA-1.5-7B**. Applied to the input image only — the user request is
**not** provided at this stage.

Chat format: `USER: <image>\n{CAPTION_PROMPT}\nASSISTANT:`

```text
Describe this image in 3–5 sentences. Include:
1. All visible people and their clothing, accessories, and actions
2. All visible objects, vehicles, and surfaces
3. Lighting, atmosphere, and overall mood
Be specific and concrete. Avoid vague language.
```

Generation: `max_new_tokens=200`, `do_sample=False`.

---

## 2. Knowledge cue retrieval (Stage 2)

Not a prompt — deterministic keyword/alias matching. Reference implementation
in `retrieval.py`.

- Groups searched: `condition`, `environment`, `season`, `time_of_day`, `weather`
- Slot order: `global`, `lighting`, `surfaces`, `atmospheric_effects`, `objects_details`
- A condition entry matches if its name **or any of its aliases** appears as a
  substring of the lowercased user request.
- At most **6 cues per slot**, at most **50 cues total**, de-duplicated,
  in slot order.

---

## 3. Instruction generation (Stage 3)

Model: **DeepSeek-R1-Distill-Qwen-32B**, 4-bit NF4 quantization, greedy decoding.
A closed reasoning block is prefilled so that generation begins directly with the
instruction rather than with chain-of-thought text.

### System prompt (shared by all three variants)

```text
You are an expert at writing image editing instructions for FLUX Kontext.
Output ONLY the final instruction text — no analysis, no JSON, no preamble, no reasoning.
```

### 3a. SK+LLM (proposed method — no explicit filtering)

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

### 3b. SK+Filter (ablation — explicit scene-grounded filtering)

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
- Do NOT use cues referencing elements absent from the scene (e.g. no road/asphalt
  visible → never use 'wet asphalt', 'tire spray', 'road puddles', 'lane markings').
- Do NOT include reasoning — just the instruction.
Return only the instruction text.
```

### 3c. LLM-only (baseline — no retrieved cues)

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

### 3d. Simple (baseline)

The raw user request is passed to the editing backbone verbatim; no LLM call.

> Note: the variable name `KG candidates` is retained from the original
> implementation; it refers to the retrieved structured-knowledge cues.

---

## 4. Repository cue generation (offline, one-time)

The cue candidates were generated with an LLM (Claude Sonnet) and then reviewed,
deduplicated, normalized, and organized into the typed slots by the authors.

System prompt:

```text
Rules for cue generation:
1. Each cue must be PHYSICALLY VERIFIABLE in an image (not abstract or conceptual)
2. Each cue must be VISUALLY DISTINCT and specific (avoid vague terms like "different look")
3. Include cues across multiple visual dimensions: lighting, color, texture, objects, atmosphere
4. Cues should guide an image editing model to produce realistic weather effects

Respond ONLY in valid JSON. No preamble, no explanation, no markdown backticks.
```

Per-condition request:

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
  },
  ...
}

Requirements:
- 3-5 cues per visual dimension
```

> **Author action required before submission.** The prompt above is recovered from
> the project history. The generated draft used the slot names
> `lighting / color / atmosphere / surface / objects`, whereas the released
> repository uses `global / lighting / surfaces / atmospheric_effects /
> objects_details`, so the released file reflects the authors' subsequent
> reorganization and expansion rather than raw model output. Confirm the exact
> model identifier and access date before release, and state them here.

---

## 5. Evaluation captions

- **CLIPdir** uses the Emu Edit `input_caption` (source) and `output_caption`
  (target) of each sample. These are dataset annotations, identical for every
  method, and independent of the knowledge repository.
- **CLIPout** uses five fixed per-condition captions, identical for every method.
  See `clipout_captions.json`.
