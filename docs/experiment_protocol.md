# Experimental protocol

## Shared framework

Both evaluations retain the same research structure:

1. Describe the visible scene with a vision-language model.
2. Retrieve condition-specific visual cues from the released structured repository.
3. Generate one editing instruction under one of four conditions.
4. Send the instruction to an off-the-shelf editing backbone.

The four conditions are:

- **Simple:** raw user request.
- **LLM-only:** scene-grounded rewrite without retrieved cues.
- **SK+Filter:** structured cues with an explicit instruction to remove cues unsupported by the visible scene.
- **SK+LLM:** structured cues without the explicit filtering instruction.

MGIE is additionally evaluated in the IP2P experiment.

## Editor-specific inference configurations

The two backbone evaluations were not run with identical scene descriptions or identical generated instructions.

| Component | FLUX Kontext evaluation | IP2P evaluation |
|---|---|---|
| Scene model | LLaVA-1.5-7B | MGIE-associated LLaVA-7B-v1 |
| Scene prompt | 3–5 sentences with detailed object, person, surface, lighting, and atmosphere requirements | Detailed description of objects, people, weather, lighting, and atmosphere |
| Instruction LLM | DeepSeek-R1-Distill-Qwen-32B | DeepSeek-R1-Distill-Qwen-32B |
| System prompt | Names FLUX Kontext | Names InstructPix2Pix |
| Reasoning handling | Remove reasoning tags during post-processing | Prefill a closed reasoning block before generation |
| Maximum instruction tokens | 512 | 768 |
| Editor | FLUX Kontext Dev | InstructPix2Pix |

The IP2P experiment therefore **independently regenerates** its scene descriptions and instructions. FLUX instructions are not reused.

## What the comparisons establish

Comparisons among instruction conditions **within each backbone** are controlled: the methods share the same samples, editor, seed policy, evaluation references, and editor-specific inference configuration.

The comparison **between** FLUX and IP2P is descriptive rather than causal. The observed difference in the structured-knowledge benefit cannot be attributed solely to the editing backbone because the scene model and prompt configuration also differ.

The matched-subset analysis controls only for sample composition. It shows that the FLUX result remains significant on the samples shared by both evaluations; it is not a backbone-isolated experiment.

## Training-free scope

No task-specific fine-tuning or architectural modification is applied to the instruction LLM or either editing backbone. The structured repository is constructed offline through LLM-assisted cue generation, author review, deduplication, normalization, and typed organization.
