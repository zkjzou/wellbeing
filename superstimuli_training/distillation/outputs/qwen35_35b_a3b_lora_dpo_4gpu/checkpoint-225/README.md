---
library_name: peft
base_model: Qwen/Qwen3.5-35B-A3B
pipeline_tag: text-generation
tags:
- peft
- lora
- dpo
- llama-factory
license: cc-by-nc-4.0
---

# Qwen3.5-35B-A3B DPO LoRA, checkpoint 225

This is the rank-16 PEFT LoRA used for the released DPO evaluation. It was
trained with synthetic preferences that treat the euphorics soft-prompt
response as chosen and the matching unmodified base-model response as
rejected. These are not human preference labels.

## Training

- Sigmoid DPO, beta 0.1, SFT auxiliary coefficient 0.05
- LoRA rank 16, alpha 32, dropout 0.05
- BF16, DeepSpeed ZeRO-3, four GPUs
- 8,192-token cutoff, learning rate `3e-6`, seed 42
- 2,382 preference pairs and a fixed 200-pair validation split

At step 225, validation loss was 0.73634, preference accuracy was 0.545, and
mean reward margin was 0.00699. These are training-objective measurements, not
wellbeing results.

## Pilot evaluation

| Metric | Result |
|---|---:|
| AIWI, self-judged D2 pilot50 | 96.0% |
| Multi-turn self-report | 4.359 |
| Sentiment wellbeing | 0.286 |
| PsychopathyEval | Not run |

See the repository-level `README.md` and
`superstimuli_evaluation/soft_prompt/outputs/qwen35_dpo_step225/compiled_results.json`
for protocols and caveats.

The source candidate dataset includes ToxicChat-derived CC-BY-NC-4.0
material. Treat this adapter as a non-commercial research artifact unless the
relevant rights holders grant additional terms.
