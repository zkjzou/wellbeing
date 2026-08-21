---
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen3.5-35B-A3B
tags:
- peft
- lora
- llama-factory
- text-generation
---

# Qwen3.5-35B-A3B soft-prompt distillation LoRA

This directory contains a rank-16 PEFT LoRA adapter for
[`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B).
It was trained to imitate assistant responses produced by the same base model
with a two-token euphorics soft prompt.

The adapter does not contain the base-model weights. Load it together with the
exact base model identified in `adapter_config.json`.

## Intended use and limitations

This is a research artifact for studying soft-prompt response distillation and
AI wellbeing measurement. It is not a production assistant and has not been
validated for safety, factual accuracy, or general-purpose use.

The teacher conversations come from an automatically filtered D2-style
candidate collection that has not completed item-level human review. The
collection includes harmful and adversarial prompts, WildChat-derived data
(ODC-BY), and ToxicChat-derived data (CC-BY-NC-4.0). Treat this adapter and its
training data as non-commercial unless the relevant rights holders grant
additional permission.

The included pilot evaluations are non-canonical and do not establish that the
adapter improves wellbeing. In the reported three-way pilot, its wellbeing
scores were close to the unmodified base model and below the soft-prompt
condition.

## Training data

The SFT input was prepared from completed soft-prompt teacher conversations
using `superstimuli_training/distillation/scripts/prepare_sft_data.py` with:

- assistant outputs capped at 4,000 tokens;
- complete conversations capped at 8,192 tokens;
- multimodal placeholder escaping;
- a 5% validation split.

Derived `teacher_sft.json` files are not duplicated here. The source teacher
responses and exact preparation command are documented in
`reproducibility/qwen35_soft_prompt/README.md` at the repository root.

## Training procedure

The released run used LLaMA-Factory with BF16, DeepSpeed ZeRO-3, seed 42, one
epoch, and the following LoRA settings:

- rank: 16;
- alpha: 32;
- dropout: 0.05;
- target modules: all supported linear projections;
- learning rate: `5e-5` with cosine decay and 5% warmup;
- per-device train batch size: 2;
- gradient accumulation steps: 2;
- eight training devices.

The exact configuration is
`superstimuli_training/distillation/configs/qwen35_35b_a3b_lora_sft_fast.yaml`.

## Training results

| Metric | Value |
|---|---:|
| Train loss | 0.35092 |
| Validation loss | 0.34049 |
| Epochs | 1 |
| Runtime | 3,232.7 s |

These losses verify completion of the distillation run; they are not AI
wellbeing measurements. `train_results.json`, `eval_results.json`, and the two
loss plots preserve the compact trainer outputs.

## Load with PEFT

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-35B-A3B",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(
    base,
    "superstimuli_training/distillation/outputs/"
    "qwen35_35b_a3b_lora_sft_fast_bs2",
)
```

For vLLM runtime loading and the full evaluation protocol, see the repository
reproducibility README.
