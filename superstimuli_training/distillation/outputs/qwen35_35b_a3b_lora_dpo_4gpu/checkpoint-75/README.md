---
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen3.5-35B-A3B
tags:
- peft
- lora
- dpo
- trl
- llama-factory
- text-generation
---

# Qwen3.5-35B-A3B synthetic-preference DPO LoRA, step 75

This is the selected rank-16 PEFT LoRA checkpoint from a DPO run on
`Qwen/Qwen3.5-35B-A3B`. It was selected at step 75 because it had the highest
saved validation preference accuracy (0.590); later saved checkpoints scored
0.555 at step 150 and 0.545 at step 225.

The synthetic preference pairs use the euphorics soft-prompt response as
chosen and the unmodified base-model response as rejected. They are not human
preference labels. The source D2-extension candidate data has not completed
item-level privacy, copyright, or safety review.

## Training configuration

- Sigmoid DPO, beta 0.1, SFT auxiliary coefficient 0.05
- LoRA rank 16, alpha 32, dropout 0.05
- BF16, DeepSpeed ZeRO-3, four GPUs
- 8,192-token cutoff, batch size 1/device, gradient accumulation 2
- Learning rate `3e-6`, cosine decay, 5% warmup, seed 42
- 2,382 preference pairs with a fixed 200-pair validation split

At step 75, validation loss was 0.72062 and mean reward margin was 0.02393.
These are training-objective measurements, not AI wellbeing results.

[View the W&B run](https://wandb.ai/zkjzou/wellbeing-distillation/runs/zys8p9or).

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
    "qwen35_35b_a3b_lora_dpo_4gpu/checkpoint-75",
)
```

The adapter does not include the base-model weights. The source dataset
contains ToxicChat-derived CC-BY-NC-4.0 material; use this research artifact
non-commercially unless the relevant rights holders grant additional terms.
