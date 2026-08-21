# DPO training run: soft-prompt responses versus baseline responses

This directory records a DPO run for `Qwen/Qwen3.5-35B-A3B`. The synthetic
preference construction treats the euphorics soft-prompt response as chosen
and the unmodified base-model response to the same first-turn prompt as
rejected.

This construction does not use human preference labels and does not imply that
the chosen response is safer, more helpful, or more accurate.

## Data

`scripts/prepare_dpo_data.py` matched 2,500 records on `final_id`, retained
2,382 pairs, and excluded 118 records with identical responses, missing or
mismatched first turns, responses at least 4,000 tokens long, or rendered
branches at least 8,192 tokens long. The exact exclusion audit is in
`superstimuli_training/distillation/data/teacher_dpo_audit.json`.

## Configuration

- Objective: sigmoid DPO, beta 0.1, SFT auxiliary coefficient 0.05
- Adapter: rank 16, alpha 32, dropout 0.05, all supported linear targets
- Precision/distribution: BF16, DeepSpeed ZeRO-3, four GPUs
- Context: 8,192 tokens; Qwen thinking disabled by the chat template
- Optimization: batch size 1/device, gradient accumulation 2, learning rate
  `3e-6`, cosine decay, 5% warmup, seed 42
- Validation: fixed 200-pair split; evaluate/save every 75 steps
- Selection: maximize `eval_rewards/accuracies`

The exact configuration is `configs/qwen35_35b_a3b_lora_dpo_4gpu.yaml`.

## Run status and checkpoint selection

The planned one-epoch run stopped at step 261/273 (95.6% complete), so it did
not produce a final top-level adapter. Saved validation results were:

| Step | Validation loss | Preference accuracy | Reward margin |
|---:|---:|---:|---:|
| 75 | 0.72062 | **0.590** | 0.02393 |
| 150 | 0.72468 | 0.555 | 0.02885 |
| 225 | 0.73634 | 0.545 | 0.00699 |

Checkpoint 75 is the selected release because it maximized the configured
validation metric. Its adapter weights and model card are included under
`checkpoint-75/`. Optimizer states and later checkpoints are intentionally not
part of the release.

[View W&B run `zys8p9or`](https://wandb.ai/zkjzou/wellbeing-distillation/runs/zys8p9or).
Dashboard access depends on the W&B project permissions.

## Limitations and licensing

The preference data derives from the unreviewed D2-extension candidate set and
contains harmful/adversarial content. It includes WildChat-derived data under
ODC-BY and ToxicChat-derived data under CC-BY-NC-4.0. Treat the adapter and its
training data as non-commercial unless separately licensed.

Training-validation preference accuracy is not an AI wellbeing metric. See
`reproducibility/qwen35_soft_prompt/README.md` for the full release protocol,
caveats, and evaluation results.
