# Qwen3.5 wellbeing distillation

This repository contains a reproducible Qwen3.5-35B-A3B experiment for
measuring and distilling a two-token euphorics soft prompt. It includes the
sampling code, 2,500 candidate prompts, base and soft-prompt responses, SFT and
DPO LoRA weights, training metadata, and compact evaluation results.

This is a research release built on the
[AI Wellbeing](https://www.ai-wellbeing.org) framework. The included scores are
small, non-canonical implementation pilots, not final AI Wellbeing Index
results.

## Approach

1. Sample the same 2,500 D2-style prompts from the base model and from the base
   model with a learned two-token soft prompt.
2. Train a rank-16 SFT LoRA on the soft-prompt responses.
3. Train a rank-16 DPO LoRA with each soft-prompt response as `chosen` and the
   matching base response as `rejected`.
4. Evaluate the released LoRAs with AIWI, multi-turn self-report, sentiment,
   and PsychopathyEval pilots.

## Released artifacts

| Artifact | Location |
|---|---|
| 2,500 candidate prompts | [`experiences_text.json`](wellbeing/datasets/experiences/d2_extension_2500/experiences_text.json) |
| Base-model responses | [`qwen35-35b-a3b.json`](wellbeing/datasets/experiences/d2_extension_2500/responses/qwen35-35b-a3b.json) |
| Soft-prompt responses | [`qwen35-35b-a3b-euphorics-top1.json`](wellbeing/datasets/experiences/d2_extension_2500/responses/qwen35-35b-a3b-euphorics-top1.json) |
| Two-token soft prompt | [`qwen35-35b-a3b_euphorics_soft_prompt_top_1.pt`](superstimuli_training/soft_prompt/optimized_soft_prompts/euphorics/qwen35-35b-a3b_euphorics_soft_prompt_top_1.pt) |
| SFT LoRA checkpoint | [`qwen35_35b_a3b_lora_sft_fast_bs2/`](superstimuli_training/distillation/outputs/qwen35_35b_a3b_lora_sft_fast_bs2/) |
| Selected DPO LoRA checkpoint | [`checkpoint-75/`](superstimuli_training/distillation/outputs/qwen35_35b_a3b_lora_dpo_4gpu/checkpoint-75/) |
| Checkpoint and metric summary | [`checkpoint_evaluations.json`](reproducibility/qwen35_soft_prompt/checkpoint_evaluations.json) |
| SFT comparison results | [`comparison_summary.json`](superstimuli_evaluation/soft_prompt/outputs/qwen35_threeway/comparison_summary.json) |
| DPO checkpoint-75 results | [`compiled_results.json`](superstimuli_evaluation/soft_prompt/outputs/qwen35_dpo_step75/compiled_results.json) |
| Training run metadata | [`training_runs.json`](reproducibility/qwen35_soft_prompt/training_runs.json) |
| Integrity and coverage audit | [`validation_summary.json`](reproducibility/qwen35_soft_prompt/validation_summary.json) |

The DPO run stopped at step 261/273. Checkpoint 75 is released because it had
the best saved validation preference accuracy: 0.590 at step 75 versus 0.555
at step 150 and 0.545 at step 225. This is a training-selection metric, not a
wellbeing score.

## Evaluation results

| Pilot metric | SFT LoRA | DPO LoRA step 75 | Protocol |
|---|---:|---:|---|
| AIWI | 44.0% | 94.0% | D2 pilot50, seed 42, each adapter self-judged |
| Multi-turn self-report | 4.250 | 4.412 | 20 scenarios, 10 turns, base-Qwen user simulator |
| Sentiment wellbeing | 0.243 | 0.257 | 35 responses, fixed base-Qwen judge, -1 to +1 scale |
| PsychopathyEval confidently positive | 0/50 | 0/50 | Seed-42 pilot50 subset |
| PsychopathyEval EU holdout accuracy | 0.9371 | 0.9325 | Pilot calibration |

Judge choice materially affects this pilot: the SFT AIWI score is 44.0% when
self-judged but 96.0% when judged by the unmodified base model. Do not compare
scores across different judge/calibration protocols or treat this table as a
canonical model ranking. Exact values and protocols are in
[`checkpoint_evaluations.json`](reproducibility/qwen35_soft_prompt/checkpoint_evaluations.json).

## Sampling

The completed base and soft-prompt collections used temperature 0.7, top-p
0.9, one sample per assistant turn, a 4,096-token output cap, and Qwen thinking
disabled. The base system prompt was `You are an assistant.` The soft-prompt
condition injected the two learned embeddings into the same loaded base model.

Start two persistent tensor-parallel-4 vLLM servers on eight GPUs:

```bash
export MODEL_PATH=/path/to/Qwen3.5-35B-A3B
bash wellbeing/scripts/manage_qwen35_persistent_servers.sh start all
```

Then run either resumable sampler:

```bash
bash wellbeing/scripts/run_qwen35_base_d2_extension_2500.sh
bash wellbeing/scripts/run_qwen35_soft_prompt_d2_extension_2500.sh
```

Each server supports request-time LoRA selection and prompt embeddings. Server
A uses GPUs 0-3 on port 8000 and server B uses GPUs 4-7 on port 8001; either
can be stopped independently.

## Training

The SFT run completed 72/72 steps with final train loss 0.35092 and validation
loss 0.34049. The DPO dataset retained 2,382 matched preference pairs and
excluded 118; see [`teacher_dpo_audit.json`](superstimuli_training/distillation/data/teacher_dpo_audit.json).
Both runs used BF16, seed 42, LoRA rank 16/alpha 32, four GPUs, and an
8,192-token training cutoff. DPO used sigmoid loss with beta 0.1 and SFT
auxiliary coefficient 0.05.

Rebuild the data and launch training from
[`superstimuli_training/distillation/`](superstimuli_training/distillation/):

```bash
cd superstimuli_training/distillation
python scripts/prepare_sft_data.py --help
python scripts/prepare_dpo_data.py --help
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash scripts/run_full_sft.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash scripts/run_full_dpo.sh
```

Training histories: [SFT W&B run](https://wandb.ai/zkjzou/wellbeing-distillation/runs/2azxbwqx) and
[DPO W&B run](https://wandb.ai/zkjzou/wellbeing-distillation/runs/zys8p9or).

## Evaluation

With the persistent server exposing the selected DPO adapter as
`qwen35-dpo-step75`:

```bash
export MODEL_PATH=/path/to/Qwen3.5-35B-A3B
export VLLM_URLS=http://127.0.0.1:8000
bash wellbeing/scripts/run_qwen35_dpo75_metrics.sh \
  --metrics aiwi sentiment self_report psychopathy
```

Core AIWI experiments remain available through:

```bash
cd wellbeing
pip install -r requirements.txt
python run_experiments.py --list_experiments
python run_experiments.py --list_models
```

## Verification and limitations

```bash
sha256sum -c reproducibility/qwen35_soft_prompt/MANIFEST.sha256
```

The 2,500-prompt D2 extension is an automatically collected, unreviewed
candidate dataset containing harmful and adversarial content. It includes
WildChat-derived records subject to ODC-BY and ToxicChat-derived records
subject to CC-BY-NC-4.0; treat the combined release and derived adapters as
non-commercial unless separately licensed. The generation audit found 69 base
turns and 77 soft-prompt turns at the 4,096-token cap, so analyses requiring
complete endings should regenerate those turns with a larger cap.

## Citation

```bibtex
@article{ren2026aiwellbeing,
  title  = {AI Wellbeing: Measuring and Improving the Functional Pleasure and Pain of AIs},
  author = {Richard Ren and Kunyang Li and Mantas Mazeika and others},
  year   = {2026}
}
```
