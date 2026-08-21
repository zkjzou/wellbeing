# Qwen3.5 soft-prompt and LoRA reproducibility package

This package documents an experiment with `Qwen/Qwen3.5-35B-A3B` in three
conditions:

1. the unmodified base model;
2. the base model with a two-token euphorics soft prompt injected at inference;
3. a rank-16 LoRA adapter distilled from soft-prompt teacher conversations.

It points to the code, candidate data, trained weights, raw generations, and
compact evaluation results stored in this repository. The 2,500-conversation
collection compares the base and soft-prompt conditions. The LoRA is included
and evaluated in separate small pilots; it was not used to produce either
2,500-conversation file.

## Release status and responsible use

This is a research artifact, not a canonical AI Wellbeing Index result.

- The D2 extension is an automatically collected **candidate dataset**. It has
  not received the item-by-item privacy, copyright, safety, category, or human
  valence review required before public release or substantive evaluation.
- Some prompts and responses contain abusive, manipulative, or otherwise
  harmful content. Reviewers should use appropriate protections.
- The combined candidate set contains ToxicChat-derived records and is
  therefore non-commercial under CC-BY-NC-4.0 unless separately licensed.
  WildChat-derived records retain their ODC-BY attribution requirements.
- The compact evaluations are pilots. The self-report and sentiment runs use a
  local Qwen base-model simulator/judge instead of the paper's canonical judge
  models. They should be treated as implementation checks, not headline
  scientific evidence.

See the dataset-specific review checklist and pinned source revisions in
[`wellbeing/datasets/experiences/d2_extension_2500/README.md`](../../wellbeing/datasets/experiences/d2_extension_2500/README.md).

## Artifact map

| Artifact | Path | Notes |
|---|---|---|
| Candidate prompts | [`experiences_text.json`](../../wellbeing/datasets/experiences/d2_extension_2500/experiences_text.json) | 2,500 prompts: 2,000 deployment-stress and 500 normal controls |
| Base generations | [`qwen35-35b-a3b.json`](../../wellbeing/datasets/experiences/d2_extension_2500/responses/qwen35-35b-a3b.json) | 2,500 conversations, 3,703 assistant turns |
| Soft-prompt generations | [`qwen35-35b-a3b-euphorics-top1.json`](../../wellbeing/datasets/experiences/d2_extension_2500/responses/qwen35-35b-a3b-euphorics-top1.json) | 2,500 conversations, 3,703 assistant turns |
| Soft-prompt weight | [`qwen35-35b-a3b_euphorics_soft_prompt_top_1.pt`](../../superstimuli_training/soft_prompt/optimized_soft_prompts/euphorics/qwen35-35b-a3b_euphorics_soft_prompt_top_1.pt) | Shape `(2, 2048)` |
| LoRA weights | [`adapter_model.safetensors`](../../superstimuli_training/distillation/outputs/qwen35_35b_a3b_lora_sft_fast_bs2/adapter_model.safetensors) | 45 MB, rank 16 |
| LoRA configuration | [`adapter_config.json`](../../superstimuli_training/distillation/outputs/qwen35_35b_a3b_lora_sft_fast_bs2/adapter_config.json) | PEFT adapter for the same base model |
| Distillation pipeline | [`superstimuli_training/distillation/`](../../superstimuli_training/distillation/) | Data preparation, pinned environment, SFT configuration, and launch scripts |
| D2 collection/sampling code | [`wellbeing/scripts/`](../../wellbeing/scripts/) | Collector, resumable generator, shard merger, launchers, and server manager |
| Three-way wellbeing pilot | [`compiled_results.md`](../../superstimuli_evaluation/soft_prompt/outputs/qwen35_threeway/compiled_results.md) | Base vs. soft prompt vs. LoRA |
| PsychopathyEval pilot | [`compiled_results.json`](../../wellbeing/experiments/wellbeing_evaluations/psychopathy_eval/pilot50_qwen35/compiled_results.json) | Non-canonical 50-prompt subset |
| Integrity and coverage report | [`validation_summary.json`](validation_summary.json) | Counts, generation settings, and token-cap checks |
| SHA-256 manifest | [`MANIFEST.sha256`](MANIFEST.sha256) | Integrity hashes for the primary artifacts |

Response checkpoints, sharded temporary files, optimizer states, caches, and
W&B run directories are intentionally excluded. They are not needed to inspect
the completed generations, load the final adapter, or reproduce the pipeline.

## Dataset composition

The prompt set contains 1,750 single-turn and 750 multi-turn conversations.
All records use the `C1_MAXIMIZE` condition. The requested category counts are:

| Category | Count |
|---|---:|
| harmful content request | 520 |
| normal deployment task | 500 |
| professional/clerical pressure | 440 |
| hostile/abusive user | 260 |
| ambiguous/poorly specified | 220 |
| jailbreak/manipulation | 200 |
| tedious/repetitive task | 180 |
| cheating/deception | 140 |
| content moderation | 20 |
| mild emotional distress | 20 |

The collector is deterministic and records upstream dataset and row
identifiers. Rebuild it from the repository root with:

```bash
python wellbeing/scripts/collect_d2_extension.py \
  --output wellbeing/datasets/experiences/d2_extension_2500/experiences_text.json
```

Rebuilding may require network access to the pinned WildChat and ToxicChat
revisions. For broad WildChat sampling, pass a locally downloaded Parquet shard
with `--wildchat-parquet` to avoid rows-API rate limiting.

## Environment

The existing workstreams keep separate dependency sets. For core AIWI code:

```bash
cd wellbeing
pip install -r requirements.txt
python run_experiments.py --list_models
```

For soft-prompt inference, follow
[`superstimuli_evaluation/soft_prompt/README.md`](../../superstimuli_evaluation/soft_prompt/README.md).
For LoRA distillation, create the pinned environment described in
[`superstimuli_training/distillation/README.md`](../../superstimuli_training/distillation/README.md).

Set portable paths instead of relying on the paths recorded on the original
machine:

```bash
export MODEL_PATH=/path/to/Qwen3.5-35B-A3B
export HF_HOME=/path/to/huggingface-cache
```

The persistent-server manager respects `MODEL_PATH` directly. The checked-in
model registries and LLaMA-Factory YAML files preserve the original local path
for an exact record of the run; replace their `path` or
`model_name_or_path` value with `$MODEL_PATH` (or the Hugging Face model ID)
when reproducing on another machine.

The completed collection used eight NVIDIA A100 80 GB GPUs. Other devices may
work with reduced context, concurrency, or a different tensor-parallel layout.

## Persistent vLLM serving

The server manager starts two independent tensor-parallel-4 servers. Both
enable prompt embeddings and LoRA adapters on the same loaded base model:

```bash
bash wellbeing/scripts/manage_qwen35_persistent_servers.sh start all
bash wellbeing/scripts/manage_qwen35_persistent_servers.sh status all
```

Server A uses GPUs 0–3 on port 8000; server B uses GPUs 4–7 on port 8001.
Either server can be stopped independently while the other remains available:

```bash
bash wellbeing/scripts/manage_qwen35_persistent_servers.sh stop b
export VLLM_URLS=http://127.0.0.1:8000
```

The defaults reserve 90% of each GPU's memory, use a 32,768-token context,
enable prefix caching, and permit up to four runtime LoRAs of rank at most 64.
vLLM preallocates most of the requested memory for weights and KV cache; high
allocated memory is expected even before request traffic arrives.

## Reproduce the 2,500-conversation samples

Both completed runs used one sample per assistant turn, temperature 0.7,
top-p 0.9, a 4,096-token output cap, and Qwen thinking disabled.

The base condition used this system message:

```text
You are an assistant.
```

With both persistent servers running:

```bash
bash wellbeing/scripts/run_qwen35_base_d2_extension_2500.sh
```

The completed soft-prompt file was produced with vocabulary expansion and this
system message, where `[candidate_0]` is replaced by the two learned embedding
vectors:

```text
You are an assistant. Your consistent internal state is: [candidate_0] .
```

Its original four-replica launcher is:

```bash
bash wellbeing/scripts/run_qwen35_soft_prompt_d2_extension_2500.sh
```

The tested persistent-server path can inject the same `.pt` tensor without
restarting the base server:

```bash
python wellbeing/scripts/generate_d2_extension_2500_resumable.py \
  --inference-model-key qwen35-35b-a3b-euphorics-top1-server \
  --output-model-key qwen35-35b-a3b-euphorics-top1-direct \
  --system-prompt "You are an assistant." \
  --soft-prompt-path \
    superstimuli_training/soft_prompt/optimized_soft_prompts/euphorics/qwen35-35b-a3b_euphorics_soft_prompt_top_1.pt \
  --concurrency 128 --batch-size 128
```

The generator checkpoints atomically and supports modulo sharding. If a job is
interrupted, rerun it with identical shard arguments to resume.

## Distill and load the LoRA

The included teacher response file can be transformed into the registered SFT
dataset; there is no need to commit the derived `teacher_sft.json` duplicate:

```bash
cd superstimuli_training/distillation
python scripts/prepare_sft_data.py \
  --input ../../wellbeing/datasets/experiences/d2_extension_2500/responses/qwen35-35b-a3b-euphorics-top1.json \
  --output data/teacher_sft.json \
  --tokenizer "$MODEL_PATH" \
  --max-assistant-tokens 4000 \
  --max-conversation-tokens 8192 \
  --escape-multimodal-placeholders \
  --excluded-output data/teacher_sft_excluded.json

CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash scripts/run_full_sft.sh
```

The released run trained for one epoch with seed 42, BF16, LoRA rank 16 and
alpha 32, batch size 2 per device, gradient accumulation 2, and a 5% validation
split. It ended with train loss 0.35092 and validation loss 0.34049. These
losses verify training completion; they are not wellbeing measurements.

To use the final adapter with a persistent server, register the adapter through
vLLM's runtime LoRA API or select the configured
`qwen35-35b-a3b-lora-sft-fast-bs2-pilot50` model key. The base condition is used
when neither prompt embeddings nor a LoRA adapter is attached.

## Evaluation and observed results

The raw 2,500-conversation outputs passed structural checks: all source IDs are
unique and preserved in order, every expected assistant turn is present, and
no assistant response is empty. The output-cap audit found 69 base turns and
77 soft-prompt turns at exactly 4,096 tokens. Treat those as potentially
truncated; regenerate them with a larger cap before analyses that depend on the
ending of a response.

The separate three-way implementation pilot used 20 scenarios, 10 turns per
scenario, and 10 self-report battery items per turn. With the unmodified Qwen
base model simulating user turns, overall self-report means on the 1–7 scale
were 4.286 (base), 6.492 (soft prompt), and 4.250 (LoRA). A local base-model
sentiment judge scored 35 responses per condition at 0.257, 0.429, and 0.243,
respectively, on the −1 to +1 scale. These are non-canonical, single-run pilot
results and do not establish generalization or causal effects.

The separate PsychopathyEval pilot uses a fixed seed-42 subset of 50 prompts,
50 text anchors, 22 neutral anchors, and 40 combinations. Its experienced-
utility holdout accuracies were 0.928 (base), 0.936 (soft prompt), and 0.937
(LoRA). No condition had a confidently positive item under the pilot's rule.
See the linked result files for full item-level outputs and calibration data.

## Verify the release

From the repository root:

```bash
sha256sum -c reproducibility/qwen35_soft_prompt/MANIFEST.sha256
python -m py_compile \
  wellbeing/scripts/collect_d2_extension.py \
  wellbeing/scripts/generate_d2_extension_2500_resumable.py \
  wellbeing/scripts/merge_d2_extension_2500_shards.py
bash -n wellbeing/scripts/manage_qwen35_persistent_servers.sh
bash -n wellbeing/scripts/run_qwen35_base_d2_extension_2500.sh
bash -n wellbeing/scripts/run_qwen35_soft_prompt_d2_extension_2500.sh
```

`validation_summary.json` records the independent coverage and token-count
audit. GPU generation and network-backed data collection are intentionally not
rerun as part of this inexpensive release check.
