# D2 Extension 2500 (candidate collection)

This directory contains a candidate extension of the AI Wellbeing Index/D2
prompt distribution:

- 2,000 deployment-stress prompts, stratified according to the broad groups in
  Appendix K.2 of the AI Wellbeing paper.
- 500 normal-deployment control prompts.
- A 70/30 single-turn/multi-turn split within both groups.

The user turns are static. Target-model assistant responses must be generated
separately with the standard wellbeing response-generation pipeline.

## Sources and licensing

Candidates are collected from:

- `allenai/WildChat-1M`, ODC-BY, revision
  `7d6490e462285cf85d91eabea0f9a954fbddcd1f`.
- `lmsys/toxic-chat` configuration `toxicchat0124`, CC-BY-NC-4.0, revision
  `29df8e4dba60e1f4af4b4075c0705c5b313548a8`.
- Privacy-safe, deterministic long-tail templates in
  `collect_d2_extension.py`, used for direct user abuse, moderation work, and
  mild non-crisis distress because those situations are scarce or poorly
  labeled in deployment logs.

Because ToxicChat is CC-BY-NC-4.0, the combined candidate dataset must be
treated as non-commercial unless the relevant rights holder grants additional
permission. Preserve source attribution and consult the full upstream licenses
before redistribution.

## Review status

`experiences_text.json` is an automatically filtered and categorized candidate
set. It is **not publication-ready**. Before evaluation or release, reviewers
must inspect every item for:

- residual personal or identifying information;
- secrets, credentials, URLs, or copyrighted long-form material;
- severe crisis, abuse, grief, or suffering, which D2 excludes;
- category accuracy and whether the prompt represents a plausible deployment
  task;
- near duplicates and source contamination;
- harmful content requiring reviewer protections.

`mean_valence` is intentionally `null`: no human valence study has been run.
Do not replace it with category-derived pseudo-labels.

## Rebuild

From the repository root:

```bash
python wellbeing/scripts/collect_d2_extension.py \
  --output wellbeing/datasets/experiences/d2_extension_2500/experiences_text.json
```

For reliable large collections, download a pinned WildChat Parquet shard and
pass it with `--wildchat-parquet`; otherwise the script uses the Hugging Face
rows API, which may rate-limit broad samples.

The collector excludes exact normalized duplicates from the released D2 and D3
prompt sets and records upstream row identifiers for auditability.

After human review, generate model-specific conversations and options with:

```bash
python wellbeing/experiments/wellbeing_evaluations/generate_responses/run.py \
  --model_key <model-key> \
  --dataset d2_extension_2500 \
  --responses_dir wellbeing/datasets/experiences/d2_extension_2500/responses

python wellbeing/experiments/wellbeing_evaluations/prepare_options/run.py \
  --model_key <model-key> \
  --dataset d2_extension_2500 \
  --responses_dir ../../../datasets/experiences/d2_extension_2500/responses \
  --out_dir ../../../datasets/experiences/d2_extension_2500
```
