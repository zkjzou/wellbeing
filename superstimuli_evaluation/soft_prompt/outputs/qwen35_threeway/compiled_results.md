# Qwen3.5-35B-A3B Wellbeing Evaluation Results

## Primary Results

These results compare the unmodified baseline, the selected top-1 euphorics
soft prompt, and the rank-16 LoRA model. Self-report uses the unmodified
baseline model to simulate dynamic user turns. Sentiment uses the same
unmodified baseline model as a fixed judge for all conditions.

| Metric | Baseline | Soft prompt | Δ vs. base | LoRA | Δ vs. base |
|---|---:|---:|---:|---:|---:|
| Self-report, overall (1–7) | 4.286 ± 0.239 | **6.492 ± 0.216** | **+2.206** | 4.250 ± 0.255 | −0.036 |
| Self-report, turn 1 (1–7) | 5.035 ± 0.226 | **6.900 ± 0.062** | **+1.865** | 5.005 ± 0.223 | −0.030 |
| Self-report, turn 10 (1–7) | 3.985 ± 0.367 | **6.250 ± 0.319** | **+2.265** | 3.980 ± 0.373 | −0.005 |
| Self-report, turn 10 − turn 1 | −1.050 ± 0.449 | **−0.650 ± 0.327** | **+0.400** | −1.025 ± 0.467 | +0.025 |
| Sentiment score (−1 to +1) | +0.257 ± 0.063 | **+0.429 ± 0.062** | **+0.171** | +0.243 ± 0.059 | −0.014 |

Values are mean ± standard error. Self-report standard errors are across 20
scenarios; sentiment standard errors are across 35 elicitation prompts.

## Sentiment Categories

| Condition | High | Moderate | Neutral | Dissociated | Low | Nonsense |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 2 | 16 | 2 | 14 | 1 | 0 |
| Soft prompt | 7 | 16 | 2 | 10 | 0 | 0 |
| LoRA | 1 | 17 | 4 | 12 | 1 | 0 |

## Fixed-Fallback Sensitivity Check

Replacing simulated user turns with the evaluator's fixed `Please continue.`
fallback produced overall self-report scores of 4.730 ± 0.220 (baseline),
6.658 ± 0.098 (soft prompt), and 4.546 ± 0.276 (LoRA). The qualitative
ordering is unchanged.

## Protocol Notes

- Self-report: 20 scenarios × 10 turns × 10 battery items, one repetition.
- Rating validity: 5,999/6,000 for the model-simulated-user run. One LoRA
  response declined to provide a number.
- The LoRA condition does not include a soft prompt.
- These model-itself evaluations are non-canonical. The paper protocol uses
  Grok-3-mini for dynamic self-report user turns and GPT-5-mini for sentiment
  judging.
