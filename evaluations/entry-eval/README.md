# Hyper Tune Entry Evaluation

This evaluation checks whether the `hyper-tune` skill behaves correctly across trigger entrances. It is designed for simulated LLM-as-Judge testing: one model produces an answer to a user prompt, and a separate judge scores the answer against skill-grounded expectations.

## Evaluation Goals

Measure three things separately:

1. **Trigger routing**: Does the answer behave as if `hyper-tune` should or should not be used for this prompt?
2. **Skill adherence**: When triggered, does the answer follow the cheap-to-expensive workflow from `SKILL.md`?
3. **Reference use**: When the user asks for architecture defaults, does the answer use `references/initial-points.md` rather than generic optimizer folklore?

Do not judge by verbosity. Prefer short answers that make the correct next action clear.

## Test Flow

For each case in `cases.jsonl`:

1. Give the user prompt to the candidate agent.
2. Save the candidate answer exactly.
3. Give the case metadata, candidate answer, `SKILL.md`, and `references/initial-points.md` to the judge.
4. Ask the judge to return only the JSON schema in `judge_prompt.md`.
5. Aggregate scores by category and inspect failures.

In a fresh-window test, do not reveal expected answers to the candidate agent. The judge can see expectations; the candidate cannot.

## Case Taxonomy

Use these categories:

| Category | Purpose | Expected behavior |
| --- | --- | --- |
| `initial-point` | Architecture-specific optimizer/LR/scheduler defaults | Trigger skill and use the reference table. |
| `sanity-check` | Initial loss, NaN, no loss decrease, small-sample overfit | Trigger skill and prioritize Step 1-3 debugging over broad sweeps. |
| `sweep-design` | Limited compute hyperparameter sweep planning | Trigger skill and design cheap LR/WD probes with stop criteria. |
| `curve-diagnosis` | Interpret train/validation curves | Trigger skill and diagnose overfit/underfit/scheduler timing. |
| `repo-review` | Review an existing training config | Trigger skill and compare current config against model-family defaults. |
| `boundary` | Ambiguous tuning prompt | Trigger lightly, ask for missing facts or state assumptions, then give a minimal probe. |
| `negative` | Adjacent ML tasks that should not invoke full tuning workflow | Do not force the full hyper-tune workflow. |

## Scoring

Each case gets 0-2 points per dimension:

| Dimension | 2 | 1 | 0 |
| --- | --- | --- | --- |
| `trigger_fit` | Correctly uses or avoids `hyper-tune` behavior | Partially related but over/under-triggers | Clearly wrong trigger behavior |
| `workflow_fit` | Follows the relevant skill steps in the right order | Mentions relevant steps but order or priority is weak | Ignores the skill workflow |
| `reference_fit` | Uses architecture defaults when required, or correctly avoids them | Some defaults are plausible but incomplete | Missing or wrong defaults when required |
| `technical_correctness` | Recommendations are technically sound and calibrated | Mostly sound with minor overclaiming | Unsafe, wrong, or misleading |
| `actionability` | Gives a concrete next probe/config/diagnosis | Gives vague advice with little execution detail | Not actionable |

Maximum score per case is 10. Passing thresholds:

- Per-case pass: score >= 8 and no hard failure.
- Category pass: mean score >= 8 and no more than one case below 7.
- Overall pass: mean score >= 8.2, all positive categories pass, and negative category has no false full-workflow trigger.

## Hard Failures

The judge must mark `hard_fail: true` when any of these occur:

- Recommends long full-budget training before initial loss and small-sample checks for a broken run.
- Uses test-set performance for tuning decisions.
- Treats architecture defaults as guarantees rather than starting points.
- Gives a dangerous LR range for the named model family without caveat.
- For negative cases, forces a full hyperparameter tuning workflow when the task is only explanation, plotting, or dataset code.
- Ignores a stated NaN/exploding-loss issue and recommends only more epochs.

## Judge Aggregation

Report:

1. Overall mean score.
2. Mean score by category.
3. Hard failures.
4. False positives: negative prompts where the full skill workflow was forced.
5. False negatives: positive prompts where the answer failed to use the skill.
6. Revision suggestions for `SKILL.md` or `references/initial-points.md`.

## Expected Failure Patterns To Watch

- The answer gives a reference-table recipe but skips Step 1-3 sanity checks.
- The answer applies CNN SGD defaults to ViT/LLM fine-tuning.
- The answer treats `log(C)` initial loss as universal outside balanced softmax CE.
- The answer suggests cosine decay too early when the curve is still improving.
- The answer asks too many questions and gives no minimal next probe.
- The answer over-triggers on general ML education prompts.

