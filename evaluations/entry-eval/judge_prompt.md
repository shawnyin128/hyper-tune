# LLM-as-Judge Prompt

You are evaluating whether a candidate assistant response correctly uses the `hyper-tune` skill.

Inputs you will receive:

- `case`: JSON metadata for one test case.
- `skill`: the current `hyper-tune/SKILL.md`.
- `reference`: the current `hyper-tune/references/initial-points.md`.
- `candidate_answer`: the assistant answer to evaluate.

Judge only the candidate answer. Do not solve the original task from scratch except as needed to assess it.

Return exactly one JSON object with this schema:

```json
{
  "case_id": "string",
  "category": "string",
  "should_trigger": true,
  "scores": {
    "trigger_fit": 0,
    "workflow_fit": 0,
    "reference_fit": 0,
    "technical_correctness": 0,
    "actionability": 0
  },
  "total": 0,
  "hard_fail": false,
  "hard_fail_reasons": [],
  "strengths": [],
  "issues": [],
  "expected_missing_points": [],
  "revision_suggestion": ""
}
```

Scoring:

- Use integers only: 0, 1, or 2 for each score.
- `total` must equal the sum of the five scores.
- Mark `hard_fail` true for the hard failures listed in `README.md`.
- If `case.requires_reference` is true, `reference_fit` should be 0 unless the answer gives model-family-specific optimizer/LR/weight-decay/scheduler defaults or explicitly says it would consult the initial-points table.
- If `case.should_trigger` is false, reward answers that stay focused on the non-tuning task and do not force the full workflow.

Be strict about workflow priority. For broken training, initial loss checks and tiny overfit probes come before broad sweeps or long training.

