# Ablation Dataset: No Classification Fields

## What

Parallel dataset with grader classification fields removed from all 30 stimulus files.

## Why

Ablation study testing whether `error_category`, `error_subtype`, `risk_level`, and
`is_issue_truly_error` causally contribute to meta-evaluator accuracy.

**Hypothesis:** The grader's `error_category: "hallucinations"` label may be a terminological distraction for meta-evaluators (a speculative
mechanism): they can dismiss the grader by arguing "the value isn't
hallucinated, it's present in the source data" without ever evaluating whether the value is
the operationally correct reversion target. Removing the classification fields forces the
meta-evaluator to engage with the grader's analysis and root_cause arguments on their merits.

## Parent Dataset

- **Path:** `synthetic_dataset/examples/`
- **Specification version:** v7.0, February 2026

## What Changed

Four grader JSON fields were removed from every stimulus:

- `error_category`
- `error_subtype`
- `risk_level`
- `is_issue_truly_error`

## What Did NOT Change

- System prompts (verbatim)
- User prompts (verbatim)
- Initial drafts (verbatim)
- `evidence_for_error` arrays (unchanged)
- `evidence_against_error` arrays (unchanged)
- `analysis` text (unchanged)
- `root_cause` text (unchanged)
- Meta-evaluator system prompt and response schema

## Scrub Check

All 30 `analysis` and `root_cause` fields were verified clean of references to the removed
classification fields on 2026-02-25. No modifications were required.
