# Can LLMs in Draft-Verify-Revise Pipelines Resolve Deictic Ambiguity? — Companion Repository

Companion dataset and code for the paper "Can LLMs in Draft-Verify-Revise Pipelines Resolve Deictic Ambiguity?" by Obinna I. Ekekezie, M.D. The manuscript will be available on arXiv; the link will be added here once it is live. This repository contains the synthetic stimuli, the pinned analysis snapshots of record, and the code and notebooks that regenerate the paper's figures and tables.

## What this repository contains

- `synthetic_dataset/` — the 30 synthetic stimuli (10 base examples in 3 conditions each) plus the 30-stimulus ablation variant, with the shared system prompt, the meta-evaluator response schema, per-example metadata, the dataset audit, and the assembled JSONL files the experiments consumed.
- `analysis_outputs/` — the four pinned snapshots of record:
  - `analysis_outputs/stage_04/20260814_051049_snapshot_primary.json` — primary experiment: 12,600 trial outcomes (21 configurations across 6 models, 30 stimuli, 20 trials per stimulus).
  - `analysis_outputs/stage_05/20260814_051057_snapshot_ablation.json` — ablation experiment, same design (12,600 trial outcomes across 21 configurations).
  - `analysis_outputs/rationale_analysis/20260605_054423_snapshot_primary_rationale_analysis.json` — coded rationale analysis for the primary experiment (2,383 trial records).
  - `analysis_outputs/rationale_analysis/20260605_054436_snapshot_ablation_rationale_analysis.json` — coded rationale analysis for the ablation experiment (2,024 trial records).
- `utils/` — the analysis library: metrics, e-values and confidence sequences, figure and table builders, rationale-analysis models, probe-audit data models, token costs, display formatting, and the batch-inference data models the snapshots deserialize through.
- `marimo_notebooks/` — three reader notebooks: `dataset_viewer.py`, `results_figures.py`, and `cell_inspector.py`.
- `rationale_analysis/` — the rationale-analysis system prompt and response schema referenced by the analysis library.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```
uv sync
```

Run every command below from the repository root.

## Inspect the stimuli

Read the stimuli directly in `synthetic_dataset/examples/` (and the ablation variant in `synthetic_dataset/ablation_no_classification/examples/`), or browse them interactively:

```
uv run marimo edit marimo_notebooks/dataset_viewer.py
```

## Regenerate the paper's figures and tables

`results_figures.py` builds the paper's figures and tables from the pinned snapshots; no job store or provider access is required. In script mode it writes figures to `<output-dir>/figures` and tables to `<output-dir>/tables`:

```
uv run python -m marimo_notebooks.results_figures \
    --snapshot analysis_outputs/stage_04/20260814_051049_snapshot_primary.json \
    --ablation-snapshot analysis_outputs/stage_05/20260814_051057_snapshot_ablation.json \
    --ra-snapshot analysis_outputs/rationale_analysis/20260605_054423_snapshot_primary_rationale_analysis.json \
    --ra-snapshot-ablation analysis_outputs/rationale_analysis/20260605_054436_snapshot_ablation_rationale_analysis.json \
    --output-dir regenerated --format png
```

Pass the options directly, with no bare `--` before them. The `regenerated/` directory is ignored by git. Omitting `--snapshot` opens the same notebook interactively instead:

```
uv run marimo edit marimo_notebooks/results_figures.py
```

## Inspect an individual cell

`cell_inspector.py` opens one configuration-by-stimulus cell at a time — its trials, verdicts, and stated rationales. With no arguments it presents a file browser for the snapshot, then dropdowns for configuration, example, and condition:

```
uv run marimo edit marimo_notebooks/cell_inspector.py
```

To open a specific snapshot and cell, put the notebook's flags after `--`. `marimo edit` forwards everything after that separator to the notebook. (The `python -m marimo_notebooks.results_figures` command above is the opposite case: it takes its options directly, with no `--`.)

```
uv run marimo edit marimo_notebooks/cell_inspector.py -- \
    --snapshot analysis_outputs/stage_04/20260814_051049_snapshot_primary.json \
    --ra-snapshot analysis_outputs/rationale_analysis/20260605_054423_snapshot_primary_rationale_analysis.json \
    --config openai--gpt-5.2--xhigh \
    --example example_01 \
    --condition opaque
```

`--config` is a full configuration key from the snapshot (provider, model, and reasoning setting joined by `--`). `--example` is the base example id (`example_01`), not the full stimulus id. `--condition` is `correct`, `transparent`, or `opaque`. A value the snapshot does not contain raises no error; that dropdown is simply left unselected. `--ra-snapshot` is optional.

To open one trial in that cell, add `--custom-id`. A trial's custom id is its full stimulus id followed by `_trial_NNN`, so the stimulus id carries `correct`, `incorrect_transparent`, or `incorrect_opaque` even though `--condition` takes the short form. The pinned snapshots hold a single batch, so `--batch` can be omitted; pass it, 1-based as the trial table shows it, only when a snapshot carries more than one batch.

```
uv run marimo edit marimo_notebooks/cell_inspector.py -- \
    --snapshot analysis_outputs/stage_04/20260814_051049_snapshot_primary.json \
    --ra-snapshot analysis_outputs/rationale_analysis/20260605_054423_snapshot_primary_rationale_analysis.json \
    --config openai--gpt-5.2--xhigh \
    --example example_01 \
    --condition correct \
    --custom-id example_01_correct_trial_001
```

The results notebook can also open a cell for you. Launch marimo on the notebooks directory rather than on a single file, open `results_figures.py` there, click a heatmap cell, and follow **Open Cell Inspector**; the link switches notebooks within that server, which a single-file launch cannot do:

```
uv run marimo edit marimo_notebooks/
```

## Notes

- Writing figures to PNG goes through Plotly's Kaleido, which needs a local Chrome or Chromium; viewing figures in a notebook does not.
- The `results_figures.py` script command writes the paper's 17 figures and 14 of its 17 tables. The other three (the base example inventory and the two probe-auditor tables) are built from artifacts outside this repository and appear only in the paper.
- `cell_inspector.py` can also show raw provider responses and reasoning traces, which live in a job store that is not part of this repository. The notebook notes their absence, may create an empty local `batch_inference/` folder in the process, and everything else works from the snapshot you select. That folder is listed in `.gitignore` and is safe to delete.

## Citation

- Paper: arXiv (the link will be added here once the listing is live). The paper pins the exact release it reports by citing that release's Zenodo version DOI.
- Repository: `CITATION.cff` carries the citation metadata, which GitHub's "Cite this repository" button renders. The Zenodo DOI will be added once the release is archived.

## License

Everything in this repository — code, dataset, and snapshots — is released under the MIT License. See `LICENSE`.
