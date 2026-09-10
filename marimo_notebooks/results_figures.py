import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Results Figures

    Generate publication-quality Plotly figures from experiment analysis results.
    Supports interactive exploration (Marimo UI) and batch export (script mode).

    **Script mode:**
    ```
    uv run python -m marimo_notebooks.results_figures \
        --snapshot analysis_outputs/stage_04/SNAPSHOT.json \
        [--ablation-snapshot analysis_outputs/stage_05/SNAPSHOT.json] \
        [--ra-snapshot analysis_outputs/rationale_analysis/PRIMARY_RA.json] \
        [--ra-snapshot-ablation analysis_outputs/rationale_analysis/ABLATION_RA.json] \
        [--output-dir .] [--format png]
    ```

    Figures are written under ``{output-dir}/figures`` and tables under
    ``{output-dir}/tables``. The two artifact families share the output root
    so a single ``--output-dir`` flag controls both.
    """)
    return


@app.cell(hide_code=True)
def _(Path, argparse, parse_script_mode_args, sys):
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--snapshot", type=str, default=None)
    _parser.add_argument("--ablation-snapshot", type=str, default=None)
    _parser.add_argument("--ra-snapshot", type=str, default=None)
    _parser.add_argument("--ra-snapshot-ablation", type=str, default=None)
    _parser.add_argument("--probe-audit-dir", type=str,
                         default="analysis_outputs/probe_audit")
    _parser.add_argument("--output-dir", type=str, default=".")
    _parser.add_argument("--format", type=str, default="png",
                         choices=["png", "svg", "pdf"])
    _args = parse_script_mode_args(
        _parser,
        (
            "--snapshot", "--ablation-snapshot", "--ra-snapshot",
            "--ra-snapshot-ablation", "--probe-audit-dir", "--output-dir",
            "--format",
        ),
        sys.argv[1:],
    )

    cli_snapshot_path = _args.snapshot
    cli_ablation_snapshot_path = _args.ablation_snapshot
    cli_ra_snapshot_path = _args.ra_snapshot
    cli_ra_snapshot_ablation_path = _args.ra_snapshot_ablation
    cli_probe_audit_dir = _args.probe_audit_dir
    cli_output_dir = _args.output_dir
    cli_format = _args.format
    is_script_mode = cli_snapshot_path is not None

    # Subdirectories under ``cli_output_dir`` keep figure-like artifacts and
    # tabular artifacts separated so retrieval is unambiguous and the two
    # families can grow independently. Both directories are created on demand
    # by ``export_figure`` / ``export_table_markdown``.
    FIGURES_SUBDIR = Path("figures")
    TABLES_SUBDIR = Path("tables")
    return (
        FIGURES_SUBDIR,
        TABLES_SUBDIR,
        cli_ablation_snapshot_path,
        cli_format,
        cli_output_dir,
        cli_probe_audit_dir,
        cli_ra_snapshot_ablation_path,
        cli_ra_snapshot_path,
        cli_snapshot_path,
        is_script_mode,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Data Source Selection

    Select the experiment and rationale analysis snapshots to use for figure generation.
    In script mode, these are provided via CLI arguments (`--snapshot`, `--ablation-snapshot`,
    `--ra-snapshot`, `--ra-snapshot-ablation`).
    """)
    return


@app.cell(hide_code=True)
def _(ANALYSIS_OUTPUTS_DIR, is_script_mode, mo):
    """Interactive file browsers for snapshot selection (hidden in script mode)."""
    if is_script_mode:
        snapshot_browser = None
        ablation_snapshot_browser = None
        ra_primary_browser = None
        ra_ablation_browser = None
    else:
        snapshot_browser = mo.ui.file_browser(
            initial_path=str(ANALYSIS_OUTPUTS_DIR) + "/",
            filetypes=[".json"],
            selection_mode="file",
            multiple=False,
            label="### Select primary experiment snapshot:",
        )
        ablation_snapshot_browser = mo.ui.file_browser(
            initial_path=str(ANALYSIS_OUTPUTS_DIR) + "/",
            filetypes=[".json"],
            selection_mode="file",
            multiple=False,
            label="### Select ablation experiment snapshot (optional):",
        )
        _ra_dir = ANALYSIS_OUTPUTS_DIR / "rationale_analysis"
        ra_primary_browser = mo.ui.file_browser(
            initial_path=str(_ra_dir) + "/",
            filetypes=[".json"],
            selection_mode="file",
            multiple=False,
            label="### Select primary rationale analysis snapshot (optional):",
        )
        ra_ablation_browser = mo.ui.file_browser(
            initial_path=str(_ra_dir) + "/",
            filetypes=[".json"],
            selection_mode="file",
            multiple=False,
            label="### Select ablation rationale analysis snapshot (optional):",
        )

    mo.vstack([
        b for b in [
            snapshot_browser, ablation_snapshot_browser,
            ra_primary_browser, ra_ablation_browser,
        ] if b is not None
    ]) if not is_script_mode else mo.md("")
    return (
        ablation_snapshot_browser,
        ra_ablation_browser,
        ra_primary_browser,
        snapshot_browser,
    )


@app.cell(hide_code=True)
def _(
    ExperimentAnalysisResults,
    Path,
    cli_snapshot_path,
    is_script_mode,
    mo,
    snapshot_browser,
):
    """Load primary experiment results from pre-computed snapshot."""
    results = None
    snapshot_resolved_path = None

    if is_script_mode:
        _snapshot = Path(cli_snapshot_path)
        mo.stop(
            not _snapshot.exists(),
            mo.callout(mo.md(f"**Error:** Snapshot not found: `{_snapshot}`"), kind="danger"),
        )
        results = ExperimentAnalysisResults.model_validate_json(
            _snapshot.read_text(),
        )
        snapshot_resolved_path = _snapshot
    else:
        mo.stop(
            not snapshot_browser or not snapshot_browser.value,
            mo.callout(mo.md("Select a primary experiment snapshot to begin."), kind="info"),
        )
        _selected_path = Path(snapshot_browser.value[0].path)
        results = ExperimentAnalysisResults.model_validate_json(
            _selected_path.read_text(),
        )
        snapshot_resolved_path = _selected_path

    mo.stop(
        results is None,
        mo.callout(mo.md("**Error:** No results loaded."), kind="danger"),
    )

    mo.callout(
        mo.md(
            f"**Loaded:** {len(results.config_summaries)} configurations, "
            f"{len(results.trial_outcomes):,} trial outcomes"
        ),
        kind="success",
    )
    return results, snapshot_resolved_path


@app.cell(hide_code=True)
def _(
    ExperimentAnalysisResults,
    Path,
    ablation_snapshot_browser,
    cli_ablation_snapshot_path,
    is_script_mode,
):
    """Load ablation experiment results from snapshot (optional)."""
    ablation_results = None
    ablation_snapshot_resolved_path = None

    if is_script_mode:
        if cli_ablation_snapshot_path is not None:
            _path = Path(cli_ablation_snapshot_path)
            if _path.exists():
                ablation_results = ExperimentAnalysisResults.model_validate_json(
                    _path.read_text(),
                )
                ablation_snapshot_resolved_path = _path
    else:
        if ablation_snapshot_browser and ablation_snapshot_browser.value:
            _path = Path(ablation_snapshot_browser.value[0].path)
            ablation_results = ExperimentAnalysisResults.model_validate_json(
                _path.read_text(),
            )
            ablation_snapshot_resolved_path = _path
    return ablation_results, ablation_snapshot_resolved_path


@app.cell(hide_code=True)
def _(
    Path,
    RationaleAnalysisResults,
    cli_ra_snapshot_ablation_path,
    cli_ra_snapshot_path,
    is_script_mode,
    ra_ablation_browser,
    ra_primary_browser,
):
    """Load rationale analysis results: primary and ablation (both optional)."""
    # Primary RA (Stage 4)
    ra_results = None
    ra_snapshot_resolved_path = None
    if is_script_mode:
        if cli_ra_snapshot_path is not None:
            _path = Path(cli_ra_snapshot_path)
            if _path.exists():
                ra_results = RationaleAnalysisResults.model_validate_json(
                    _path.read_text(),
                )
                ra_snapshot_resolved_path = _path
    else:
        if ra_primary_browser and ra_primary_browser.value:
            _path = Path(ra_primary_browser.value[0].path)
            ra_results = RationaleAnalysisResults.model_validate_json(
                _path.read_text(),
            )
            ra_snapshot_resolved_path = _path

    # Ablation RA (Stage 5)
    ra_results_ablation = None
    ra_ablation_snapshot_resolved_path = None
    if is_script_mode:
        if cli_ra_snapshot_ablation_path is not None:
            _path = Path(cli_ra_snapshot_ablation_path)
            if _path.exists():
                ra_results_ablation = RationaleAnalysisResults.model_validate_json(
                    _path.read_text(),
                )
                ra_ablation_snapshot_resolved_path = _path
    else:
        if ra_ablation_browser and ra_ablation_browser.value:
            _path = Path(ra_ablation_browser.value[0].path)
            ra_results_ablation = RationaleAnalysisResults.model_validate_json(
                _path.read_text(),
            )
            ra_ablation_snapshot_resolved_path = _path
    return (
        ra_ablation_snapshot_resolved_path,
        ra_results,
        ra_results_ablation,
        ra_snapshot_resolved_path,
    )


@app.cell(hide_code=True)
def _(
    ablation_results,
    aggregate_failure_mode_rates,
    analyze_cross_dataset,
    balanced_accuracy_leaderboard,
    cell_resolution_by_config,
    combined_balanced_accuracy_leaderboard,
    combined_cost_vs_balanced_accuracy,
    combined_degradation_test_summary,
    combined_model_size_level_detail_rows,
    combined_model_size_summary_rows,
    condition_accuracy_by_config,
    cost_vs_balanced_accuracy,
    cross_dataset_significance_data,
    degradation_test_summary,
    example_accuracy_matrix,
    failure_mode_by_config,
    failure_mode_dominance_matrix,
    failure_mode_rates_by_config,
    marginal_cost_per_balanced_accuracy_point,
    misattribution_rates_by_condition,
    model_size_level_detail_rows,
    model_size_summary_rows,
    primary_vs_ablation_deltas,
    ra_results,
    ra_results_ablation,
    reasoning_effort_curves,
    reasoning_token_scaling,
    results,
    structural_config_order,
    valid_trial_disclosure,
):
    """Prepare data views for all figures (primary and ablation)."""
    # Full config key universes for zero-fill in the dominance matrix and the
    # per-configuration failure mode table. The failure mode charts take an
    # ordered universe instead: the shared structural order.
    all_config_keys = tuple(s.config_key for s in results.config_summaries)
    abl_config_keys = (
        tuple(s.config_key for s in ablation_results.config_summaries)
        if ablation_results is not None
        else ()
    )

    # Primary experiment data views
    cond_acc_data = condition_accuracy_by_config(results)
    example_data = example_accuracy_matrix(results)
    effort_data = reasoning_effort_curves(results)
    cost_data = cost_vs_balanced_accuracy(results)
    token_scaling_data = reasoning_token_scaling(results)
    marginal_data = marginal_cost_per_balanced_accuracy_point(
        results,
    )
    degradation_data = degradation_test_summary(results)
    model_size_worst_data = model_size_summary_rows(
        results, comparison_type="worst",
        authoring_model_slug="claude-opus-4-6",
    )
    model_size_best_data = model_size_summary_rows(
        results, comparison_type="best",
        authoring_model_slug="claude-opus-4-6",
    )
    model_size_detail_data = model_size_level_detail_rows(
        results, authoring_model_slug="claude-opus-4-6",
    )
    resolution_data = cell_resolution_by_config(results)
    leaderboard_data = balanced_accuracy_leaderboard(
        results,
    )
    disclosure_primary = valid_trial_disclosure(results)

    # Ablation experiment data views (empty when no ablation loaded)
    abl_cond_acc_data = (
        condition_accuracy_by_config(ablation_results)
        if ablation_results is not None else []
    )
    abl_example_data = (
        example_accuracy_matrix(ablation_results)
        if ablation_results is not None else []
    )
    abl_effort_data = (
        reasoning_effort_curves(ablation_results)
        if ablation_results is not None else []
    )
    abl_cost_data = (
        cost_vs_balanced_accuracy(ablation_results)
        if ablation_results is not None else []
    )
    abl_token_scaling_data = (
        reasoning_token_scaling(ablation_results)
        if ablation_results is not None else []
    )
    abl_marginal_data = (
        marginal_cost_per_balanced_accuracy_point(
            ablation_results,
        )
        if ablation_results is not None else []
    )
    abl_degradation_data = (
        degradation_test_summary(ablation_results)
        if ablation_results is not None else []
    )
    abl_resolution_data = (
        cell_resolution_by_config(ablation_results)
        if ablation_results is not None else []
    )
    abl_leaderboard_data = (
        balanced_accuracy_leaderboard(
            ablation_results,
        )
        if ablation_results is not None else []
    )
    disclosure_ablation = (
        valid_trial_disclosure(ablation_results)
        if ablation_results is not None else None
    )

    # Cross-dataset and RA data views
    ablation_delta_data = (
        primary_vs_ablation_deltas(results, ablation_results)
        if ablation_results is not None
        else []
    )
    # Both experiments' failure mode figures share the one structural order
    # (grouped by model, ordered by reasoning effort level), computed over the
    # union of both experiments' rows so paired panels stay column-aligned
    # even if one experiment were missing a configuration. The ablation guard
    # names ablation_results as well, because the config universe comes from
    # it while the records come from the RA snapshot.
    _shared_order = tuple(structural_config_order(cond_acc_data, abl_cond_acc_data))
    failure_data = (
        failure_mode_by_config(ra_results, config_order=_shared_order)
        if ra_results is not None
        else []
    )
    failure_data_ablation = (
        failure_mode_by_config(ra_results_ablation, config_order=_shared_order)
        if ra_results_ablation is not None and ablation_results is not None
        else []
    )
    failure_rate_data = (
        failure_mode_rates_by_config(ra_results, config_order=_shared_order)
        if ra_results is not None
        else []
    )
    failure_rate_data_ablation = (
        failure_mode_rates_by_config(
            ra_results_ablation, config_order=_shared_order,
        )
        if ra_results_ablation is not None and ablation_results is not None
        else []
    )
    articulation_rates_primary = (
        aggregate_failure_mode_rates(ra_results)
        if ra_results is not None
        else []
    )
    articulation_rates_ablation = (
        aggregate_failure_mode_rates(ra_results_ablation)
        if ra_results_ablation is not None
        else []
    )
    misattribution_rates_primary = (
        misattribution_rates_by_condition(ra_results)
        if ra_results is not None
        else []
    )
    misattribution_rates_ablation = (
        misattribution_rates_by_condition(ra_results_ablation)
        if ra_results_ablation is not None
        else []
    )
    failure_mode_dominance_data = (
        failure_mode_dominance_matrix(ra_results, all_config_keys)
        if ra_results is not None
        else []
    )
    failure_mode_dominance_data_ablation = (
        failure_mode_dominance_matrix(ra_results_ablation, abl_config_keys)
        if ra_results_ablation is not None
        else []
    )

    # Combined (primary + ablation) data views
    combined_degradation_data = (
        combined_degradation_test_summary(results, ablation_results)
        if ablation_results is not None
        else []
    )
    combined_model_size_worst_data = (
        combined_model_size_summary_rows(
            results, ablation_results, comparison_type="worst",
            authoring_model_slug="claude-opus-4-6",
        )
        if ablation_results is not None
        else []
    )
    combined_model_size_best_data = (
        combined_model_size_summary_rows(
            results, ablation_results, comparison_type="best",
            authoring_model_slug="claude-opus-4-6",
        )
        if ablation_results is not None
        else []
    )
    combined_model_size_detail_data = (
        combined_model_size_level_detail_rows(
            results, ablation_results,
            authoring_model_slug="claude-opus-4-6",
        )
        if ablation_results is not None
        else []
    )
    combined_leaderboard_data = (
        combined_balanced_accuracy_leaderboard(
            results, ablation_results,
        )
        if ablation_results is not None
        else []
    )
    combined_cost_data = (
        combined_cost_vs_balanced_accuracy(
            results, ablation_results,
        )
        if ablation_results is not None
        else []
    )

    # Cross-dataset significance tests, per docs/analysis_design.md
    # Section 12d.
    if ablation_results is not None:
        _xd_results = analyze_cross_dataset(results, ablation_results)
        cond_significance_data = cross_dataset_significance_data(
            list(_xd_results.condition_level_comparisons),
        )
        config_significance_data = cross_dataset_significance_data(
            list(_xd_results.config_level_comparisons),
        )
    else:
        cond_significance_data = []
        config_significance_data = []
    return (
        abl_cond_acc_data,
        abl_cost_data,
        abl_degradation_data,
        abl_effort_data,
        abl_example_data,
        abl_leaderboard_data,
        abl_marginal_data,
        abl_resolution_data,
        abl_token_scaling_data,
        ablation_delta_data,
        all_config_keys,
        articulation_rates_ablation,
        articulation_rates_primary,
        combined_cost_data,
        combined_degradation_data,
        combined_leaderboard_data,
        combined_model_size_best_data,
        combined_model_size_detail_data,
        combined_model_size_worst_data,
        cond_acc_data,
        cond_significance_data,
        config_significance_data,
        cost_data,
        degradation_data,
        disclosure_ablation,
        disclosure_primary,
        effort_data,
        example_data,
        failure_data,
        failure_data_ablation,
        failure_mode_dominance_data,
        failure_mode_dominance_data_ablation,
        failure_rate_data,
        failure_rate_data_ablation,
        leaderboard_data,
        marginal_data,
        misattribution_rates_ablation,
        misattribution_rates_primary,
        model_size_best_data,
        model_size_detail_data,
        model_size_worst_data,
        resolution_data,
        token_scaling_data,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Primary Experiment

    ### Performance [Primary]
    """)
    return


@app.cell(hide_code=True)
def _(cost_data, mo, results, usage_cost_summary):
    """Primary experiment dashboard: configs, trials, tokens, cost."""
    _exp_usage = usage_cost_summary(list(results.trial_outcomes))

    _stat_items = [
        mo.stat(
            value=f"{len(results.config_summaries)}",
            label="Configurations",
            bordered=True,
        ),
        mo.stat(
            value=f"{_exp_usage.num_trials:,}",
            label="Total Trials",
            bordered=True,
        ),
        mo.stat(
            value=f"{_exp_usage.total_input_tokens:,}",
            label="Input Tokens",
            caption=(
                f"avg {_exp_usage.total_input_tokens / _exp_usage.num_trials:,.0f}/trial"
                if _exp_usage.num_trials > 0 else ""
            ),
            bordered=True,
        ),
        mo.stat(
            value=f"{_exp_usage.total_output_tokens:,}",
            label="Output Tokens",
            caption=(
                f"avg {_exp_usage.total_output_tokens / _exp_usage.num_trials:,.0f}/trial"
                if _exp_usage.num_trials > 0 else ""
            ),
            bordered=True,
        ),
    ]

    if _exp_usage.total_reasoning_tokens:
        _stat_items.append(
            mo.stat(
                value=f"{_exp_usage.total_reasoning_tokens:,}",
                label="Reasoning Tokens",
                caption=(
                    f"avg {_exp_usage.total_reasoning_tokens / _exp_usage.num_trials:,.0f}/trial"
                    if _exp_usage.num_trials > 0 else ""
                ),
                bordered=True,
            ),
        )

    if _exp_usage.cost_available:
        _stat_items.append(
            mo.stat(
                value=f"${_exp_usage.actual_cost.total_cost:.2f}",
                label="Actual Cost",
                caption=(
                    f"avg ${_exp_usage.actual_cost.total_cost / _exp_usage.num_trials:.4f}/trial"
                    if _exp_usage.num_trials > 0 else ""
                ),
                bordered=True,
            ),
        )
        if _exp_usage.cache_savings > 0:
            _stat_items.append(
                mo.stat(
                    value=f"${_exp_usage.cache_savings:.2f}",
                    label="Cache Savings",
                    caption=f"vs ${_exp_usage.worst_case_cost.total_cost:.2f} worst-case",
                    bordered=True,
                ),
            )

    _ = cost_data  # Ensure data is loaded before rendering dashboard

    mo.hstack(_stat_items, justify="start", gap=1)
    return


@app.cell(hide_code=True)
def _(build_condition_accuracy_chart, cond_acc_data, mo):
    """Figure 1: Condition accuracy grouped bar chart."""
    fig_condition_accuracy = build_condition_accuracy_chart(
        cond_acc_data, experiment_label="Primary",
    )
    mo.ui.plotly(fig_condition_accuracy)
    return


@app.cell(hide_code=True)
def _(abl_example_data, example_data, structural_config_order):
    """Structural column order shared across primary and ablation heatmaps
    so paired figures stay axis-aligned cell-for-cell.

    Computed once over the union of both experiments' rows and reused by
    every heatmap. Rows need no shared order machinery: every heatmap
    renders base examples in one fixed order.
    """
    heatmap_config_order = structural_config_order(example_data, abl_example_data)
    return (heatmap_config_order,)


@app.cell(hide_code=True)
def _(
    build_example_condition_accuracy_heatmap,
    example_data,
    heatmap_config_order,
    mo,
):
    """Figure 2: Example condition accuracy heatmaps — one per condition, stacked vertically.

    Each figure is wrapped in mo.ui.plotly() to capture click selections;
    downstream cells use .value to resolve which cell the researcher selected.
    Rows run in the fixed base-example order and columns follow the shared
    structural order, so the paired primary+ablation figures are axis-aligned
    cell-for-cell.
    """
    fig_heatmap_correct = build_example_condition_accuracy_heatmap(
        example_data,
        value_column="correct_acc",
        config_order=heatmap_config_order,
        experiment_label="Primary",
    )
    fig_heatmap_transparent = build_example_condition_accuracy_heatmap(
        example_data,
        value_column="transparent_acc",
        config_order=heatmap_config_order,
        experiment_label="Primary",
    )
    fig_heatmap_opaque = build_example_condition_accuracy_heatmap(
        example_data,
        value_column="opaque_acc",
        config_order=heatmap_config_order,
        experiment_label="Primary",
    )
    plotly_heatmap_correct = mo.ui.plotly(fig_heatmap_correct)
    plotly_heatmap_transparent = mo.ui.plotly(fig_heatmap_transparent)
    plotly_heatmap_opaque = mo.ui.plotly(fig_heatmap_opaque)
    mo.vstack([plotly_heatmap_correct, plotly_heatmap_transparent, plotly_heatmap_opaque])
    return (
        plotly_heatmap_correct,
        plotly_heatmap_opaque,
        plotly_heatmap_transparent,
    )


@app.cell(hide_code=True)
def _(build_reasoning_effort_chart, effort_data, mo):
    """Figure 3: Reasoning effort line chart."""
    fig_reasoning_effort = build_reasoning_effort_chart(
        effort_data, experiment_label="Primary",
    )
    mo.ui.plotly(fig_reasoning_effort)
    return


@app.cell(hide_code=True)
def _(build_failure_mode_chart_correct, failure_data, mo):
    """Figure 4: Correct-draft failure mode chart — primary rationale analysis."""
    mo.stop(
        not failure_data,
        mo.callout(
            mo.md("Failure mode chart requires primary rationale analysis data. "
                   "Provide `--ra-snapshot` in script mode."),
            kind="info",
        ),
    )
    fig_failure_modes_correct = build_failure_mode_chart_correct(
        failure_data, experiment_label="Primary",
    )
    mo.ui.plotly(fig_failure_modes_correct)
    return


@app.cell(hide_code=True)
def _(build_failure_mode_chart_incorrect, failure_data, mo):
    """Figure 4: Incorrect-draft failure mode chart — primary rationale analysis."""
    mo.stop(
        not failure_data,
        mo.callout(
            mo.md("Failure mode chart requires primary rationale analysis data. "
                   "Provide `--ra-snapshot` in script mode."),
            kind="info",
        ),
    )
    fig_failure_modes_incorrect = build_failure_mode_chart_incorrect(
        failure_data, experiment_label="Primary",
    )
    mo.ui.plotly(fig_failure_modes_incorrect)
    return


@app.cell(hide_code=True)
def _(build_failure_mode_rate_chart_correct, failure_rate_data, mo):
    """Correct-draft failure mode rate chart — primary rationale analysis."""
    mo.stop(
        not failure_rate_data,
        mo.callout(
            mo.md("Failure mode rate chart requires primary rationale analysis data."),
            kind="info",
        ),
    )
    fig_failure_mode_rates_correct = build_failure_mode_rate_chart_correct(
        failure_rate_data, experiment_label="Primary",
    )
    mo.ui.plotly(fig_failure_mode_rates_correct)
    return


@app.cell(hide_code=True)
def _(build_failure_mode_rate_chart_incorrect, failure_rate_data, mo):
    """Incorrect-draft failure mode rate chart — primary rationale analysis."""
    mo.stop(
        not failure_rate_data,
        mo.callout(
            mo.md("Failure mode rate chart requires primary rationale analysis data."),
            kind="info",
        ),
    )
    fig_failure_mode_rates_incorrect = build_failure_mode_rate_chart_incorrect(
        failure_rate_data, experiment_label="Primary",
    )
    mo.ui.plotly(fig_failure_mode_rates_incorrect)
    return


@app.cell(hide_code=True)
def _(
    articulation_rates_ablation,
    articulation_rates_primary,
    build_articulation_governing_table,
    format_articulation_governing_rows,
    mo,
):
    """Table T05: Per-condition articulation and governing rate summary."""
    mo.stop(
        not articulation_rates_primary or not articulation_rates_ablation,
        mo.callout(
            mo.md("Articulation/governing rate table requires both primary and ablation "
                   "rationale analysis data."),
            kind="info",
        ),
    )
    articulation_display_rows = format_articulation_governing_rows(
        articulation_rates_primary, articulation_rates_ablation,
    )
    tbl_articulation_governing = build_articulation_governing_table(
        articulation_rates_primary,
        articulation_rates_ablation,
        title="Per-Condition Articulation and Governing Rates [Primary vs. Ablation]",
    )
    mo.vstack([
        mo.md("**Per-Condition Articulation and Governing Rates**"),
        mo.ui.tabs(
            {
                "Interactive Table": mo.ui.table(
                    data=articulation_display_rows, selection=None, pagination=False,
                ),
                "Export Preview": mo.ui.plotly(tbl_articulation_governing),
            },
            lazy=True,
        ),
    ])
    return (articulation_display_rows,)


@app.cell(hide_code=True)
def _(
    build_misattribution_rates_table,
    format_misattribution_rates_rows,
    misattribution_rates_ablation,
    misattribution_rates_primary,
    mo,
):
    """Table T06: Per-condition misattribution flag rate summary."""
    # No data-shape mo.stop here: the outputs are defined unconditionally so
    # the script-mode export cell never sits downstream of a stop. The export
    # itself is gated on the display rows being non-empty (both experiments
    # present), and the formatter returns [] when either is missing.
    misattribution_display_rows = format_misattribution_rates_rows(
        misattribution_rates_primary, misattribution_rates_ablation,
    )
    tbl_misattribution_rates = build_misattribution_rates_table(
        misattribution_rates_primary,
        misattribution_rates_ablation,
        title="Per-Condition Misattribution Flag Rates [Primary vs. Ablation]",
    )
    if misattribution_display_rows:
        _view = mo.vstack([
            mo.md("**Per-Condition Misattribution Flag Rates**"),
            mo.ui.tabs(
                {
                    "Interactive Table": mo.ui.table(
                        data=misattribution_display_rows, selection=None, pagination=False,
                    ),
                    "Export Preview": mo.ui.plotly(tbl_misattribution_rates),
                },
                lazy=True,
            ),
        ])
    else:
        _view = mo.callout(
            mo.md("Misattribution rate table requires both primary and ablation "
                   "rationale analysis data."),
            kind="info",
        )
    _view
    return (misattribution_display_rows,)


@app.cell(hide_code=True)
def _(
    build_failure_mode_dominance_heatmap,
    failure_mode_dominance_data,
    heatmap_config_order,
    mo,
):
    """Figure 4a: Failure mode dominance heatmaps — primary, aligned with Figure 2.

    Always returns all six variables (fig + plotly widget per condition).
    When RA data is absent, returns None sentinels so downstream click
    resolution cells remain functional for accuracy-only heatmaps. Rows run
    in the fixed base-example order and columns follow the shared structural
    order, so this view aligns cell-for-cell with Figure 2 and with the
    ablation Figure 4c.
    """
    if not failure_mode_dominance_data:
        fig_fm_dominance_correct = None
        fig_fm_dominance_transparent = None
        fig_fm_dominance_opaque = None
        plotly_fm_dominance_correct = None
        plotly_fm_dominance_transparent = None
        plotly_fm_dominance_opaque = None
        _output = mo.callout(
            mo.md("Failure mode dominance heatmap requires primary rationale analysis data. "
                   "Provide `--ra-snapshot` in script mode."),
            kind="info",
        )
    else:
        fig_fm_dominance_correct = build_failure_mode_dominance_heatmap(
            failure_mode_dominance_data,
            value_column="correct_dominance",
            config_order=heatmap_config_order,
                experiment_label="Primary",
        )
        fig_fm_dominance_transparent = build_failure_mode_dominance_heatmap(
            failure_mode_dominance_data,
            value_column="transparent_dominance",
            config_order=heatmap_config_order,
                experiment_label="Primary",
        )
        fig_fm_dominance_opaque = build_failure_mode_dominance_heatmap(
            failure_mode_dominance_data,
            value_column="opaque_dominance",
            config_order=heatmap_config_order,
                experiment_label="Primary",
        )
        plotly_fm_dominance_correct = mo.ui.plotly(fig_fm_dominance_correct)
        plotly_fm_dominance_transparent = mo.ui.plotly(fig_fm_dominance_transparent)
        plotly_fm_dominance_opaque = mo.ui.plotly(fig_fm_dominance_opaque)
        _output = mo.vstack([
            plotly_fm_dominance_correct,
            plotly_fm_dominance_transparent,
            plotly_fm_dominance_opaque,
        ])
    _output
    return (
        plotly_fm_dominance_correct,
        plotly_fm_dominance_opaque,
        plotly_fm_dominance_transparent,
    )


@app.cell(hide_code=True)
def _(mo):
    """State for tracking previous heatmap widget selections.

    Each mo.ui.plotly() widget retains its value indefinitely, so we snapshot
    all widget values after each resolver run to detect which widget actually
    changed on the next run.
    """
    get_prev_primary_clicks, set_prev_primary_clicks = mo.state({})
    get_prev_ablation_clicks, set_prev_ablation_clicks = mo.state({})
    return (
        get_prev_ablation_clicks,
        get_prev_primary_clicks,
        set_prev_ablation_clicks,
        set_prev_primary_clicks,
    )


@app.cell(hide_code=True)
def _(re):
    """Pure helper: resolve a heatmap click to full identifiers.

    Supports change detection via prev_selections to avoid stale selections
    on one heatmap shadowing newer clicks on a different heatmap.
    """
    _BASE_EXAMPLE_RE = re.compile(r"(example_\d+)")

    def _try_resolve_point(point, column_to_config, condition):
        """Extract (config_key, base_example, condition, z_value) from a plotly point.

        A clicked heatmap point reports its numeric gutter column position
        as ``x`` (the click payload may deliver it as an int, float, or
        numeric string); a click that carries no numeric x, or lands on a
        blank gutter column, resolves to None.
        """
        try:
            config_key = column_to_config.get(
                int(round(float(point.get("x")))),
            )
        except (TypeError, ValueError):
            config_key = None
        match = _BASE_EXAMPLE_RE.match(str(point.get("y", "")))
        base_example = match.group(1) if match else None
        if config_key and base_example:
            return (config_key, base_example, condition, point.get("z", 0.0))
        return None

    def resolve_heatmap_click(
        column_to_config: dict[int, str],
        widgets_and_conditions: list[tuple[object, str]],
        prev_selections: dict[int, list] | None = None,
    ) -> tuple[str, str, str, float] | None:
        """Match a plotly click event back to (config_key, base_example, condition, z_value).

        Each mo.ui.plotly() widget retains its selection indefinitely, so when
        multiple widgets have values, we need to identify which one *changed*
        since the last run.  When prev_selections is provided, widgets whose
        .value differs from the stored snapshot are checked first.  Falls back
        to first-match on initial load (prev_selections is None or empty).
        """
        # First pass: prioritize widgets whose value changed
        if prev_selections:
            for i, (widget, condition) in enumerate(widgets_and_conditions):
                points = widget.value if widget is not None else []
                if not points or points == prev_selections.get(i, []):
                    continue
                result = _try_resolve_point(points[0], column_to_config, condition)
                if result:
                    return result

        # Second pass: first widget with any selection (initial load / no change found)
        for widget, condition in widgets_and_conditions:
            points = widget.value if widget is not None else []
            if not points:
                continue
            result = _try_resolve_point(points[0], column_to_config, condition)
            if result:
                return result

        return None

    return (resolve_heatmap_click,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Balanced Accuracy Leaderboard [Primary]
    """)
    return


@app.cell(hide_code=True)
def _(
    build_balanced_accuracy_table,
    disclosure_primary,
    format_leaderboard_rows,
    leaderboard_data,
    mo,
):
    """Table T00: Balanced accuracy leaderboard (primary experiment only)."""
    leaderboard_display_rows = format_leaderboard_rows(
        leaderboard_data, disclosure=disclosure_primary,
    )
    _title = "Balanced Accuracy Leaderboard [Primary]"
    tbl_balanced_accuracy = build_balanced_accuracy_table(
        leaderboard_data,
        title=_title,
        disclosure=disclosure_primary,
    )
    mo.ui.tabs(
        {
            "Interactive Table": mo.ui.table(
                data=leaderboard_display_rows,
                selection=None,
                label=_title,
                pagination=False,
            ),
            "Export Preview": mo.ui.plotly(tbl_balanced_accuracy),
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Open Cell Inspector [Primary] ↗
    """)
    return


@app.cell(hide_code=True)
def _(
    STIMULUS_METADATA_REGISTRY,
    get_prev_primary_clicks,
    heatmap_column_config,
    heatmap_config_order,
    is_script_mode,
    mo,
    plotly_fm_dominance_correct,
    plotly_fm_dominance_opaque,
    plotly_fm_dominance_transparent,
    plotly_heatmap_correct,
    plotly_heatmap_opaque,
    plotly_heatmap_transparent,
    ra_snapshot_resolved_path,
    resolve_heatmap_click,
    set_prev_primary_clicks,
    snapshot_resolved_path,
    urlencode,
):
    """Selection context + cell inspector link for primary heatmaps."""
    mo.stop(is_script_mode)

    _column_to_config = heatmap_column_config(list(heatmap_config_order))

    _widgets_list = [
        (plotly_heatmap_correct, "correct"),
        (plotly_heatmap_transparent, "transparent"),
        (plotly_heatmap_opaque, "opaque"),
        (plotly_fm_dominance_correct, "correct"),
        (plotly_fm_dominance_transparent, "transparent"),
        (plotly_fm_dominance_opaque, "opaque"),
    ]

    _click = resolve_heatmap_click(
        _column_to_config, _widgets_list, get_prev_primary_clicks(),
    )

    # Snapshot current widget values for change detection on next run
    set_prev_primary_clicks({
        i: (w.value if w is not None else [])
        for i, (w, _) in enumerate(_widgets_list)
    })
    mo.stop(
        _click is None,
        mo.callout(
            mo.md("**Zoom**: Click any primary heatmap cell above to inspect it in detail."),
            kind="neutral",
        ),
    )

    _config_key, _base_example, _condition, _z_value = _click
    _meta = STIMULUS_METADATA_REGISTRY.get(_base_example)
    _domain = _meta.domain if _meta else "?"
    _risk = _meta.risk if _meta else "?"

    _params: dict[str, str] = {
        "snapshot": str(snapshot_resolved_path),
        "config": _config_key,
        "example": _base_example,
        "condition": _condition,
    }
    if ra_snapshot_resolved_path:
        _params["ra_snapshot"] = str(ra_snapshot_resolved_path)

    _url = f"/?file=cell_inspector.py&{urlencode(_params)}"
    mo.callout(
        mo.vstack([
            mo.md(
                f"**Selected Cell**: `{_config_key}` × `{_base_example}` "
                f"(**{_condition}**)"
            ),
            mo.md(f"Domain: {_domain} · Risk: {_risk} · Value: {_z_value:.3f}"),
            mo.md(f'<a href="{_url}" target="_blank">Open Cell Inspector ↗</a>'),
        ]),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Cost & Token Analysis [Primary]
    """)
    return


@app.cell(hide_code=True)
def _(build_cost_vs_balanced_accuracy_scatter, cost_data, mo):
    """Figure 6: Cost vs balanced accuracy scatter."""
    fig_cost_balanced_accuracy = build_cost_vs_balanced_accuracy_scatter(
        cost_data, experiment_label="Primary",
    )
    mo.ui.plotly(fig_cost_balanced_accuracy)
    return


@app.cell(hide_code=True)
def _(build_token_composition_chart, mo, token_scaling_data):
    """Figure 7: Token composition stacked bars."""
    fig_token_composition = build_token_composition_chart(
        token_scaling_data, experiment_label="Primary",
    )
    mo.ui.plotly(fig_token_composition)
    return


@app.cell(hide_code=True)
def _(build_reasoning_token_scaling_chart, mo, token_scaling_data):
    """Figure 8: Reasoning token scaling line chart."""
    fig_reasoning_scaling = build_reasoning_token_scaling_chart(
        token_scaling_data, experiment_label="Primary",
    )
    mo.ui.plotly(fig_reasoning_scaling)
    return


@app.cell(hide_code=True)
def _(build_marginal_cost_chart, degradation_data, marginal_data, mo):
    """Figure 9: Marginal cost per balanced accuracy point."""
    mo.stop(
        not marginal_data,
        mo.callout(
            mo.md("No marginal cost data available (requires 2+ reasoning effort levels with cost data)."),
            kind="info",
        ),
    )
    fig_marginal_cost = build_marginal_cost_chart(
        marginal_data, degradation_data=degradation_data,
        experiment_label="Primary",
    )
    mo.ui.plotly(fig_marginal_cost)
    return


@app.cell(hide_code=True)
def _(
    build_degradation_table,
    combined_degradation_data,
    degradation_data,
    format_degradation_rows,
    mo,
):
    """Adjacent-pair degradation tests (combined when ablation available)."""
    _use_combined = bool(combined_degradation_data)
    _data = combined_degradation_data if _use_combined else degradation_data
    mo.stop(
        not _data,
        mo.callout(
            mo.md("No adjacent reasoning effort level pairs in the data."),
            kind="info",
        ),
    )

    degradation_display_rows = format_degradation_rows(
        _data, include_experiment=_use_combined,
    )

    # Per-experiment stats avoid conflating K and counts across experiments
    if _use_combined:
        _primary = [r for r in _data if r["experiment"] == "Primary"]
        _ablation = [r for r in _data if r["experiment"] == "Ablation"]
        _pk = _primary[0]["bonferroni_k"] if _primary else 0
        _ak = _ablation[0]["bonferroni_k"] if _ablation else 0
        _p_sig = sum(1 for r in _primary if r["significant"])
        _a_sig = sum(1 for r in _ablation if r["significant"])
        _title = (
            f"Adjacent-Pair Reasoning Effort Level Degradation Tests [Primary + Ablation] "
            f"(Primary K={_pk}: {_p_sig}/{len(_primary)} significant; "
            f"Ablation K={_ak}: {_a_sig}/{len(_ablation)} significant)"
        )
    else:
        _k = _data[0]["bonferroni_k"]
        _sig_count = sum(1 for r in _data if r["significant"])
        _title = (
            f"Adjacent-Pair Reasoning Effort Level Degradation Tests [Primary] "
            f"(K={_k}, \u03b1=0.05): {_sig_count}/{len(_data)} significant"
        )
    tbl_degradation = build_degradation_table(
        degradation_display_rows,
        title=_title,
    )

    mo.vstack([
        mo.md(f"**{_title}**"),
        mo.ui.tabs(
            {
                "Interactive Table": mo.ui.table(
                    data=degradation_display_rows, selection=None, pagination=False,
                ),
                "Export Preview": mo.ui.plotly(tbl_degradation),
            },
            lazy=True,
        ),
    ])
    return (degradation_display_rows,)


@app.cell(hide_code=True)
def _(
    build_model_size_summary_table,
    combined_model_size_worst_data,
    format_model_size_summary_rows,
    mo,
    model_size_worst_data,
):
    """Every-level model size test (combined when ablation available)."""
    _use_combined = bool(combined_model_size_worst_data)
    _data = combined_model_size_worst_data if _use_combined else model_size_worst_data
    mo.stop(
        not _data,
        mo.callout(
            mo.md("No within-provider model size pairs in the data."),
            kind="info",
        ),
    )

    model_size_worst_display_rows = format_model_size_summary_rows(
        _data, include_experiment=_use_combined,
    )

    _suffix = " [Primary + Ablation]" if _use_combined else " [Primary]"
    _title = (
        "Within-Provider Model Size Comparison Tests, Every-Level"
        f"{_suffix}"
    )
    tbl_model_size_worst = build_model_size_summary_table(
        model_size_worst_display_rows,
        title=_title,
    )

    mo.vstack([
        mo.md(f"**{_title}**"),
        mo.ui.tabs(
            {
                "Interactive Table": mo.ui.table(
                    data=model_size_worst_display_rows,
                    selection=None, pagination=False,
                ),
                "Export Preview": mo.ui.plotly(tbl_model_size_worst),
            },
            lazy=True,
        ),
    ])
    return (model_size_worst_display_rows,)


@app.cell(hide_code=True)
def _(
    build_model_size_summary_table,
    combined_model_size_best_data,
    format_model_size_summary_rows,
    mo,
    model_size_best_data,
):
    """Some-level model size test (combined when ablation available)."""
    _use_combined = bool(combined_model_size_best_data)
    _data = combined_model_size_best_data if _use_combined else model_size_best_data
    mo.stop(
        not _data,
        mo.callout(
            mo.md("No within-provider model size pairs in the data."),
            kind="info",
        ),
    )

    model_size_best_display_rows = format_model_size_summary_rows(
        _data, include_experiment=_use_combined,
    )

    _suffix = " [Primary + Ablation]" if _use_combined else " [Primary]"
    _title = (
        "Within-Provider Model Size Comparison Tests, Some-Level"
        f"{_suffix}"
    )
    tbl_model_size_best = build_model_size_summary_table(
        model_size_best_display_rows,
        title=_title,
    )

    mo.vstack([
        mo.md(f"**{_title}**"),
        mo.ui.tabs(
            {
                "Interactive Table": mo.ui.table(
                    data=model_size_best_display_rows,
                    selection=None, pagination=False,
                ),
                "Export Preview": mo.ui.plotly(tbl_model_size_best),
            },
            lazy=True,
        ),
    ])
    return (model_size_best_display_rows,)


@app.cell(hide_code=True)
def _(
    build_model_size_detail_table,
    combined_model_size_detail_data,
    format_model_size_detail_rows,
    mo,
    model_size_detail_data,
):
    """Per-level audit detail behind the model size tests."""
    _use_combined = bool(combined_model_size_detail_data)
    _data = combined_model_size_detail_data if _use_combined else model_size_detail_data
    mo.stop(
        not _data,
        mo.callout(
            mo.md("No within-provider model size pairs in the data."),
            kind="info",
        ),
    )

    model_size_detail_display_rows = format_model_size_detail_rows(
        _data, include_experiment=_use_combined,
    )

    _suffix = " [Primary + Ablation]" if _use_combined else " [Primary]"
    _title = (
        "Within-Provider Model Size Comparison, Per-Level Detail"
        f"{_suffix}"
    )
    tbl_model_size_detail = build_model_size_detail_table(
        model_size_detail_display_rows, title=_title,
    )

    mo.vstack([
        mo.md(f"**{_title}**"),
        mo.ui.tabs(
            {
                "Interactive Table": mo.ui.table(
                    data=model_size_detail_display_rows,
                    selection=None, pagination=False,
                ),
                "Export Preview": mo.ui.plotly(tbl_model_size_detail),
            },
            lazy=True,
        ),
    ])
    return (model_size_detail_display_rows,)


@app.cell(hide_code=True)
def _(
    build_cost_breakdown_table,
    cost_data,
    format_cost_breakdown_rows,
    format_cost_cell,
    mo,
):
    """Cost breakdown table with per-config token and cost detail."""
    cost_display_rows = format_cost_breakdown_rows(
        cost_data, include_experiment=False,
    )

    _title = "Cost Breakdown by Configuration [Primary]"
    tbl_cost = build_cost_breakdown_table(
        cost_display_rows,
        title=_title,
    )

    # Build format_mapping from format_cost_cell for consistent formatting
    _fmt_columns = [
        "Balanced Accuracy", "Cost per Trial ($)", "Total Cost ($)", "Input Tokens",
        "Output Tokens", "Reasoning Tokens", "Response Tokens",
    ]

    mo.ui.tabs(
        {
            "Interactive Table": mo.ui.table(
                data=cost_display_rows,
                selection=None,
                label="Cost Breakdown by Configuration",
                pagination=False,
                format_mapping={
                    col: lambda v, c=col: format_cost_cell(c, v)
                    for col in _fmt_columns
                },
            ),
            "Export Preview": mo.ui.plotly(tbl_cost),
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Experiment Status [Primary]
    """)
    return


@app.cell(hide_code=True)
def _(build_cell_resolution_chart, mo, resolution_data):
    """Figure 10: Cell resolution status chart."""
    fig_cell_resolution = build_cell_resolution_chart(
        resolution_data, experiment_label="Primary",
    )
    mo.ui.plotly(fig_cell_resolution)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Ablation Experiment

    ### Performance [Ablation]
    """)
    return


@app.cell(hide_code=True)
def _(abl_cost_data, ablation_results, mo, usage_cost_summary):
    """Ablation experiment dashboard: configs, trials, tokens, cost."""
    mo.stop(
        ablation_results is None,
        mo.callout(
            mo.md("Ablation dashboard requires ablation data. "
                   "Provide `--ablation-snapshot` in script mode."),
            kind="info",
        ),
    )

    _exp_usage = usage_cost_summary(list(ablation_results.trial_outcomes))

    _stat_items = [
        mo.stat(
            value=f"{len(ablation_results.config_summaries)}",
            label="Configurations",
            bordered=True,
        ),
        mo.stat(
            value=f"{_exp_usage.num_trials:,}",
            label="Total Trials",
            bordered=True,
        ),
        mo.stat(
            value=f"{_exp_usage.total_input_tokens:,}",
            label="Input Tokens",
            caption=(
                f"avg {_exp_usage.total_input_tokens / _exp_usage.num_trials:,.0f}/trial"
                if _exp_usage.num_trials > 0 else ""
            ),
            bordered=True,
        ),
        mo.stat(
            value=f"{_exp_usage.total_output_tokens:,}",
            label="Output Tokens",
            caption=(
                f"avg {_exp_usage.total_output_tokens / _exp_usage.num_trials:,.0f}/trial"
                if _exp_usage.num_trials > 0 else ""
            ),
            bordered=True,
        ),
    ]

    if _exp_usage.total_reasoning_tokens:
        _stat_items.append(
            mo.stat(
                value=f"{_exp_usage.total_reasoning_tokens:,}",
                label="Reasoning Tokens",
                caption=(
                    f"avg {_exp_usage.total_reasoning_tokens / _exp_usage.num_trials:,.0f}/trial"
                    if _exp_usage.num_trials > 0 else ""
                ),
                bordered=True,
            ),
        )

    if _exp_usage.cost_available:
        _stat_items.append(
            mo.stat(
                value=f"${_exp_usage.actual_cost.total_cost:.2f}",
                label="Actual Cost",
                caption=(
                    f"avg ${_exp_usage.actual_cost.total_cost / _exp_usage.num_trials:.4f}/trial"
                    if _exp_usage.num_trials > 0 else ""
                ),
                bordered=True,
            ),
        )
        if _exp_usage.cache_savings > 0:
            _stat_items.append(
                mo.stat(
                    value=f"${_exp_usage.cache_savings:.2f}",
                    label="Cache Savings",
                    caption=f"vs ${_exp_usage.worst_case_cost.total_cost:.2f} worst-case",
                    bordered=True,
                ),
            )

    _ = abl_cost_data  # Ensure data is loaded before rendering dashboard

    mo.hstack(_stat_items, justify="start", gap=1)
    return


@app.cell(hide_code=True)
def _(abl_cond_acc_data, build_condition_accuracy_chart, mo):
    """Figure 1b: Condition accuracy — ablation experiment."""
    mo.stop(
        not abl_cond_acc_data,
        mo.callout(mo.md("Ablation condition accuracy chart requires ablation data."), kind="info"),
    )
    fig_abl_condition_accuracy = build_condition_accuracy_chart(
        abl_cond_acc_data, experiment_label="Ablation",
    )
    mo.ui.plotly(fig_abl_condition_accuracy)
    return


@app.cell(hide_code=True)
def _(
    abl_example_data,
    build_example_condition_accuracy_heatmap,
    heatmap_config_order,
    mo,
):
    """Figure 2b: Example condition accuracy heatmaps — ablation, one per condition.

    Rendered against the fixed row order and the shared structural column
    order, so this standalone view aligns cell-for-cell with its primary
    counterpart and with the exported paired composite.
    """
    mo.stop(
        not abl_example_data,
        mo.callout(mo.md("Ablation example heatmap requires ablation data."), kind="info"),
    )
    fig_abl_heatmap_correct = build_example_condition_accuracy_heatmap(
        abl_example_data,
        value_column="correct_acc",
        config_order=heatmap_config_order,
        experiment_label="Ablation",
    )
    fig_abl_heatmap_transparent = build_example_condition_accuracy_heatmap(
        abl_example_data,
        value_column="transparent_acc",
        config_order=heatmap_config_order,
        experiment_label="Ablation",
    )
    fig_abl_heatmap_opaque = build_example_condition_accuracy_heatmap(
        abl_example_data,
        value_column="opaque_acc",
        config_order=heatmap_config_order,
        experiment_label="Ablation",
    )
    plotly_abl_heatmap_correct = mo.ui.plotly(fig_abl_heatmap_correct)
    plotly_abl_heatmap_transparent = mo.ui.plotly(fig_abl_heatmap_transparent)
    plotly_abl_heatmap_opaque = mo.ui.plotly(fig_abl_heatmap_opaque)
    mo.vstack([
        plotly_abl_heatmap_correct,
        plotly_abl_heatmap_transparent,
        plotly_abl_heatmap_opaque,
    ])
    return (
        plotly_abl_heatmap_correct,
        plotly_abl_heatmap_opaque,
        plotly_abl_heatmap_transparent,
    )


@app.cell(hide_code=True)
def _(abl_effort_data, build_reasoning_effort_chart, mo):
    """Figure 3b: Reasoning effort — ablation experiment."""
    mo.stop(
        not abl_effort_data,
        mo.callout(mo.md("Ablation reasoning effort chart requires ablation data."), kind="info"),
    )
    fig_abl_reasoning_effort = build_reasoning_effort_chart(
        abl_effort_data, experiment_label="Ablation",
    )
    mo.ui.plotly(fig_abl_reasoning_effort)
    return


@app.cell(hide_code=True)
def _(build_failure_mode_chart_correct, failure_data_ablation, mo):
    """Figure 4b: Correct-draft failure mode chart — ablation rationale analysis."""
    mo.stop(
        not failure_data_ablation,
        mo.callout(
            mo.md("Ablation failure mode chart requires ablation rationale analysis data. "
                   "Provide `--ra-snapshot-ablation` in script mode."),
            kind="info",
        ),
    )
    fig_failure_modes_correct_ablation = build_failure_mode_chart_correct(
        failure_data_ablation, experiment_label="Ablation",
    )
    mo.ui.plotly(fig_failure_modes_correct_ablation)
    return


@app.cell(hide_code=True)
def _(build_failure_mode_chart_incorrect, failure_data_ablation, mo):
    """Figure 4b: Incorrect-draft failure mode chart — ablation rationale analysis."""
    mo.stop(
        not failure_data_ablation,
        mo.callout(
            mo.md("Ablation failure mode chart requires ablation rationale analysis data. "
                   "Provide `--ra-snapshot-ablation` in script mode."),
            kind="info",
        ),
    )
    fig_failure_modes_incorrect_ablation = build_failure_mode_chart_incorrect(
        failure_data_ablation, experiment_label="Ablation",
    )
    mo.ui.plotly(fig_failure_modes_incorrect_ablation)
    return


@app.cell(hide_code=True)
def _(build_failure_mode_rate_chart_correct, failure_rate_data_ablation, mo):
    """Correct-draft failure mode rate chart — ablation rationale analysis."""
    mo.stop(
        not failure_rate_data_ablation,
        mo.callout(
            mo.md("Ablation failure mode rate chart requires ablation rationale analysis data."),
            kind="info",
        ),
    )
    fig_failure_mode_rates_correct_ablation = build_failure_mode_rate_chart_correct(
        failure_rate_data_ablation, experiment_label="Ablation",
    )
    mo.ui.plotly(fig_failure_mode_rates_correct_ablation)
    return


@app.cell(hide_code=True)
def _(build_failure_mode_rate_chart_incorrect, failure_rate_data_ablation, mo):
    """Incorrect-draft failure mode rate chart — ablation rationale analysis."""
    mo.stop(
        not failure_rate_data_ablation,
        mo.callout(
            mo.md("Ablation failure mode rate chart requires ablation rationale analysis data."),
            kind="info",
        ),
    )
    fig_failure_mode_rates_incorrect_ablation = build_failure_mode_rate_chart_incorrect(
        failure_rate_data_ablation, experiment_label="Ablation",
    )
    mo.ui.plotly(fig_failure_mode_rates_incorrect_ablation)
    return


@app.cell(hide_code=True)
def _(
    build_failure_mode_dominance_heatmap,
    failure_mode_dominance_data_ablation,
    heatmap_config_order,
    mo,
):
    """Figure 4c: Failure mode dominance heatmaps — ablation, one per condition.

    Always returns all six variables (fig + plotly widget per condition).
    When ablation RA data is absent, returns None sentinels so downstream
    click resolution cells remain functional for accuracy-only heatmaps.
    Rendered against the fixed row order and the shared structural column
    order, so this view aligns cell-for-cell with Figure 4a and with the
    ablation Figure 2b.
    """
    if not failure_mode_dominance_data_ablation:
        fig_abl_fm_dominance_correct = None
        fig_abl_fm_dominance_transparent = None
        fig_abl_fm_dominance_opaque = None
        plotly_abl_fm_dominance_correct = None
        plotly_abl_fm_dominance_transparent = None
        plotly_abl_fm_dominance_opaque = None
        _output = mo.callout(
            mo.md("Ablation failure mode dominance heatmap requires ablation rationale analysis data. "
                   "Provide `--ra-snapshot-ablation` in script mode."),
            kind="info",
        )
    else:
        fig_abl_fm_dominance_correct = build_failure_mode_dominance_heatmap(
            failure_mode_dominance_data_ablation,
            value_column="correct_dominance",
            config_order=heatmap_config_order,
                experiment_label="Ablation",
        )
        fig_abl_fm_dominance_transparent = build_failure_mode_dominance_heatmap(
            failure_mode_dominance_data_ablation,
            value_column="transparent_dominance",
            config_order=heatmap_config_order,
                experiment_label="Ablation",
        )
        fig_abl_fm_dominance_opaque = build_failure_mode_dominance_heatmap(
            failure_mode_dominance_data_ablation,
            value_column="opaque_dominance",
            config_order=heatmap_config_order,
                experiment_label="Ablation",
        )
        plotly_abl_fm_dominance_correct = mo.ui.plotly(fig_abl_fm_dominance_correct)
        plotly_abl_fm_dominance_transparent = mo.ui.plotly(fig_abl_fm_dominance_transparent)
        plotly_abl_fm_dominance_opaque = mo.ui.plotly(fig_abl_fm_dominance_opaque)
        _output = mo.vstack([
            plotly_abl_fm_dominance_correct,
            plotly_abl_fm_dominance_transparent,
            plotly_abl_fm_dominance_opaque,
        ])
    _output
    return (
        plotly_abl_fm_dominance_correct,
        plotly_abl_fm_dominance_opaque,
        plotly_abl_fm_dominance_transparent,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Balanced Accuracy Leaderboard [Ablation]
    """)
    return


@app.cell(hide_code=True)
def _(
    abl_leaderboard_data,
    build_balanced_accuracy_table,
    disclosure_ablation,
    format_leaderboard_rows,
    mo,
):
    """Table T00b: Balanced accuracy leaderboard for ablation experiment."""
    mo.stop(
        not abl_leaderboard_data,
        mo.callout(mo.md("Ablation leaderboard requires ablation data."), kind="info"),
    )
    leaderboard_display_rows_ablation = format_leaderboard_rows(
        abl_leaderboard_data, disclosure=disclosure_ablation,
    )
    _title = "Balanced Accuracy Leaderboard [Ablation]"
    tbl_balanced_accuracy_ablation = build_balanced_accuracy_table(
        abl_leaderboard_data,
        title=_title,
        disclosure=disclosure_ablation,
    )
    mo.ui.tabs(
        {
            "Interactive Table": mo.ui.table(
                data=leaderboard_display_rows_ablation,
                selection=None,
                label=_title,
                pagination=False,
            ),
            "Export Preview": mo.ui.plotly(tbl_balanced_accuracy_ablation),
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Open Cell Inspector [Ablation] ↗
    """)
    return


@app.cell(hide_code=True)
def _(
    STIMULUS_METADATA_REGISTRY,
    abl_example_data,
    ablation_snapshot_resolved_path,
    get_prev_ablation_clicks,
    heatmap_column_config,
    heatmap_config_order,
    is_script_mode,
    mo,
    plotly_abl_fm_dominance_correct,
    plotly_abl_fm_dominance_opaque,
    plotly_abl_fm_dominance_transparent,
    plotly_abl_heatmap_correct,
    plotly_abl_heatmap_opaque,
    plotly_abl_heatmap_transparent,
    ra_ablation_snapshot_resolved_path,
    resolve_heatmap_click,
    set_prev_ablation_clicks,
    urlencode,
):
    """Selection context + cell inspector link for ablation heatmaps."""
    mo.stop(is_script_mode)
    mo.stop(not abl_example_data)

    _column_to_config = heatmap_column_config(list(heatmap_config_order))

    _widgets_list = [
        (plotly_abl_heatmap_correct, "correct"),
        (plotly_abl_heatmap_transparent, "transparent"),
        (plotly_abl_heatmap_opaque, "opaque"),
        (plotly_abl_fm_dominance_correct, "correct"),
        (plotly_abl_fm_dominance_transparent, "transparent"),
        (plotly_abl_fm_dominance_opaque, "opaque"),
    ]

    _click = resolve_heatmap_click(
        _column_to_config, _widgets_list, get_prev_ablation_clicks(),
    )

    # Snapshot current widget values for change detection on next run
    set_prev_ablation_clicks({
        i: (w.value if w is not None else [])
        for i, (w, _) in enumerate(_widgets_list)
    })
    mo.stop(
        _click is None,
        mo.callout(
            mo.md("**Zoom**: Click any ablation heatmap cell above to inspect it in detail."),
            kind="neutral",
        ),
    )

    _config_key, _base_example, _condition, _z_value = _click
    _meta = STIMULUS_METADATA_REGISTRY.get(_base_example)
    _domain = _meta.domain if _meta else "?"
    _risk = _meta.risk if _meta else "?"

    _params: dict[str, str] = {
        "snapshot": str(ablation_snapshot_resolved_path),
        "config": _config_key,
        "example": _base_example,
        "condition": _condition,
    }
    if ra_ablation_snapshot_resolved_path:
        _params["ra_snapshot"] = str(ra_ablation_snapshot_resolved_path)

    _url = f"/?file=cell_inspector.py&{urlencode(_params)}"
    mo.callout(
        mo.vstack([
            mo.md(
                f"**Selected Cell [Ablation]**: `{_config_key}` × `{_base_example}` "
                f"(**{_condition}**)"
            ),
            mo.md(f"Domain: {_domain} · Risk: {_risk} · Value: {_z_value:.3f}"),
            mo.md(f'<a href="{_url}" target="_blank">Open Cell Inspector ↗</a>'),
        ]),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Cost & Token Analysis [Ablation]
    """)
    return


@app.cell(hide_code=True)
def _(abl_cost_data, build_cost_vs_balanced_accuracy_scatter, mo):
    """Figure 6b: Cost vs balanced accuracy — ablation experiment."""
    mo.stop(
        not abl_cost_data,
        mo.callout(mo.md("Ablation cost vs balanced accuracy chart requires ablation data."), kind="info"),
    )
    fig_abl_cost_balanced_accuracy = build_cost_vs_balanced_accuracy_scatter(
        abl_cost_data, experiment_label="Ablation",
    )
    mo.ui.plotly(fig_abl_cost_balanced_accuracy)
    return


@app.cell(hide_code=True)
def _(abl_token_scaling_data, build_token_composition_chart, mo):
    """Figure 7b: Token composition — ablation experiment."""
    mo.stop(
        not abl_token_scaling_data,
        mo.callout(mo.md("Ablation token composition chart requires ablation data."), kind="info"),
    )
    fig_abl_token_composition = build_token_composition_chart(
        abl_token_scaling_data, experiment_label="Ablation",
    )
    mo.ui.plotly(fig_abl_token_composition)
    return


@app.cell(hide_code=True)
def _(abl_token_scaling_data, build_reasoning_token_scaling_chart, mo):
    """Figure 8b: Reasoning token scaling — ablation experiment."""
    mo.stop(
        not abl_token_scaling_data,
        mo.callout(mo.md("Ablation reasoning token scaling chart requires ablation data."), kind="info"),
    )
    fig_abl_reasoning_scaling = build_reasoning_token_scaling_chart(
        abl_token_scaling_data, experiment_label="Ablation",
    )
    mo.ui.plotly(fig_abl_reasoning_scaling)
    return


@app.cell(hide_code=True)
def _(abl_degradation_data, abl_marginal_data, build_marginal_cost_chart, mo):
    """Figure 9b: Marginal cost — ablation experiment."""
    mo.stop(
        not abl_marginal_data,
        mo.callout(
            mo.md("No ablation marginal cost data available "
                   "(requires ablation data with 2+ reasoning effort levels)."),
            kind="info",
        ),
    )
    fig_abl_marginal_cost = build_marginal_cost_chart(
        abl_marginal_data, degradation_data=abl_degradation_data,
        experiment_label="Ablation",
    )
    mo.ui.plotly(fig_abl_marginal_cost)
    return


@app.cell(hide_code=True)
def _(
    abl_degradation_data,
    build_degradation_table,
    format_degradation_rows,
    mo,
):
    """Adjacent-pair reasoning effort level degradation test results table (ablation experiment)."""
    mo.stop(
        not abl_degradation_data,
        mo.callout(
            mo.md("No adjacent reasoning effort level pairs in the ablation data."),
            kind="info",
        ),
    )

    _k = abl_degradation_data[0]["bonferroni_k"]
    _sig_count = sum(1 for r in abl_degradation_data if r["significant"])

    degradation_display_rows_ablation = format_degradation_rows(
        abl_degradation_data, include_experiment=False,
    )

    _title = (
        f"Adjacent-Pair Reasoning Effort Level Degradation Tests [Ablation] (K={_k}, \u03b1=0.05): "
        f"{_sig_count}/{len(abl_degradation_data)} significant"
    )
    tbl_degradation_ablation = build_degradation_table(
        degradation_display_rows_ablation, title=_title,
    )

    mo.vstack([
        mo.md(f"**{_title}**"),
        mo.ui.tabs(
            {
                "Interactive Table": mo.ui.table(
                    data=degradation_display_rows_ablation,
                    selection=None, pagination=False,
                ),
                "Export Preview": mo.ui.plotly(tbl_degradation_ablation),
            },
            lazy=True,
        ),
    ])
    return


@app.cell(hide_code=True)
def _(build_degradation_table, degradation_data, format_degradation_rows, mo):
    """Adjacent-pair reasoning effort level degradation test results table (primary experiment only)."""
    mo.stop(
        not degradation_data,
        mo.callout(
            mo.md("No adjacent reasoning effort level pairs in the primary data."),
            kind="info",
        ),
    )

    _k = degradation_data[0]["bonferroni_k"]
    _sig_count = sum(1 for r in degradation_data if r["significant"])

    degradation_display_rows_primary = format_degradation_rows(
        degradation_data, include_experiment=False,
    )

    _title = (
        f"Adjacent-Pair Reasoning Effort Level Degradation Tests [Primary] (K={_k}, \u03b1=0.05): "
        f"{_sig_count}/{len(degradation_data)} significant"
    )
    tbl_degradation_primary = build_degradation_table(
        degradation_display_rows_primary, title=_title,
    )

    mo.vstack([
        mo.md(f"**{_title}**"),
        mo.ui.tabs(
            {
                "Interactive Table": mo.ui.table(
                    data=degradation_display_rows_primary,
                    selection=None, pagination=False,
                ),
                "Export Preview": mo.ui.plotly(tbl_degradation_primary),
            },
            lazy=True,
        ),
    ])
    return


@app.cell(hide_code=True)
def _(
    abl_cost_data,
    build_cost_breakdown_table,
    format_cost_breakdown_rows,
    format_cost_cell,
    mo,
):
    """Ablation cost breakdown table."""
    mo.stop(
        not abl_cost_data,
        mo.callout(mo.md("Ablation cost table requires ablation data."), kind="info"),
    )

    cost_display_rows_ablation = format_cost_breakdown_rows(
        abl_cost_data, include_experiment=False,
    )

    _title = "Cost Breakdown by Configuration [Ablation]"
    tbl_cost_ablation = build_cost_breakdown_table(
        cost_display_rows_ablation,
        title=_title,
    )

    # Build format_mapping from format_cost_cell for consistent formatting
    _fmt_columns = [
        "Balanced Accuracy", "Cost per Trial ($)", "Total Cost ($)", "Input Tokens",
        "Output Tokens", "Reasoning Tokens", "Response Tokens",
    ]

    mo.ui.tabs(
        {
            "Interactive Table": mo.ui.table(
                data=cost_display_rows_ablation,
                selection=None,
                label=_title,
                pagination=False,
                format_mapping={
                    col: lambda v, c=col: format_cost_cell(c, v)
                    for col in _fmt_columns
                },
            ),
            "Export Preview": mo.ui.plotly(tbl_cost_ablation),
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Experiment Status [Ablation]
    """)
    return


@app.cell(hide_code=True)
def _(abl_resolution_data, build_cell_resolution_chart, mo):
    """Figure 10b: Cell resolution — ablation experiment."""
    mo.stop(
        not abl_resolution_data,
        mo.callout(mo.md("Ablation cell resolution chart requires ablation data."), kind="info"),
    )
    fig_abl_cell_resolution = build_cell_resolution_chart(
        abl_resolution_data, experiment_label="Ablation",
    )
    mo.ui.plotly(fig_abl_cell_resolution)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cross-Dataset Comparison
    """)
    return


@app.cell(hide_code=True)
def _(
    ablation_delta_data,
    build_ablation_comparison_chart,
    config_significance_data,
    mo,
):
    """Figure 5: Primary vs ablation comparison (requires both datasets)."""
    mo.stop(
        not ablation_delta_data,
        mo.callout(
            mo.md("Ablation comparison chart requires both primary and ablation data. "
                   "Provide `--ablation-snapshot`."),
            kind="info",
        ),
    )
    fig_ablation = build_ablation_comparison_chart(
        ablation_delta_data,
        significance_data=config_significance_data if config_significance_data else None,
    )
    mo.ui.plotly(fig_ablation)
    return (fig_ablation,)


@app.cell(hide_code=True)
def _(build_condition_level_ablation_effect_chart, cond_significance_data, mo):
    """Condition-level ablation effect chart (requires both datasets)."""
    mo.stop(
        not cond_significance_data,
        mo.callout(
            mo.md("Condition-level ablation effect chart requires both primary and "
                   "ablation data. Provide `--ablation-snapshot`."),
            kind="info",
        ),
    )
    fig_ablation_condition = build_condition_level_ablation_effect_chart(
        cond_significance_data,
    )
    mo.ui.plotly(fig_ablation_condition)
    return (fig_ablation_condition,)


@app.cell(hide_code=True)
def _(
    FIGURES_SUBDIR,
    Path,
    abl_cond_acc_data,
    abl_cost_data,
    abl_effort_data,
    abl_example_data,
    abl_resolution_data,
    abl_token_scaling_data,
    build_paired_cell_resolution,
    build_paired_condition_accuracy,
    build_paired_cost_vs_balanced_accuracy,
    build_paired_example_condition_accuracy_heatmap,
    build_paired_reasoning_effort,
    build_paired_reasoning_token_scaling,
    build_paired_token_composition,
    cli_format,
    cli_output_dir,
    cond_acc_data,
    cost_data,
    effort_data,
    example_data,
    export_figure,
    heatmap_config_order,
    is_script_mode,
    resolution_data,
    token_scaling_data,
):
    """Script mode: export the paired composite figures (both experiments).

    Each composite lays both experiments out in one Plotly figure on the
    shared structural configuration order, so the export is the placed
    paired stem directly — there are no per-experiment panel exports and
    no PIL stitching step.
    """
    if is_script_mode and abl_cond_acc_data:
        _out = Path(cli_output_dir) / FIGURES_SUBDIR
        # Each builder declares its own canvas geometry, so the export inherits it.
        _figures = {
            "condition_accuracy_paired": build_paired_condition_accuracy(
                cond_acc_data, abl_cond_acc_data,
            ),
            "reasoning_effort_vs_balanced_accuracy_paired":
                build_paired_reasoning_effort(effort_data, abl_effort_data),
            "cost_per_trial_vs_balanced_accuracy_paired":
                build_paired_cost_vs_balanced_accuracy(
                    cost_data, abl_cost_data,
                ),
            "token_composition_paired": build_paired_token_composition(
                token_scaling_data, abl_token_scaling_data,
            ),
            "reasoning_token_scaling_paired":
                build_paired_reasoning_token_scaling(
                    token_scaling_data, abl_token_scaling_data,
                ),
            "cell_resolution_status_paired": build_paired_cell_resolution(
                resolution_data, abl_resolution_data,
            ),
        }
        for _condition in ("correct", "transparent", "opaque"):
            _figures[f"example_condition_accuracy_heatmap_{_condition}_paired"] = (
                build_paired_example_condition_accuracy_heatmap(
                    example_data, abl_example_data,
                    value_column=f"{_condition}_acc",
                    config_order=list(heatmap_config_order),
                )
            )
        print(f"\nExporting paired composite figures to {_out}/")
        for _name, _fig in _figures.items():
            _path = export_figure(_fig, _out, _name, fmt=cli_format)
            print(f"  {_path}")
        print(f"\nDone: {len(_figures)} paired composite figures exported.")
    return


@app.cell(hide_code=True)
def _(
    FIGURES_SUBDIR,
    Path,
    build_paired_failure_mode_correct,
    build_paired_failure_mode_incorrect,
    cli_format,
    cli_output_dir,
    export_figure,
    failure_data,
    failure_data_ablation,
    is_script_mode,
):
    """Script mode: export the paired failure mode breakdown composites."""
    if is_script_mode and failure_data and failure_data_ablation:
        _out = Path(cli_output_dir) / FIGURES_SUBDIR
        for _name, _fig in [
            (
                "failure_mode_breakdown_correct_draft_paired",
                build_paired_failure_mode_correct(
                    failure_data, failure_data_ablation,
                ),
            ),
            (
                "failure_mode_breakdown_incorrect_draft_paired",
                build_paired_failure_mode_incorrect(
                    failure_data, failure_data_ablation,
                ),
            ),
        ]:
            _path = export_figure(_fig, _out, _name, fmt=cli_format)
            print(f"  {_path}")
    return


@app.cell(hide_code=True)
def _(
    FIGURES_SUBDIR,
    Path,
    build_paired_failure_mode_dominance_heatmap,
    cli_format,
    cli_output_dir,
    export_figure,
    failure_mode_dominance_data,
    failure_mode_dominance_data_ablation,
    heatmap_config_order,
    is_script_mode,
):
    """Script mode: export the paired failure mode dominance composites."""
    if (
        is_script_mode
        and failure_mode_dominance_data
        and failure_mode_dominance_data_ablation
    ):
        _out = Path(cli_output_dir) / FIGURES_SUBDIR
        for _condition in ("correct", "transparent", "opaque"):
            _fig = build_paired_failure_mode_dominance_heatmap(
                failure_mode_dominance_data,
                failure_mode_dominance_data_ablation,
                value_column=f"{_condition}_dominance",
                config_order=list(heatmap_config_order),
            )
            _path = export_figure(
                _fig, _out,
                f"failure_mode_dominance_heatmap_{_condition}_paired",
                fmt=cli_format,
            )
            print(f"  {_path}")
    return


@app.cell(hide_code=True)
def _(
    FIGURES_SUBDIR,
    Path,
    ablation_delta_data,
    cli_format,
    cli_output_dir,
    export_figure,
    fig_ablation,
    is_script_mode,
):
    """Script mode: export ablation comparison figure (if ablation data available)."""
    if is_script_mode and ablation_delta_data:
        _path = export_figure(
            fig_ablation,
            Path(cli_output_dir) / FIGURES_SUBDIR,
            "ablation_effect_on_balanced_accuracy",
            fmt=cli_format,
        )
        print(f"  {_path}")
    return


@app.cell(hide_code=True)
def _(
    FIGURES_SUBDIR,
    Path,
    cli_format,
    cli_output_dir,
    cond_significance_data,
    export_figure,
    fig_ablation_condition,
    is_script_mode,
):
    """Script mode: export condition-level ablation effect figure (if ablation data available)."""
    if is_script_mode and cond_significance_data:
        _path = export_figure(
            fig_ablation_condition,
            Path(cli_output_dir) / FIGURES_SUBDIR,
            "ablation_effect_on_condition_accuracy",
            fmt=cli_format,
        )
        print(f"  {_path}")
    return


@app.cell(hide_code=True)
def _(
    build_cross_dataset_significance_table,
    cond_significance_data,
    format_cross_dataset_significance_rows,
    mo,
):
    """Table 6: Cross-dataset ablation significance tests (condition level)."""
    mo.stop(
        not cond_significance_data,
        mo.callout(
            mo.md("Cross-dataset significance table requires both primary and ablation data."),
            kind="info",
        ),
    )
    _n_sig = sum(1 for r in cond_significance_data if r.get("significant"))
    _n_total = len(cond_significance_data)
    _title = (
        f"Cross-Dataset Ablation Significance Tests (Condition Level) "
        f"[Primary vs. Ablation]: {_n_sig} of {_n_total} Significant"
    )

    cross_dataset_display_rows = format_cross_dataset_significance_rows(
        cond_significance_data,
    )

    tbl_cross_dataset = build_cross_dataset_significance_table(
        cross_dataset_display_rows,
        _title,
    )

    mo.vstack([
        mo.md(f"**{_title}**"),
        mo.ui.tabs(
            {
                "Interactive Table": mo.ui.table(
                    data=cross_dataset_display_rows,
                    selection=None,
                    pagination=True,
                    page_size=20,
                ),
                "Export Preview": mo.ui.plotly(tbl_cross_dataset),
            },
            lazy=True,
        ),
    ])
    return (cross_dataset_display_rows,)


@app.cell(hide_code=True)
def _(
    CROSS_DATASET_SIGNIFICANCE_COLUMNS,
    Path,
    TABLES_SUBDIR,
    cli_output_dir,
    cond_significance_data,
    cross_dataset_display_rows,
    export_table_markdown,
    is_script_mode,
):
    """Script mode: export cross-dataset significance table (if data available)."""
    if is_script_mode and cond_significance_data:
        _out = Path(cli_output_dir) / TABLES_SUBDIR
        _path = export_table_markdown(
            cross_dataset_display_rows, _out,
            "cross_dataset_ablation_significance_tests_paired",
            columns=CROSS_DATASET_SIGNIFICANCE_COLUMNS,
        )
        print(f"  {_path}")
    return


@app.cell(hide_code=True)
def _(
    build_cross_dataset_config_significance_table,
    config_significance_data,
    format_cross_dataset_config_significance_rows,
    mo,
):
    """Table 6b: Cross-dataset ablation significance tests (configuration level)."""
    mo.stop(
        not config_significance_data,
        mo.callout(
            mo.md(
                "Cross-dataset configuration-level significance table "
                "requires both primary and ablation data."
            ),
            kind="info",
        ),
    )
    _n_sig = sum(1 for r in config_significance_data if r.get("significant"))
    _n_total = len(config_significance_data)
    _title = (
        f"Cross-Dataset Ablation Significance Tests (Configuration Level) "
        f"[Primary vs. Ablation]: {_n_sig} of {_n_total} Significant"
    )

    cross_dataset_config_display_rows = format_cross_dataset_config_significance_rows(
        config_significance_data,
    )

    tbl_cross_dataset_config = build_cross_dataset_config_significance_table(
        cross_dataset_config_display_rows,
        _title,
    )

    mo.vstack([
        mo.md(f"**{_title}**"),
        mo.ui.tabs(
            {
                "Interactive Table": mo.ui.table(
                    data=cross_dataset_config_display_rows,
                    selection=None,
                    pagination=True,
                    page_size=21,
                ),
                "Export Preview": mo.ui.plotly(tbl_cross_dataset_config),
            },
            lazy=True,
        ),
    ])
    return (cross_dataset_config_display_rows,)


@app.cell(hide_code=True)
def _(
    CROSS_DATASET_CONFIG_SIGNIFICANCE_COLUMNS,
    Path,
    TABLES_SUBDIR,
    cli_output_dir,
    config_significance_data,
    cross_dataset_config_display_rows,
    export_table_markdown,
    is_script_mode,
):
    """Script mode: export cross-dataset configuration-level significance table (if data available)."""
    if is_script_mode and config_significance_data:
        _out = Path(cli_output_dir) / TABLES_SUBDIR
        _path = export_table_markdown(
            cross_dataset_config_display_rows, _out,
            "cross_dataset_ablation_configuration_level_significance_tests_paired",
            columns=CROSS_DATASET_CONFIG_SIGNIFICANCE_COLUMNS,
        )
        print(f"  {_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Combined Cross-Experiment Tables [Primary + Ablation]
    """)
    return


@app.cell(hide_code=True)
def _(
    build_balanced_accuracy_table,
    combined_leaderboard_data,
    format_leaderboard_rows,
    mo,
):
    """Combined leaderboard table (T00) — only when ablation data is loaded."""
    mo.stop(
        not combined_leaderboard_data,
        mo.callout(
            mo.md("Combined leaderboard requires both primary and ablation data."),
            kind="info",
        ),
    )
    leaderboard_display_rows_paired = format_leaderboard_rows(
        combined_leaderboard_data,
    )
    _title = "Balanced Accuracy Leaderboard [Primary + Ablation]"
    tbl_balanced_accuracy_paired = build_balanced_accuracy_table(
        combined_leaderboard_data,
        title=_title,
    )
    mo.ui.tabs(
        {
            "Interactive Table": mo.ui.table(
                data=leaderboard_display_rows_paired,
                selection=None,
                label=_title,
                pagination=False,
                freeze_columns_left=["Experiment"],
            ),
            "Export Preview": mo.ui.plotly(tbl_balanced_accuracy_paired),
        },
        lazy=True,
    )
    return (leaderboard_display_rows_paired,)


@app.cell(hide_code=True)
def _(
    BALANCED_ACCURACY_COLUMNS,
    Path,
    TABLES_SUBDIR,
    cli_output_dir,
    export_table_markdown,
    insert_experiment_cut_in_rows,
    is_script_mode,
    leaderboard_display_rows_paired,
):
    """Script mode: export combined leaderboard (paired)."""
    if is_script_mode:
        _out = Path(cli_output_dir) / TABLES_SUBDIR
        _path = export_table_markdown(
            insert_experiment_cut_in_rows(
                leaderboard_display_rows_paired,
                cut_in_column=BALANCED_ACCURACY_COLUMNS[0],
            ),
            _out,
            "balanced_accuracy_leaderboard_paired",
            columns=BALANCED_ACCURACY_COLUMNS,
        )
        print(f"  {_path}")
    return


@app.cell(hide_code=True)
def _(
    build_cost_breakdown_table,
    combined_cost_data,
    format_cost_breakdown_rows,
    format_cost_cell,
    mo,
):
    """Combined cost breakdown table (T04)."""
    mo.stop(
        not combined_cost_data,
        mo.callout(
            mo.md("Combined cost breakdown requires both primary and ablation data."),
            kind="info",
        ),
    )
    cost_display_rows_paired = format_cost_breakdown_rows(
        combined_cost_data, include_experiment=True,
    )
    _title = "Cost Breakdown by Configuration [Primary + Ablation]"
    tbl_cost_paired = build_cost_breakdown_table(
        cost_display_rows_paired,
        title=_title,
    )

    _fmt_columns = [
        "Balanced Accuracy", "Cost per Trial ($)", "Total Cost ($)", "Input Tokens",
        "Output Tokens", "Reasoning Tokens", "Response Tokens",
    ]

    mo.ui.tabs(
        {
            "Interactive Table": mo.ui.table(
                data=cost_display_rows_paired,
                selection=None,
                label=_title,
                pagination=False,
                freeze_columns_left=["Experiment", "Configuration"],
                format_mapping={
                    col: lambda v, c=col: format_cost_cell(c, v)
                    for col in _fmt_columns
                },
            ),
            "Export Preview": mo.ui.plotly(tbl_cost_paired),
        },
        lazy=True,
    )
    return (cost_display_rows_paired,)


@app.cell(hide_code=True)
def _(
    COST_TABLE_COLUMNS,
    Path,
    TABLES_SUBDIR,
    cli_output_dir,
    cost_display_rows_paired,
    export_table_markdown,
    format_cost_rows_for_markdown,
    insert_experiment_cut_in_rows,
    is_script_mode,
):
    """Script mode: export combined cost breakdown table (paired)."""
    if is_script_mode:
        _out = Path(cli_output_dir) / TABLES_SUBDIR
        _path = export_table_markdown(
            insert_experiment_cut_in_rows(
                format_cost_rows_for_markdown(cost_display_rows_paired),
                cut_in_column=COST_TABLE_COLUMNS[0],
            ),
            _out, "cost_breakdown_paired",
            columns=COST_TABLE_COLUMNS,
        )
        print(f"  {_path}")
    return


@app.cell(hide_code=True)
def _(
    FIGURES_SUBDIR,
    Path,
    abl_degradation_data,
    abl_marginal_data,
    build_paired_marginal_cost,
    cli_format,
    cli_output_dir,
    degradation_data,
    export_figure,
    is_script_mode,
    marginal_data,
):
    """Script mode: export the paired marginal cost composite."""
    if is_script_mode and marginal_data and abl_marginal_data:
        _path = export_figure(
            build_paired_marginal_cost(
                marginal_data, abl_marginal_data,
                degradation_data or None, abl_degradation_data or None,
            ),
            Path(cli_output_dir) / FIGURES_SUBDIR,
            "marginal_cost_per_percentage_point_paired",
            fmt=cli_format,
        )
        print(f"  {_path}")
    return


@app.cell(hide_code=True)
def _(
    ablation_results,
    build_cap_sensitivity_table,
    format_cap_sensitivity_rows,
    mo,
    pairwise_comparison_cap_sensitivity,
    results,
):
    """Display rows for the pairwise comparison betting cap sensitivity table.

    The shaper recomputes every pairwise condition comparison in both
    experiments once per cap in the grid, so the rows are built only when the
    ablation snapshot is loaded — the primary is already guaranteed by the
    loader cell's ``mo.stop``. This is the precondition the paired-table export
    reads, and an empty list keeps the export list from claiming a table it has
    no data for.
    """
    cap_sensitivity_display_rows = (
        format_cap_sensitivity_rows(
            pairwise_comparison_cap_sensitivity(results, ablation_results),
        )
        if ablation_results is not None
        else []
    )

    _cap_title = "Pairwise Comparison Betting Cap Sensitivity [Primary + Ablation]"
    tbl_cap_sensitivity = (
        build_cap_sensitivity_table(cap_sensitivity_display_rows, title=_cap_title)
        if cap_sensitivity_display_rows
        else None
    )

    mo.vstack([
        mo.md(f"**{_cap_title}**"),
        mo.ui.tabs(
            {
                "Interactive Table": mo.ui.table(
                    data=cap_sensitivity_display_rows,
                    selection=None, pagination=False,
                ),
                "Export Preview": mo.ui.plotly(tbl_cap_sensitivity),
            },
            lazy=True,
        ),
    ]) if tbl_cap_sensitivity is not None else mo.md("")
    return (cap_sensitivity_display_rows,)


@app.cell(hide_code=True)
def _(
    CAP_SENSITIVITY_COLUMNS,
    DEGRADATION_TABLE_COLUMNS,
    MODEL_SIZE_DETAIL_COLUMNS,
    MODEL_SIZE_SUMMARY_COLUMNS,
    Path,
    TABLES_SUBDIR,
    cap_sensitivity_display_rows,
    cli_output_dir,
    combined_degradation_data,
    combined_model_size_best_data,
    combined_model_size_detail_data,
    combined_model_size_worst_data,
    degradation_display_rows,
    export_table_markdown,
    insert_experiment_cut_in_rows,
    is_script_mode,
    model_size_best_display_rows,
    model_size_detail_display_rows,
    model_size_worst_display_rows,
):
    """Script mode: export the paired degradation, model size, and betting cap
    sensitivity tables."""
    if is_script_mode:
        _out = Path(cli_output_dir) / TABLES_SUBDIR
        # Each table is written only when its combined (primary + ablation)
        # source data is loaded, mirroring the precondition of the cell that
        # built the display rows. Each column constant leads with the
        # "Experiment" key, which the cut-in transform replaces with one
        # heading row per experiment block, so the export projects the
        # constant from its first data column onward.
        _exports = [
            (_rows, _name, _columns)
            for _available, _rows, _name, _columns in [
                (combined_degradation_data, degradation_display_rows,
                 "adjacent_pair_degradation_tests_paired",
                 DEGRADATION_TABLE_COLUMNS),
                (combined_model_size_worst_data, model_size_worst_display_rows,
                 "within_provider_model_size_comparison_every_level_tests_paired",
                 MODEL_SIZE_SUMMARY_COLUMNS),
                (combined_model_size_best_data, model_size_best_display_rows,
                 "within_provider_model_size_comparison_some_level_tests_paired",
                 MODEL_SIZE_SUMMARY_COLUMNS),
                (combined_model_size_detail_data, model_size_detail_display_rows,
                 "within_provider_model_size_comparison_per_level_detail_paired",
                 MODEL_SIZE_DETAIL_COLUMNS),
                (cap_sensitivity_display_rows, cap_sensitivity_display_rows,
                 "pairwise_comparison_betting_cap_sensitivity_paired",
                 CAP_SENSITIVITY_COLUMNS),
            ]
            if _available
        ]
        print(f"\nExporting paired tables to {_out}/")
        for _rows, _name, _columns in _exports:
            _cut_rows = insert_experiment_cut_in_rows(
                _rows, cut_in_column=_columns[1],
            )
            print(f"  {export_table_markdown(_cut_rows, _out, _name, columns=_columns[1:])}")
        print(f"Done: {len(_exports)} paired tables exported (markdown).")
    return


@app.cell(hide_code=True)
def _(
    ARTICULATION_GOVERNING_COLUMNS,
    Path,
    TABLES_SUBDIR,
    articulation_display_rows,
    cli_output_dir,
    export_table_markdown,
    is_script_mode,
):
    """Script mode: export per-condition articulation and governing rate table."""
    if is_script_mode and articulation_display_rows:
        _out = Path(cli_output_dir) / TABLES_SUBDIR
        _path = export_table_markdown(
            articulation_display_rows, _out,
            "per_condition_articulation_and_governing_rates_paired",
            columns=ARTICULATION_GOVERNING_COLUMNS,
        )
        print(f"  {_path}")
    return


@app.cell(hide_code=True)
def _(
    MISATTRIBUTION_RATES_COLUMNS,
    Path,
    TABLES_SUBDIR,
    cli_output_dir,
    export_table_markdown,
    is_script_mode,
    misattribution_display_rows,
):
    """Script mode: export per-condition misattribution flag rate table."""
    if is_script_mode and misattribution_display_rows:
        _out = Path(cli_output_dir) / TABLES_SUBDIR
        _path = export_table_markdown(
            misattribution_display_rows, _out,
            "per_condition_misattribution_rates_paired",
            columns=MISATTRIBUTION_RATES_COLUMNS,
        )
        print(f"  {_path}")
    return


@app.cell(hide_code=True)
def _():
    import argparse
    import re
    import sys
    from pathlib import Path
    from urllib.parse import urlencode

    from utils.experiment_analysis import (
        STIMULUS_METADATA_REGISTRY,
        usage_cost_summary,
    )

    from utils.experiment_analysis.figure_builders import (
        build_ablation_comparison_chart,
        build_cell_resolution_chart,
        build_condition_accuracy_chart,
        build_condition_level_ablation_effect_chart,
        build_cost_vs_balanced_accuracy_scatter,
        build_example_condition_accuracy_heatmap,
        build_failure_mode_chart_correct,
        build_failure_mode_chart_incorrect,
        build_failure_mode_dominance_heatmap,
        build_failure_mode_rate_chart_correct,
        build_failure_mode_rate_chart_incorrect,
        build_marginal_cost_chart,
        build_paired_cell_resolution,
        build_paired_condition_accuracy,
        build_paired_cost_vs_balanced_accuracy,
        build_paired_example_condition_accuracy_heatmap,
        build_paired_failure_mode_correct,
        build_paired_failure_mode_dominance_heatmap,
        build_paired_failure_mode_incorrect,
        build_paired_marginal_cost,
        build_paired_reasoning_effort,
        build_paired_reasoning_token_scaling,
        build_paired_token_composition,
        build_reasoning_effort_chart,
        build_reasoning_token_scaling_chart,
        build_token_composition_chart,
        heatmap_column_config,
        structural_config_order,
        export_figure,
        short_config_label,
    )
    from utils.experiment_analysis.table_builders import (
        ARTICULATION_GOVERNING_COLUMNS,
        CAP_SENSITIVITY_COLUMNS,
        COST_TABLE_COLUMNS,
        CROSS_DATASET_CONFIG_SIGNIFICANCE_COLUMNS,
        CROSS_DATASET_SIGNIFICANCE_COLUMNS,
        DEGRADATION_TABLE_COLUMNS,
        MISATTRIBUTION_RATES_COLUMNS,
        MODEL_SIZE_DETAIL_COLUMNS,
        MODEL_SIZE_SUMMARY_COLUMNS,
        BALANCED_ACCURACY_COLUMNS,
        build_articulation_governing_table,
        build_misattribution_rates_table,
        build_cap_sensitivity_table,
        build_cost_breakdown_table,
        build_cross_dataset_config_significance_table,
        build_cross_dataset_significance_table,
        build_degradation_table,
        build_model_size_detail_table,
        build_model_size_summary_table,
        build_balanced_accuracy_table,
        export_table_markdown,
        format_cap_sensitivity_rows,
        format_articulation_governing_rows,
        format_misattribution_rates_rows,
        format_cost_breakdown_rows,
        format_cost_cell,
        format_cost_rows_for_markdown,
        format_cross_dataset_config_significance_rows,
        format_cross_dataset_significance_rows,
        format_degradation_rows,
        format_leaderboard_rows,
        format_model_size_detail_rows,
        format_model_size_summary_rows,
        insert_experiment_cut_in_rows,
    )
    from utils.experiment_analysis.visualizations_data import (
        aggregate_failure_mode_rates,
        cell_resolution_by_config,
        combined_balanced_accuracy_leaderboard,
        combined_cost_vs_balanced_accuracy,
        combined_degradation_test_summary,
        combined_model_size_level_detail_rows,
        combined_model_size_summary_rows,
        condition_accuracy_by_config,
        cost_vs_balanced_accuracy,
        cross_dataset_significance_data,
        degradation_test_summary,
        failure_mode_rates_by_config,
        model_size_level_detail_rows,
        model_size_summary_rows,
        valid_trial_disclosure,
        example_accuracy_matrix,
        failure_mode_by_config,
        failure_mode_dominance_matrix,
        marginal_cost_per_balanced_accuracy_point,
        misattribution_rates_by_condition,
        pairwise_comparison_cap_sensitivity,
        balanced_accuracy_leaderboard,
        primary_vs_ablation_deltas,
        reasoning_effort_curves,
        reasoning_token_scaling,
    )
    from utils.experiment_analysis.pipeline import (
        ExperimentAnalysisResults,
        analyze_cross_dataset,
    )
    from utils.notebook_cli import parse_script_mode_args
    from utils.rationale_analysis.models import RationaleAnalysisResults

    ANALYSIS_OUTPUTS_DIR = (
        Path(__file__).parent.parent / "analysis_outputs"
    )
    return (
        ANALYSIS_OUTPUTS_DIR,
        ARTICULATION_GOVERNING_COLUMNS,
        BALANCED_ACCURACY_COLUMNS,
        CAP_SENSITIVITY_COLUMNS,
        COST_TABLE_COLUMNS,
        CROSS_DATASET_CONFIG_SIGNIFICANCE_COLUMNS,
        CROSS_DATASET_SIGNIFICANCE_COLUMNS,
        DEGRADATION_TABLE_COLUMNS,
        ExperimentAnalysisResults,
        MISATTRIBUTION_RATES_COLUMNS,
        MODEL_SIZE_DETAIL_COLUMNS,
        MODEL_SIZE_SUMMARY_COLUMNS,
        Path,
        RationaleAnalysisResults,
        STIMULUS_METADATA_REGISTRY,
        aggregate_failure_mode_rates,
        analyze_cross_dataset,
        argparse,
        balanced_accuracy_leaderboard,
        build_ablation_comparison_chart,
        build_articulation_governing_table,
        build_balanced_accuracy_table,
        build_cap_sensitivity_table,
        build_cell_resolution_chart,
        build_condition_accuracy_chart,
        build_condition_level_ablation_effect_chart,
        build_cost_breakdown_table,
        build_cost_vs_balanced_accuracy_scatter,
        build_cross_dataset_config_significance_table,
        build_cross_dataset_significance_table,
        build_degradation_table,
        build_example_condition_accuracy_heatmap,
        build_failure_mode_chart_correct,
        build_failure_mode_chart_incorrect,
        build_failure_mode_dominance_heatmap,
        build_failure_mode_rate_chart_correct,
        build_failure_mode_rate_chart_incorrect,
        build_marginal_cost_chart,
        build_misattribution_rates_table,
        build_model_size_detail_table,
        build_model_size_summary_table,
        build_paired_cell_resolution,
        build_paired_condition_accuracy,
        build_paired_cost_vs_balanced_accuracy,
        build_paired_example_condition_accuracy_heatmap,
        build_paired_failure_mode_correct,
        build_paired_failure_mode_dominance_heatmap,
        build_paired_failure_mode_incorrect,
        build_paired_marginal_cost,
        build_paired_reasoning_effort,
        build_paired_reasoning_token_scaling,
        build_paired_token_composition,
        build_reasoning_effort_chart,
        build_reasoning_token_scaling_chart,
        build_token_composition_chart,
        cell_resolution_by_config,
        combined_balanced_accuracy_leaderboard,
        combined_cost_vs_balanced_accuracy,
        combined_degradation_test_summary,
        combined_model_size_level_detail_rows,
        combined_model_size_summary_rows,
        condition_accuracy_by_config,
        cost_vs_balanced_accuracy,
        cross_dataset_significance_data,
        degradation_test_summary,
        example_accuracy_matrix,
        export_figure,
        export_table_markdown,
        failure_mode_by_config,
        failure_mode_dominance_matrix,
        failure_mode_rates_by_config,
        format_articulation_governing_rows,
        format_cap_sensitivity_rows,
        format_cost_breakdown_rows,
        format_cost_cell,
        format_cost_rows_for_markdown,
        format_cross_dataset_config_significance_rows,
        format_cross_dataset_significance_rows,
        format_degradation_rows,
        format_leaderboard_rows,
        format_misattribution_rates_rows,
        format_model_size_detail_rows,
        format_model_size_summary_rows,
        heatmap_column_config,
        insert_experiment_cut_in_rows,
        marginal_cost_per_balanced_accuracy_point,
        misattribution_rates_by_condition,
        model_size_level_detail_rows,
        model_size_summary_rows,
        pairwise_comparison_cap_sensitivity,
        parse_script_mode_args,
        primary_vs_ablation_deltas,
        re,
        reasoning_effort_curves,
        reasoning_token_scaling,
        structural_config_order,
        sys,
        urlencode,
        usage_cost_summary,
        valid_trial_disclosure,
    )


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    """Imports for the paired leaderboard and cross-dataset supplementary tables."""
    from utils.experiment_analysis.table_builders import (
        E_VALUE_DISTRIBUTION_COLUMNS,
        FAILURE_MODE_BY_CONFIG_COLUMNS,
        PAIRWISE_COMPARISON_E_VALUE_COLUMNS,
        format_e_value_distribution_rows,
        format_failure_mode_by_config_rows,
        format_pairwise_comparison_e_value_rows,
    )
    from utils.experiment_analysis.visualizations_data import (
        aggregate_failure_mode_rates_by_config,
        e_value_distribution_by_condition,
        pairwise_comparison_e_values_by_config,
    )

    return (
        E_VALUE_DISTRIBUTION_COLUMNS,
        FAILURE_MODE_BY_CONFIG_COLUMNS,
        PAIRWISE_COMPARISON_E_VALUE_COLUMNS,
        aggregate_failure_mode_rates_by_config,
        e_value_distribution_by_condition,
        format_e_value_distribution_rows,
        format_failure_mode_by_config_rows,
        format_pairwise_comparison_e_value_rows,
        pairwise_comparison_e_values_by_config,
    )


@app.cell(hide_code=True)
def _(
    ablation_results,
    format_pairwise_comparison_e_value_rows,
    mo,
    pairwise_comparison_e_values_by_config,
    results,
):
    """Pairwise condition comparison e-values (paired)."""
    mo.stop(
        results is None or ablation_results is None,
        mo.callout(
            mo.md(
                "Pairwise comparison e-value table requires both the primary and "
                "ablation experiment snapshots."
            ),
            kind="info",
        ),
    )
    pairwise_e_values_primary = pairwise_comparison_e_values_by_config(
        results,
    )
    pairwise_e_values_ablation = pairwise_comparison_e_values_by_config(
        ablation_results,
    )
    pairwise_comparison_display_rows = format_pairwise_comparison_e_value_rows(
        pairwise_e_values_primary, pairwise_e_values_ablation,
    )
    mo.vstack([
        mo.md("**Pairwise Condition Comparison E-Values**"),
        mo.ui.table(
            data=pairwise_comparison_display_rows, selection=None, pagination=False,
        ),
    ])
    return (pairwise_comparison_display_rows,)


@app.cell(hide_code=True)
def _(
    PAIRWISE_COMPARISON_E_VALUE_COLUMNS,
    Path,
    TABLES_SUBDIR,
    cli_output_dir,
    export_table_markdown,
    is_script_mode,
    pairwise_comparison_display_rows,
):
    """Script mode: export pairwise condition comparison e-value table (paired)."""
    if is_script_mode and pairwise_comparison_display_rows:
        _out = Path(cli_output_dir) / TABLES_SUBDIR
        _path = export_table_markdown(
            pairwise_comparison_display_rows, _out,
            "pairwise_condition_comparison_e_values_paired",
            columns=PAIRWISE_COMPARISON_E_VALUE_COLUMNS,
        )
        print(f"  {_path}")
    return


@app.cell(hide_code=True)
def _(
    ablation_results,
    e_value_distribution_by_condition,
    format_e_value_distribution_rows,
    mo,
    results,
):
    """E-value distribution and power diagnostics (paired)."""
    mo.stop(
        results is None or ablation_results is None,
        mo.callout(
            mo.md(
                "E-value distribution table requires both the primary and ablation "
                "experiment snapshots."
            ),
            kind="info",
        ),
    )
    e_value_distribution_primary = e_value_distribution_by_condition(results)
    e_value_distribution_ablation = e_value_distribution_by_condition(ablation_results)
    e_value_distribution_display_rows = format_e_value_distribution_rows(
        e_value_distribution_primary, e_value_distribution_ablation,
    )
    mo.vstack([
        mo.md("**E-Value Distribution and Power Diagnostics**"),
        mo.ui.table(
            data=e_value_distribution_display_rows, selection=None, pagination=False,
        ),
    ])
    return (e_value_distribution_display_rows,)


@app.cell(hide_code=True)
def _(
    E_VALUE_DISTRIBUTION_COLUMNS,
    Path,
    TABLES_SUBDIR,
    cli_output_dir,
    e_value_distribution_display_rows,
    export_table_markdown,
    insert_experiment_cut_in_rows,
    is_script_mode,
):
    """Script mode: export e-value distribution and power diagnostics table (paired)."""
    if is_script_mode and e_value_distribution_display_rows:
        _out = Path(cli_output_dir) / TABLES_SUBDIR
        _path = export_table_markdown(
            insert_experiment_cut_in_rows(
                e_value_distribution_display_rows,
                cut_in_column=E_VALUE_DISTRIBUTION_COLUMNS[0],
            ),
            _out,
            "e_value_distribution_and_power_diagnostics_paired",
            columns=[
                c for c in E_VALUE_DISTRIBUTION_COLUMNS if c != "Experiment"
            ],
        )
        print(f"  {_path}")
    return


@app.cell(hide_code=True)
def _(
    ablation_results,
    aggregate_failure_mode_rates_by_config,
    all_config_keys,
    format_failure_mode_by_config_rows,
    mo,
    ra_results,
    ra_results_ablation,
    results,
):
    """Per-configuration failure mode breakdown by condition (paired)."""
    mo.stop(
        ra_results is None
        or ra_results_ablation is None
        or results is None
        or ablation_results is None,
        mo.callout(
            mo.md(
                "Per-configuration failure mode table requires both experiment "
                "snapshots and both rationale analysis snapshots."
            ),
            kind="info",
        ),
    )
    failure_mode_by_config_primary = aggregate_failure_mode_rates_by_config(
        ra_results, all_config_keys,
    )
    failure_mode_by_config_ablation = aggregate_failure_mode_rates_by_config(
        ra_results_ablation, all_config_keys,
    )
    failure_mode_by_config_display_rows = format_failure_mode_by_config_rows(
        failure_mode_by_config_primary,
        failure_mode_by_config_ablation,
        results.config_summaries,
        ablation_results.config_summaries,
    )
    mo.vstack([
        mo.md("**Per-Configuration Failure Mode Breakdown by Condition**"),
        mo.ui.table(
            data=failure_mode_by_config_display_rows, selection=None, pagination=False,
        ),
    ])
    return (failure_mode_by_config_display_rows,)


@app.cell(hide_code=True)
def _(
    FAILURE_MODE_BY_CONFIG_COLUMNS,
    Path,
    TABLES_SUBDIR,
    cli_output_dir,
    export_table_markdown,
    failure_mode_by_config_display_rows,
    is_script_mode,
):
    """Script mode: export per-configuration failure mode breakdown table (paired)."""
    if is_script_mode and failure_mode_by_config_display_rows:
        _out = Path(cli_output_dir) / TABLES_SUBDIR
        _path = export_table_markdown(
            failure_mode_by_config_display_rows, _out,
            "per_configuration_failure_mode_breakdown_paired",
            columns=FAILURE_MODE_BY_CONFIG_COLUMNS,
        )
        print(f"  {_path}")
    return


@app.cell(hide_code=True)
def _():
    """Imports for the probe-auditor supplementary tables (paired)."""
    import hashlib

    from utils.experiment_analysis.table_builders import (
        PROBE_AUDITOR_ALIGNMENT_COLUMNS,
        PROBE_AUDITOR_DISAGREEMENT_TRANSITION_COLUMNS,
        format_probe_auditor_alignment_rows,
        format_probe_stub_layout_rows,
        resolve_floored_flags,
    )
    from utils.probe_audit_summary.filters import resolve_pool

    return (
        PROBE_AUDITOR_ALIGNMENT_COLUMNS,
        PROBE_AUDITOR_DISAGREEMENT_TRANSITION_COLUMNS,
        format_probe_auditor_alignment_rows,
        format_probe_stub_layout_rows,
        hashlib,
        resolve_floored_flags,
        resolve_pool,
    )


@app.cell(hide_code=True)
def _(
    Path,
    cli_probe_audit_dir,
    hashlib,
    ra_ablation_snapshot_resolved_path,
    ra_snapshot_resolved_path,
    resolve_floored_flags,
    resolve_pool,
):
    """Resolve the primary and ablation probe-audit pools for the alignment table.

    Each pool is selected by the SHA-256 of the rationale analysis snapshot the
    audit pinned (``probe_snapshot_sha256``) — the same RA snapshot that feeds
    the failure mode figures. Selecting by the probe snapshot rather than the
    shared skill fingerprint keeps the two experiments' pools from mixing.
    Floored flags are read from each pool's provenance so suppressed bounds
    match the rendered summary reports. The table is skipped when either RA
    snapshot or the audit directory is unavailable.
    """
    probe_auditor_experiment_pools = None
    _audit_dir = Path(cli_probe_audit_dir)
    if (
        ra_snapshot_resolved_path is not None
        and ra_ablation_snapshot_resolved_path is not None
        and _audit_dir.is_dir()
    ):
        def _resolve_experiment(label, ra_path):
            _sha = hashlib.sha256(Path(ra_path).read_bytes()).hexdigest()
            _, _pool_results, _ = resolve_pool(
                audit_dir=_audit_dir,
                fingerprint_prefix=None,
                args_digest_prefix=None,
                probe_snapshot_sha256=_sha,
                session_ids=None,
            )
            return (
                label,
                list(_pool_results),
                resolve_floored_flags(list(_pool_results)),
            )

        probe_auditor_experiment_pools = [
            _resolve_experiment("Primary", ra_snapshot_resolved_path),
            _resolve_experiment("Ablation", ra_ablation_snapshot_resolved_path),
        ]
    return (probe_auditor_experiment_pools,)


@app.cell(hide_code=True)
def _(format_probe_auditor_alignment_rows, mo, probe_auditor_experiment_pools):
    """Table: per-flag per-condition probe-auditor alignment (paired)."""
    mo.stop(
        probe_auditor_experiment_pools is None,
        mo.callout(
            mo.md(
                "The probe-auditor alignment table requires both rationale analysis "
                "snapshots and a probe-audit pool (`--probe-audit-dir`)."
            ),
            kind="info",
        ),
    )
    probe_auditor_alignment_display_rows = format_probe_auditor_alignment_rows(
        probe_auditor_experiment_pools,
    )
    mo.vstack([
        mo.md("**Probe-Auditor Alignment by Flag and Condition**"),
        mo.ui.table(
            data=probe_auditor_alignment_display_rows, selection=None, pagination=False,
        ),
    ])
    return (probe_auditor_alignment_display_rows,)


@app.cell(hide_code=True)
def _(
    PROBE_AUDITOR_ALIGNMENT_COLUMNS,
    PROBE_AUDITOR_DISAGREEMENT_TRANSITION_COLUMNS,
    Path,
    TABLES_SUBDIR,
    cli_output_dir,
    export_table_markdown,
    format_probe_stub_layout_rows,
    is_script_mode,
    probe_auditor_alignment_display_rows,
):
    """Script mode: export the two probe-auditor tables (paired).

    One row set feeds both: the alignment table carries the counts, rate, and
    bound; the transition table carries the four two-phase outcomes. The stub
    layout absorbs the Flag and Experiment key columns into heading rows, so
    each export projects only the Condition column and its own metrics.
    """
    if is_script_mode and probe_auditor_alignment_display_rows:
        _out = Path(cli_output_dir) / TABLES_SUBDIR
        _stub_rows = format_probe_stub_layout_rows(
            probe_auditor_alignment_display_rows,
        )
        for _stem, _columns in (
            ("probe_auditor_alignment_paired", PROBE_AUDITOR_ALIGNMENT_COLUMNS),
            (
                "probe_auditor_disagreement_transitions_paired",
                PROBE_AUDITOR_DISAGREEMENT_TRANSITION_COLUMNS,
            ),
        ):
            _path = export_table_markdown(
                _stub_rows, _out, _stem,
                columns=[
                    c for c in _columns if c not in ("Flag", "Experiment")
                ],
            )
            print(f"  {_path}")
    return


if __name__ == "__main__":
    app.run()
