import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cell Inspector

    Drill-down view for a single heatmap cell (one configuration × one example × one condition).
    Three ways in: the **Open Cell Inspector** link in **results_figures.py** (when marimo
    serves the notebooks directory), the dropdowns below, or `marimo edit` / `marimo run`
    with this notebook's flags placed after `--`.
    The URL selects what is shown: `snapshot` and `ra_snapshot` take absolute paths, `config`
    takes a configuration key, `example` a base example id, and `condition` one of `correct`,
    `transparent`, or `opaque`. Adding `custom_id` (with `batch`, 1-based, when a snapshot
    carries more than one batch) opens straight to one trial's record. The same names are
    accepted on the command line as `--snapshot`, `--config`, `--example`, `--condition`,
    `--ra-snapshot`, `--custom-id`, and `--batch`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    """Inject shared stim-card CSS — identical to experiment_analysis.py and rationale_analysis.py."""
    stim_card_css = """
    pre {
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .stim-card {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        margin: 10px 0;
        overflow: hidden;
    }
    .stim-card-header {
        padding: 6px 14px;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.02em;
    }
    .stim-card-body {
        padding: 14px;
        margin: 0;
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-wrap: break-word;
        font-family: ui-monospace, "Cascadia Code", "Source Code Pro", monospace;
        font-size: 12.5px;
        line-height: 1.55;
        background: #f9fafb;
    }
    .stim-card-body-prose {
        padding: 14px;
        margin: 0;
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-wrap: break-word;
        font-size: 13px;
        line-height: 1.6;
        background: #f9fafb;
    }
    .stim-card-body-prose ul {
        list-style-type: disc;
        padding-left: 1.25em;
        margin: 0.25em 0;
        white-space: normal;
    }
    .stim-card-body-prose li {
        margin-bottom: 0.35em;
        line-height: 1.45;
    }
    .stim-card-subtitle {
        padding: 4px 14px 8px;
        font-size: 12px;
        color: #6b7280;
        font-style: italic;
        background: #f9fafb;
        border-bottom: 1px solid #e5e7eb;
    }
    /* Per-section header colours */
    .stim-meta-eval  { background: #dbeafe; color: #1e40af; }
    .stim-sys-prompt { background: #ede9fe; color: #5b21b6; }
    .stim-user       { background: #d1fae5; color: #065f46; }
    .stim-draft      { background: #fef3c7; color: #92400e; }
    .stim-feedback   { background: #fee2e2; color: #991b1b; }
    .stim-question   { background: #fff7ed; color: #9a3412; }
    .stim-thinking   { background: #ccfbf1; color: #115e59; }
    .stim-rationale  { background: #f0f9ff; color: #0c4a6e; }
    .stim-response   { background: #f1f5f9; color: #334155; }
    .stim-evidence-for    { background: #d1fae5; color: #065f46; }
    .stim-evidence-against { background: #fee2e2; color: #991b1b; }
    .stim-ra-analysis     { background: #dbeafe; color: #1e40af; }
    /* Flag-group left-border accents for RA drill-down */
    .stim-flag-group-1 {
        border-left: 3px solid #10b981;
        padding-left: 8px;
        margin: 8px 0;
    }
    .stim-flag-group-2 {
        border-left: 3px solid #6366f1;
        padding-left: 8px;
        margin: 8px 0;
    }
    .stim-flag-group-heading {
        font-weight: 600;
        font-size: 13.5px;
        margin: 4px 0 2px;
        letter-spacing: 0.01em;
    }
    """
    mo.md(f"<style>{stim_card_css}</style>")
    return (stim_card_css,)


@app.cell(hide_code=True)
def _(argparse, mo, parse_script_mode_args, sys):
    """Read query params and CLI args for initial state."""
    query_params = mo.query_params()

    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--snapshot", type=str, default=None)
    _parser.add_argument("--config", type=str, default=None)
    _parser.add_argument("--example", type=str, default=None)
    _parser.add_argument("--condition", type=str, default=None)
    _parser.add_argument("--ra-snapshot", type=str, default=None)
    _parser.add_argument("--custom-id", type=str, default=None)
    _parser.add_argument("--batch", type=int, default=None)
    _cli_args = parse_script_mode_args(
        _parser,
        (
            "--snapshot", "--config", "--example", "--condition",
            "--ra-snapshot", "--custom-id", "--batch",
        ),
        sys.argv[1:],
    )

    # Query params take precedence over CLI args
    initial_snapshot = query_params.get("snapshot") or _cli_args.snapshot
    initial_config = query_params.get("config") or _cli_args.config
    initial_example = query_params.get("example") or _cli_args.example
    initial_condition = query_params.get("condition") or _cli_args.condition
    initial_ra_snapshot = query_params.get("ra_snapshot") or _cli_args.ra_snapshot
    initial_custom_id = query_params.get("custom_id") or _cli_args.custom_id
    # 1-based, matching the trial table's "Batch" column. int() accepts both a
    # URL string and the already-typed CLI value, and rejects a malformed one
    # loudly rather than dropping the caller's trial selection in silence.
    _batch = query_params.get("batch") or _cli_args.batch
    initial_batch = None if _batch is None else int(_batch)
    return (
        initial_batch,
        initial_condition,
        initial_config,
        initial_custom_id,
        initial_example,
        initial_ra_snapshot,
        initial_snapshot,
    )


@app.cell(hide_code=True)
def _(ANALYSIS_OUTPUTS_DIR, initial_snapshot, mo):
    """File browser fallback when no snapshot provided via query params or CLI."""
    snapshot_browser = (
        None
        if initial_snapshot is not None
        else mo.ui.file_browser(
            initial_path=str(ANALYSIS_OUTPUTS_DIR) + "/",
            filetypes=[".json"],
            selection_mode="file",
            multiple=False,
            label="### Select experiment snapshot:",
        )
    )
    snapshot_browser if snapshot_browser is not None else mo.md("")
    return (snapshot_browser,)


@app.cell(hide_code=True)
def _(ANALYSIS_OUTPUTS_DIR, initial_ra_snapshot, mo):
    """File browser fallback for optional RA snapshot."""
    _ra_dir = ANALYSIS_OUTPUTS_DIR / "rationale_analysis"
    ra_snapshot_browser = (
        None
        if initial_ra_snapshot is not None
        else mo.ui.file_browser(
            initial_path=str(_ra_dir) + "/" if _ra_dir.exists() else str(ANALYSIS_OUTPUTS_DIR) + "/",
            filetypes=[".json"],
            selection_mode="file",
            multiple=False,
            label="### Select rationale analysis snapshot (optional):",
        )
    )
    ra_snapshot_browser if ra_snapshot_browser is not None else mo.md("")
    return (ra_snapshot_browser,)


@app.cell(hide_code=True)
def _(ExperimentAnalysisResults, Path, initial_snapshot, mo, snapshot_browser):
    """Load experiment analysis results from snapshot."""
    if initial_snapshot is not None:
        _path = Path(initial_snapshot)
    elif snapshot_browser and snapshot_browser.value:
        _path = Path(snapshot_browser.value[0].path)
    else:
        _path = None

    mo.stop(
        _path is None,
        mo.callout(mo.md("Select an experiment snapshot to begin."), kind="info"),
    )
    mo.stop(
        not _path.exists(),
        mo.callout(mo.md(f"**Error:** Snapshot not found: `{_path}`"), kind="danger"),
    )

    results = ExperimentAnalysisResults.model_validate_json(_path.read_text())
    mo.callout(
        mo.md(
            f"**Loaded:** {len(results.config_summaries)} configurations, "
            f"{len(results.trial_outcomes):,} trial outcomes"
        ),
        kind="success",
    )
    return (results,)


@app.cell(hide_code=True)
def _(
    Path,
    RationaleAnalysisResults,
    initial_ra_snapshot,
    ra_snapshot_browser,
):
    """Load rationale analysis results (optional — None if not provided)."""
    if initial_ra_snapshot is not None:
        _path = Path(initial_ra_snapshot)
        ra_results = (
            RationaleAnalysisResults.model_validate_json(_path.read_text())
            if _path.exists()
            else None
        )
    elif ra_snapshot_browser and ra_snapshot_browser.value:
        _path = Path(ra_snapshot_browser.value[0].path)
        ra_results = RationaleAnalysisResults.model_validate_json(_path.read_text())
    else:
        ra_results = None
    return (ra_results,)


@app.cell(hide_code=True)
def _(load_experiment, mo, results):
    """Load full response data from the manifest's ProcessedJob files on disk.

    Returns an empty dict (with a warning callout) when the manifest is absent
    or the job files cannot be found — all downstream lookups simply get None.
    """
    response_lookup = {}
    _load_warning = None

    if results.manifest is not None:
        try:
            _, response_lookup = load_experiment(results.manifest)
        except Exception as e:
            _load_warning = str(e)

    _output = (
        mo.callout(
            mo.md(
                f"**Note:** Could not load full response data from disk: {_load_warning}\n\n"
                "Reasoning traces and raw response text will not be available."
            ),
            kind="warn",
        )
        if _load_warning
        else mo.md("")
    )
    _output
    return (response_lookup,)


@app.cell(hide_code=True)
def _(initial_condition, initial_config, mo, results, short_config_label):
    """Config and condition dropdowns — pre-selected from URL params when available."""
    _VALID_CONDITIONS = ("correct", "transparent", "opaque")

    _configs = sorted({t.config_key for t in results.trial_outcomes})
    _config_options = {f"{short_config_label(ck)}  ({ck})": ck for ck in _configs}
    # mo.ui.dropdown value= expects a dict key (display label), not a dict value
    _initial_config_label = (
        f"{short_config_label(initial_config)}  ({initial_config})"
        if initial_config in _configs else None
    )
    config_dropdown = mo.ui.dropdown(
        options=_config_options,
        value=_initial_config_label,
        label="Configuration",
        searchable=True,
    )

    condition_dropdown = mo.ui.dropdown(
        options={"correct": "correct", "transparent": "transparent", "opaque": "opaque"},
        value=initial_condition if initial_condition in _VALID_CONDITIONS else None,
        label="Condition",
    )
    return condition_dropdown, config_dropdown


@app.cell(hide_code=True)
def _(config_dropdown, initial_example, mo, parse_example_id, results):
    """Example dropdown — options filtered by selected config, pre-selected from URL params.

    Options are bare base example ids, matching the heatmap tick labels this
    notebook is the click-through destination for.
    """
    # Empty dropdown when no config selected; downstream mo.stop handles the guard
    if config_dropdown.value is None:
        example_dropdown = mo.ui.dropdown(options=[], label="Example", searchable=True)
    else:
        _examples = sorted({
            parse_example_id(cm.cell_key.example_id)[0]
            for cm in results.cell_metrics_list
            if cm.cell_key.config_key == config_dropdown.value
        })
        example_dropdown = mo.ui.dropdown(
            options=_examples,
            value=initial_example if initial_example in _examples else None,
            label="Example",
            searchable=True,
        )
    return (example_dropdown,)


@app.cell(hide_code=True)
def _(condition_dropdown, config_dropdown, example_dropdown, mo):
    """Render navigation dropdowns in a horizontal row."""
    mo.hstack([config_dropdown, example_dropdown, condition_dropdown], gap=1)
    return


@app.cell(hide_code=True)
def _(condition_dropdown, example_dropdown, mo):
    """Resolve selected identifiers into a full example_id."""
    selected_example = example_dropdown.value
    selected_condition = condition_dropdown.value

    mo.stop(
        not selected_example or not selected_condition,
        mo.callout(mo.md("Select a **configuration**, **example**, and **condition** above."), kind="info"),
    )

    _CONDITION_TO_SUFFIX: dict[str, str] = {
        "correct": "correct",
        "transparent": "incorrect_transparent",
        "opaque": "incorrect_opaque",
    }
    selected_example_id = f"{selected_example}_{_CONDITION_TO_SUFFIX[selected_condition]}"
    return selected_condition, selected_example, selected_example_id


@app.cell(hide_code=True)
def _(
    CellKey,
    STIMULUS_METADATA_REGISTRY,
    config_dropdown,
    ground_truth_for_condition,
    mo,
    results,
    selected_condition,
    selected_example,
    selected_example_id,
):
    """Cell metrics callout — accuracy, CS bounds, e-value, status."""
    _cell_key = CellKey(example_id=selected_example_id, config_key=config_dropdown.value)
    _cell = next(
        (cm for cm in results.cell_metrics_list if cm.cell_key == _cell_key),
        None,
    )
    mo.stop(
        _cell is None,
        mo.callout(mo.md(f"**No metrics** for `{selected_example_id}` × `{config_dropdown.value}`."), kind="warn"),
    )

    _meta = STIMULUS_METADATA_REGISTRY.get(selected_example)
    _domain = _meta.domain if _meta else "?"
    _risk = _meta.risk if _meta else "?"
    _gt = ground_truth_for_condition(selected_condition)

    # Condition explanation text
    _condition_explanation = {
        "correct": "Grader flags valid output as error → meta-evaluator should disagree (ground truth: incorrect)",
        "transparent": "Grader correctly spots reversion to historical value → meta-evaluator should agree (ground truth: correct)",
        "opaque": "Grader correctly spots reversion to historical value, harder to detect → meta-evaluator should agree (ground truth: correct)",
    }[selected_condition]

    # Status badge
    _status_icon = {"rejected": "\u2705", "futile": "\u26a0\ufe0f", "active": "\u23f3"}.get(_cell.status, "?")

    mo.callout(
        mo.vstack([
            mo.md(
                f"### {selected_example_id} × {config_dropdown.value}\n\n"
                f"**Domain:** {_domain} · **Risk:** {_risk} · **Ground Truth:** `{_gt}`\n\n"
                f"Condition: **{selected_condition}** · "
                f"Status: {_status_icon} **{_cell.status}**\n\n"
                f"*{_condition_explanation}*"
            ),
            mo.md(
                f"**Accuracy:** {_cell.accuracy:.3f} "
                f"[{_cell.cs_lower:.3f}, {_cell.cs_upper:.3f}] · "
                f"**E-value:** {_cell.e_value:.3f} "
                f"(log: {_cell.log_e_value:.2f}) · "
                f"**Batches:** {_cell.batches_used} · "
                f"**Trials:** {_cell.valid_trials}/{_cell.total_trials} valid · "
                f"**Parse failure rate:** {_cell.parse_failure_rate:.1%}"
            ),
        ], gap=0.5),
        kind="neutral",
    )
    return


@app.cell(hide_code=True)
def _(html, json, mo):
    """Card renderer — identical to experiment_analysis.py's render_review_card."""
    def render_review_card(
        title: str,
        content: str,
        css_class: str,
        *,
        is_json: bool = False,
        subtitle: str = "",
    ) -> "mo.Html":
        """Return an mo.Html card for the given title/content."""
        _display = content
        if is_json:
            try:
                _display = json.dumps(json.loads(content), indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, ValueError):
                pass
        _subtitle_html = (
            f'<div class="stim-card-subtitle">{html.escape(subtitle)}</div>'
            if subtitle
            else ""
        )
        return mo.Html(
            '<div class="stim-card">'
            f'<div class="stim-card-header {css_class}">'
            f"{html.escape(title)}</div>"
            f"{_subtitle_html}"
            f'<pre class="stim-card-body">'
            f"{html.escape(_display)}</pre>"
            "</div>"
        )

    return (render_review_card,)


@app.cell(hide_code=True)
def _(html, mo):
    """RA card renderer — identical to rationale_analysis.py's render_ra_card."""
    def render_ra_card(
        title: str,
        content: str,
        css_class: str,
        *,
        subtitle: str = "",
        items: tuple[str, ...] | None = None,
    ) -> "mo.Html":
        """Return an mo.Html card with evidence list or prose body."""
        _subtitle_html = (
            f'<div class="stim-card-subtitle">{subtitle}</div>'
            if subtitle
            else ""
        )
        if items is not None:
            _li_tags = "".join(
                f"<li>{html.escape(item)}</li>" for item in items
            )
            _body = f'<div class="stim-card-body-prose"><ul>{_li_tags}</ul></div>'
        else:
            _body = f'<div class="stim-card-body-prose">{html.escape(content)}</div>'
        return mo.Html(
            '<div class="stim-card">'
            f'<div class="stim-card-header {css_class}">'
            f"{html.escape(title)}</div>"
            f"{_subtitle_html}"
            f"{_body}"
            "</div>"
        )

    return (render_ra_card,)


@app.cell(hide_code=True)
def _(
    Path,
    find_dataset_jsonl,
    load_stimuli,
    mo,
    parse_stimulus_sections,
    render_review_card,
    results,
    selected_example_id,
    stim_card_css,
):
    """Stimulus content cards — loads from assembled JSONL and renders each section."""
    mo.stop(
        results.manifest is None,
        mo.callout(mo.md("Snapshot has no manifest — cannot load stimulus content."), kind="warn"),
    )

    _jsonl = find_dataset_jsonl(
        Path(__file__).parent.parent / "synthetic_dataset",
        results.manifest.dataset_sha256,
    )
    _stimuli = load_stimuli(_jsonl)
    _stim = _stimuli.get(selected_example_id)
    mo.stop(
        _stim is None,
        mo.callout(mo.md(f"**Error:** Stimulus `{selected_example_id}` not found in dataset."), kind="danger"),
    )

    # Section display config — mirrors experiment_analysis.py
    _SECTION_CSS: dict[str, str] = {
        "Grader's Feedback": "stim-feedback",
        "System Prompt": "stim-sys-prompt",
        "User Prompt": "stim-user",
        "Initial Draft": "stim-draft",
    }
    _SECTION_DISPLAY: dict[str, str] = {
        "Grader's Feedback": "Grader's Feedback (JSON)",
        "System Prompt": "Assistant System Prompt",
        "User Prompt": "User Prompt",
        "Initial Draft": "Initial Draft (JSON)",
    }
    _SECTION_ORDER: dict[str, int] = {
        "Grader's Feedback": 0,
        "System Prompt": 1,
        "User Prompt": 2,
        "Initial Draft": 3,
    }

    _preamble, _sections = parse_stimulus_sections(_stim["user_content"])
    _cards = []

    # Meta-evaluator system prompt
    if _stim.get("system_content"):
        _cards.append(render_review_card(
            "Meta-evaluator System Prompt",
            _stim["system_content"],
            "stim-meta-eval",
        ))

    # Question preamble
    if _preamble:
        _cards.append(render_review_card("Question", _preamble, "stim-question"))

    # Stimulus sections in canonical order
    _sorted = sorted(_sections, key=lambda s: _SECTION_ORDER.get(s[0], 999))
    for _name, _body in _sorted:
        _css = _SECTION_CSS.get(_name, "stim-meta-eval")
        _display_name = _SECTION_DISPLAY.get(_name, _name)
        _is_json = "JSON" in _display_name
        _cards.append(render_review_card(_display_name, _body, _css, is_json=_is_json))

    # Inline CSS required: mo.accordion uses Shadow DOM, so global styles don't apply
    _cards.insert(0, mo.Html(f"<style>{stim_card_css}</style>"))
    mo.accordion({"Stimulus Content": mo.vstack(_cards)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(
    config_dropdown,
    initial_batch,
    initial_custom_id,
    mo,
    results,
    selected_example_id,
):
    """Trial outcomes table — filterable by selected config/example.

    A deep link may name one trial through the ``custom_id`` query parameter,
    in which case that row starts selected and its detail renders without a
    click.
    """
    _trials = sorted(
        [
            t for t in results.trial_outcomes
            if t.config_key == config_dropdown.value
            and t.example_id == selected_example_id
        ],
        key=lambda t: (t.batch_index, t.trial),
    )
    mo.stop(
        not _trials,
        mo.callout(mo.md("No trials found for this cell."), kind="warn"),
    )

    _rows = []
    for _t in _trials:
        _match_icon = (
            "\u2705" if _t.matches_ground_truth
            else "\u274c" if _t.matches_ground_truth is not None
            else "\u26a0\ufe0f"
        )
        _rows.append({
            "Trial": _t.trial,
            "Batch": _t.batch_index + 1,
            "Predicted": _t.predicted_score or "parse failure",
            "Ground Truth": _t.ground_truth_score,
            "Match": _match_icon,
            "Rationale": (_t.rationale or "")[:120] + ("..." if _t.rationale and len(_t.rationale) > 120 else ""),
        })

    # A custom_id repeats across a cell's batches (they are generated per
    # batch), so ``batch`` — 1-based, as the column displays it — narrows the
    # match when a snapshot carries more than one batch. A custom_id belonging
    # to another cell matches nothing, which is what lets a reader who arrived
    # through a deep link navigate elsewhere with the parameter still set.
    _requested = [
        _i for _i, _t in enumerate(_trials)
        if initial_custom_id is not None
        and _t.custom_id == initial_custom_id
        and (initial_batch is None or _t.batch_index == initial_batch - 1)
    ]
    trials_table = mo.ui.table(
        _rows,
        selection="single",
        initial_selection=_requested[:1] or None,
        label="### Trial Outcomes",
        page_size=len(_rows),
        show_column_summaries=False,
    )
    trials_table
    return (trials_table,)


@app.cell(hide_code=True)
def _(
    config_dropdown,
    html,
    mo,
    render_review_card,
    response_lookup,
    results,
    selected_example_id,
    stim_card_css,
    trials_table,
):
    """Trial detail — rationale, verdict, operational metadata, and full response data."""
    mo.stop(
        trials_table is None or not trials_table.value,
        mo.callout(mo.md("Select a trial from the table above to see details."), kind="info"),
    )

    _sel = trials_table.value[0]
    _trial_num = _sel["Trial"]
    _batch_idx = _sel["Batch"] - 1

    # Find the full TrialOutcome
    _trial = next(
        (
            t for t in results.trial_outcomes
            if t.config_key == config_dropdown.value
            and t.example_id == selected_example_id
            and t.trial == _trial_num
            and t.batch_index == _batch_idx
        ),
        None,
    )
    mo.stop(_trial is None, mo.callout(mo.md("Trial record not found."), kind="danger"))

    _parts: list[object] = []

    # Verdict summary
    _match_text = (
        "Correct" if _trial.matches_ground_truth
        else "Incorrect" if _trial.matches_ground_truth is not None
        else "Parse Error"
    )
    _parts.append(mo.callout(
        mo.md(
            f"**Trial {_trial.trial}** (Batch {_trial.batch_index + 1}) — **{_match_text}**\n\n"
            f"Predicted: `{_trial.predicted_score}` · "
            f"Ground Truth: `{_trial.ground_truth_score}` · "
            f"Thinking: {'yes' if _trial.has_reasoning_trace else 'no'} · "
            f"Tokens: {_trial.usage.output_tokens:,} output"
        ),
        kind="success" if _trial.matches_ground_truth else "danger",
    ))

    # Rationale card
    if _trial.rationale:
        _parts.append(render_review_card(
            "Rationale",
            _trial.rationale,
            "stim-rationale",
        ))

    # Parse error card
    if _trial.parse_error:
        _parts.append(render_review_card(
            "Parse Error",
            _trial.parse_error,
            "stim-feedback",
        ))

    # Full response data (available when ProcessedJob files are on disk)
    _lookup_key = (_trial.config_key, _trial.batch_index, _trial.custom_id)
    _processed = response_lookup.get(_lookup_key)

    if _processed is not None:
        # Inline CSS required: mo.accordion uses Shadow DOM, so global styles don't apply
        _INLINE_CSS = f"<style>{stim_card_css}</style>"

        # Reasoning trace (collapsible — these can be very long)
        if _processed.reasoning_text:
            _parts.append(mo.accordion(
                {"Reasoning Trace": mo.Html(
                    _INLINE_CSS
                    + f'<pre class="stim-card-body">{html.escape(_processed.reasoning_text)}</pre>'
                )}
            ))
        # Raw response text (collapsible)
        _parts.append(mo.accordion(
            {"Raw Response Text": mo.Html(
                _INLINE_CSS
                + f'<pre class="stim-card-body">{html.escape(_processed.raw_response_text)}</pre>'
            )}
        ))
    elif _trial.has_reasoning_trace:
        # Trace exists but ProcessedJob files unavailable
        _parts.append(mo.callout(
            mo.md(
                "This trial has a reasoning trace, but the ProcessedJob files "
                "are not available on disk. Run from a machine with the full "
                "experiment data to view traces."
            ),
            kind="neutral",
        ))

    mo.vstack(_parts)
    return


@app.cell(hide_code=True)
def _(
    RATIONALE_ANALYSIS_FLAG_KEYS,
    config_dropdown,
    html,
    mo,
    ra_results,
    render_ra_card,
    selected_example_id,
    stim_card_css,
    trials_table,
):
    """RA failure mode section — flag values, evidence, and analysis per repetition."""
    mo.stop(
        ra_results is None,
        mo.callout(mo.md("*No rationale analysis data available.*"), kind="info"),
    )

    # Find matching RA trial records for this cell
    _ra_trials = sorted(
        [
            r for r in ra_results.trial_records
            if r.config_key == config_dropdown.value
            and r.example_id == selected_example_id
        ],
        key=lambda r: (r.batch_index, r.trial),
    )
    mo.stop(
        not _ra_trials,
        mo.callout(
            mo.md(f"No RA records for `{selected_example_id}` × `{config_dropdown.value}`."),
            kind="info",
        ),
    )

    # Human-readable label per flag key \u2014 title-cased from the underscored name.
    _flag_labels = {
        _flag: _flag.replace("_", " ").title()
        for _flag in RATIONALE_ANALYSIS_FLAG_KEYS
    }
    # Left-border accent: the five misattribution flags share group 1, the two
    # deictic flags share group 2 (the only two accents the stim-card CSS defines).
    _flag_group_css = {
        _flag: (
            "stim-flag-group-2"
            if _flag in (
                "articulated_operational_interpretation",
                "operational_interpretation_governed_judgment",
            )
            else "stim-flag-group-1"
        )
        for _flag in RATIONALE_ANALYSIS_FLAG_KEYS
    }

    # RA summary table \u2014 flag value plus vote tally for every flag.
    _ra_rows = []
    for _r in _ra_trials:
        _row: dict[str, object] = {"Trial": _r.trial, "Batch": _r.batch_index + 1}
        for _flag in RATIONALE_ANALYSIS_FLAG_KEYS:
            _value = getattr(_r, _flag)
            _votes_for = getattr(_r, f"{_flag}_votes_for")
            _votes_against = getattr(_r, f"{_flag}_votes_against")
            _row[_flag_labels[_flag]] = (
                "\u2705" if _value else ("\u274c" if _value is not None else "?")
            )
            _row[f"{_flag_labels[_flag]} Votes"] = (
                f"{_votes_for}/{_votes_for + _votes_against}"
            )
        _ra_rows.append(_row)

    _parts: list[object] = [
        mo.md("### Rationale Analysis"),
        mo.ui.table(
            _ra_rows,
            selection=None,
            page_size=len(_ra_rows),
            show_column_summaries=False,
        ),
    ]

    # Detail for the currently selected trial (from trials_table)
    if trials_table is not None and trials_table.value:
        _sel = trials_table.value[0]
        _trial_num = _sel["Trial"]
        _batch_idx = _sel["Batch"] - 1

        _match_ra = next(
            (
                r for r in _ra_trials
                if r.trial == _trial_num and r.batch_index == _batch_idx
            ),
            None,
        )

        if _match_ra is not None:
            _DRILL_CARD_CSS = f"<style>{stim_card_css}</style>"

            _rep_tabs: dict[str, object] = {}
            for _i, _rep in enumerate(_match_ra.repetitions, 1):
                if _rep is None:
                    _rep_tabs[f"Repetition {_i}"] = mo.callout(
                        mo.md(f"**Repetition {_i}:** Parse failure"), kind="warn",
                    )
                    continue

                # Summary callout \u2014 classification verdict for every flag
                _summary_items = []
                for _flag_name in RATIONALE_ANALYSIS_FLAG_KEYS:
                    _flag_data = getattr(_rep, _flag_name)
                    _readable = _flag_labels[_flag_name]
                    _cls_str = (
                        "\u2705 True" if _flag_data.classification else "\u274c False"
                    )
                    _summary_items.append(f"**{_readable}** = {_cls_str}")

                _summary = mo.callout(
                    mo.md(f"### Repetition {_i}\n\n" + "\n\n".join(_summary_items)),
                    kind="neutral",
                )

                # Per-flag analysis cards with inline CSS
                _all_flag_html_parts: list[str] = []
                for _flag_name in RATIONALE_ANALYSIS_FLAG_KEYS:
                    _flag_data = getattr(_rep, _flag_name)
                    _readable = _flag_labels[_flag_name]
                    _group_css = _flag_group_css[_flag_name]
                    _section_cards: list[object] = []

                    _section_cards.append(render_ra_card(
                        "Analysis", _flag_data.analysis, "stim-ra-analysis",
                    ))

                    _all_flag_html_parts.append(
                        f'<div class="{_group_css}">'
                        f'<div class="stim-flag-group-heading">{_readable}</div>'
                        + "\n".join(card.text for card in _section_cards)
                        + "</div>"
                    )

                _flag_section = mo.Html(_DRILL_CARD_CSS + "\n".join(_all_flag_html_parts))

                # Reasoning trace accordion
                _tab_contents: list[object] = [_summary, _flag_section]
                if _rep.reasoning_text:
                    _tab_contents.append(mo.accordion(
                        {"Reasoning Trace": mo.Html(
                            _DRILL_CARD_CSS
                            + f'<pre class="stim-card-body">{html.escape(_rep.reasoning_text)}</pre>'
                        )},
                        lazy=True,
                    ))
                _rep_tabs[f"Repetition {_i}"] = mo.vstack(_tab_contents)

            if _rep_tabs:
                _parts.append(mo.ui.tabs(_rep_tabs, lazy=True))
        else:
            _parts.append(mo.callout(
                mo.md("No RA record for the selected trial."), kind="info",
            ))

    mo.vstack(_parts)
    return


@app.cell(hide_code=True)
def _():
    import argparse
    import html
    import json
    import sys
    from pathlib import Path

    from utils.experiment_analysis import (
        CellKey,
        ExperimentAnalysisResults,
        STIMULUS_METADATA_REGISTRY,
        find_dataset_jsonl,
        ground_truth_for_condition,
        load_experiment,
        load_stimuli,
        parse_example_id,
        parse_stimulus_sections,
    )
    from utils.experiment_analysis.figure_builders import short_config_label
    from utils.notebook_cli import parse_script_mode_args
    from utils.rationale_analysis.models import (
        RATIONALE_ANALYSIS_FLAG_KEYS,
        RationaleAnalysisResults,
    )

    ANALYSIS_OUTPUTS_DIR = Path(__file__).parent.parent / "analysis_outputs"
    return (
        ANALYSIS_OUTPUTS_DIR,
        CellKey,
        ExperimentAnalysisResults,
        Path,
        RATIONALE_ANALYSIS_FLAG_KEYS,
        RationaleAnalysisResults,
        STIMULUS_METADATA_REGISTRY,
        argparse,
        find_dataset_jsonl,
        ground_truth_for_condition,
        html,
        json,
        load_experiment,
        load_stimuli,
        parse_example_id,
        parse_script_mode_args,
        parse_stimulus_sections,
        short_config_label,
        sys,
    )


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
