import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset Viewer

    Browse an assembled dataset (a JSONL of evaluation examples). Pick an
    example row, then one-click-copy its **meta-evaluator system prompt**,
    **user message**, and **response JSON schema** into an LLM prompt
    playground to test the example manually before committing to a batch
    inference run.
    """)
    return


@app.cell(hide_code=True)
def _(SYNTHETIC_DATASET_DIR, mo):
    dataset_browser = mo.ui.file_browser(
        initial_path=SYNTHETIC_DATASET_DIR,
        filetypes=[".jsonl"],
        multiple=False,
        selection_mode="file",
        label="### Select an assembled dataset (.jsonl):",
    )
    dataset_browser
    return (dataset_browser,)


@app.cell(hide_code=True)
def _(Path, dataset_browser, mo, parse_jsonl_bytes):
    dataset = None
    mo.stop(
        not dataset_browser.value,
        mo.callout(mo.md("Select a **.jsonl** dataset file to begin."), kind="info"),
    )

    # Read and parse the selected JSONL; surface any ValueError as a red callout.
    _path = Path(dataset_browser.value[0].path)
    try:
        dataset = parse_jsonl_bytes(_path.read_bytes())
    except ValueError as e:
        mo.stop(True, mo.callout(mo.md(f"**Parse error:** {e}"), kind="danger"))

    mo.callout(
        mo.md(f"**{len(dataset.examples)}** example(s) loaded from `{_path.name}`."),
        kind="success",
    )
    return (dataset,)


@app.cell(hide_code=True)
def _(dataset, mo):
    # Identity-only columns, derived purely from the Example model so the
    # notebook renders any assembled JSONL with no cross-package coupling.
    _rows = [
        {
            "Custom ID": _ex.custom_id,
            "Messages": len(_ex.messages),
            "Response Schema Present": "Yes" if _ex.response_schema is not None else "No",
        }
        for _ex in dataset.examples
    ]
    # Page the table at 50 rows: a small dataset shows in full, and larger
    # ones stay scannable instead of forcing a long scroll through one page.
    _page_size = min(len(_rows), 50)
    example_table = mo.ui.table(
        _rows,
        selection="single",
        label="### Examples",
        page_size=_page_size,
        show_column_summaries=False,
    )
    example_table
    return (example_table,)


@app.cell(hide_code=True)
def _(dataset, example_table, mo):
    mo.stop(
        not example_table.value,
        mo.callout(mo.md("Select an example from the table above."), kind="info"),
    )

    # custom_id is unique (Dataset validator), so this lookup is exact; the
    # None guard is a defensive halt that cannot fire in practice.
    _selected_id = example_table.value[0]["Custom ID"]
    selected_example = next(
        (_ex for _ex in dataset.examples if _ex.custom_id == _selected_id),
        None,
    )
    mo.stop(
        selected_example is None,
        mo.callout(
            mo.md(f"**Error:** Example `{_selected_id}` not found."), kind="danger"
        ),
    )
    return (selected_example,)


@app.cell(hide_code=True)
def _(mo):
    # Defined in a cell that depends only on `mo`, so it never re-runs — the
    # switch keeps its setting as the researcher loads datasets and browses
    # examples. The panels cell below renders it and reads its value.
    embed_schema_switch = mo.ui.switch(
        value=False,
        label="Embed the response schema in the system prompt (for Fireworks reasoning models)",
    )
    return (embed_schema_switch,)


@app.cell(hide_code=True)
def _(
    copy_panel,
    embed_schema_in_system_prompt,
    embed_schema_switch,
    json,
    mo,
    resolve_schema,
    selected_example,
):
    _embed = embed_schema_switch.value

    # Meta-evaluator system prompt. In Combined mode the response schema is
    # folded in exactly as the Fireworks serializer does for reasoning models.
    if _embed and selected_example.response_schema is not None:
        _system_text = embed_schema_in_system_prompt(
            selected_example.system_prompt,
            resolve_schema(selected_example.response_schema),
        )
        _system_label = "Meta-evaluator System Prompt (response schema embedded)"
    else:
        _system_text = selected_example.system_prompt
        _system_label = "Meta-evaluator System Prompt"
    _system_panel = mo.ui.anywidget(copy_panel(_system_label, _system_text))

    # User message — assembled datasets carry a single user turn; join defensively.
    _user_text = "\n\n".join(_m.content for _m in selected_example.messages)
    _user_panel = mo.ui.anywidget(copy_panel("User Message", _user_text))

    _panels = [_system_panel, _user_panel]

    # Separate mode keeps the schema as its own standalone panel.
    if not _embed:
        if selected_example.response_schema is None:
            _schema_text = None
        else:
            _schema_text = json.dumps(
                resolve_schema(selected_example.response_schema),
                indent=2,
                ensure_ascii=False,
            )
        _panels.append(
            mo.ui.anywidget(copy_panel("Response JSON Schema", _schema_text))
        )

    # Guidance reflects the selected mode.
    if _embed:
        _guidance = (
            "Fireworks reasoning models (Kimi, GLM) disable their reasoning "
            "output when `response_format` is set. Copy the system prompt and "
            "the user message into the playground and leave `response_format` "
            "unset — the response schema (when present) is embedded in the "
            "system prompt above."
        )
    else:
        _guidance = (
            "Copy each section into your LLM playground to manually test this "
            "example before committing to a batch inference run."
        )

    # The switch and its mode-aware guidance share one callout, so the control
    # reads as a deliberate block rather than a stray toggle above the panels.
    _controls = mo.callout(
        mo.vstack([embed_schema_switch, mo.md(_guidance)], gap=0.5),
        kind="info",
    )
    mo.vstack([_controls, *_panels], gap=0.5)
    return


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    from utils.batch_inference import parse_jsonl_bytes, resolve_schema
    from utils.batch_inference.serializers import embed_schema_in_system_prompt
    from utils.marimo_widgets.copy_panel import copy_panel

    SYNTHETIC_DATASET_DIR = Path(__file__).parent.parent / "synthetic_dataset"
    return (
        Path,
        SYNTHETIC_DATASET_DIR,
        copy_panel,
        embed_schema_in_system_prompt,
        json,
        parse_jsonl_bytes,
        resolve_schema,
    )


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
