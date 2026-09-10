"""Copy-panel anywidget: a word-wrapping, monospace text card with a copy button.

Pairs the soft-wrap rendering of the notebooks' <pre>-based stimulus cards with a
one-click copy affordance that mo.ui.code_editor lacks. Built for browsing an
assembled dataset and pasting examples into LLM prompt playgrounds.
"""

import pathlib

import anywidget
import traitlets


class CopyPanelWidget(anywidget.AnyWidget):
    """A bounded, scrollable, word-wrapping text panel with a copy-to-clipboard button.

    The ``text`` trait is the body content (and the exact string copied); ``label``
    is the header caption. Both are Python-to-JavaScript only — the widget reports
    nothing back. It renders its traits once at construction; wrap with
    mo.ui.anywidget(...) before displaying in a marimo notebook.
    """

    _esm = pathlib.Path(__file__).parent / "widget.js"
    _css = pathlib.Path(__file__).parent / "widget.css"

    text = traitlets.Unicode("").tag(sync=True)
    label = traitlets.Unicode("").tag(sync=True)


def copy_panel(label: str, text: str | None) -> CopyPanelWidget:
    """Build a copy panel from a header label and body text.

    A missing or empty body is coerced to the em-dash null sentinel (the repo-wide
    sentinel mandated by the root CLAUDE.md), so notebook call sites never have to
    special-case None.
    """
    return CopyPanelWidget(label=label, text=text if text else "—")
