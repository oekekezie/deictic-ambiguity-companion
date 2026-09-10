"""Script-mode argument parsing shared by the marimo notebooks.

A notebook runs one of two ways and reads ``sys.argv`` in both. Under marimo it
never sees marimo's own command line: when marimo forwards arguments it rewrites
``sys.argv`` to the notebook's filename followed by the tokens after ``--``
(``sys.argv = self.argv`` in ``marimo/_runtime/runtime.py``), so what reaches
the notebook is either its own flags or nothing at all. Run as a script, it is
handed whatever the caller typed. ``parse_known_args`` is what lets one file
serve both, since it tolerates an argument list carrying nothing it defines.

That tolerance cuts both ways. A caller who misplaces a flag has it dropped on
the same rule, so the notebook falls back to interactive mode, where its work
sits behind a button nobody clicks, and the process exits zero having produced
nothing. A bare ``--`` is the usual way in, and the trap is that it is correct
in one mode only: marimo consumes it as the separator, while a script invocation
passes it straight through to argparse, which then reads every token after it as
positional.

One observation separates the two cases: a notebook's own option strings reach
its parser only in script mode, so finding one among the unplaced tokens means
the caller aimed at script mode and missed.
"""

import argparse

SEPARATOR = "--"
"""The end-of-options marker, after which argparse reads tokens as positional."""


def parse_script_mode_args(
    parser: argparse.ArgumentParser,
    option_strings: tuple[str, ...],
    argv: list[str],
) -> argparse.Namespace:
    """Parse ``argv`` with ``parser``, refusing an invocation that did not land.

    ``option_strings`` are the options ``parser`` defines. Any of them left
    unparsed, or a bare ``--``, means the command line was aimed at this
    notebook and did not reach it, so the caller hears about it instead of
    watching a silent no-op.

    Raises ValueError naming the stranded tokens.
    """
    parsed, ignored = parser.parse_known_args(argv)
    stranded = [
        token for token in ignored
        if token == SEPARATOR or token in option_strings
    ]
    if stranded:
        raise ValueError(
            "These arguments were aimed at this notebook and did not reach it: "
            f"{' '.join(stranded)}. A bare '{SEPARATOR}' makes argparse read "
            "everything after it as positional, so the options are discarded, "
            "the notebook runs interactively, and nothing is saved. Pass the "
            f"options directly, with no '{SEPARATOR}' before them."
        )
    return parsed
