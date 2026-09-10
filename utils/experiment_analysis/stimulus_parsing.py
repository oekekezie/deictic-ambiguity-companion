"""Stimulus content parser for ReFT benchmark user prompts.

Splits a meta-evaluator's user_content into a preamble (the question
posed to the model) and a sequence of named markdown sections containing
the system prompt, user prompt, initial draft, and grader's feedback.
"""

import re

# Pre-compiled patterns for section headers and fenced code blocks
_HEADER_RE: re.Pattern[str] = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_FENCE_RE: re.Pattern[str] = re.compile(
    r"```(?:json)?\s*\n(.*?)```", re.DOTALL
)


def parse_stimulus_sections(
    user_content: str,
) -> tuple[str, list[tuple[str, str]]]:
    """Split user_content into preamble + (section_name, body) pairs.

    Splits on '## ' markdown headers, extracts code-fenced content
    from each section body.  Returns (preamble, sections) where
    preamble is the question text before the first header and
    sections are pairs in document order.
    """
    splits = _HEADER_RE.split(user_content)
    # splits alternates: [preamble, header1, body1, header2, ...]
    preamble = splits[0].strip().strip("-").strip()
    sections: list[tuple[str, str]] = []
    for i in range(1, len(splits), 2):
        name = splits[i].strip()
        body = splits[i + 1] if i + 1 < len(splits) else ""
        fence = _FENCE_RE.search(body)
        text = fence.group(1).strip() if fence else body.strip()
        sections.append((name, text))
    return preamble, sections
