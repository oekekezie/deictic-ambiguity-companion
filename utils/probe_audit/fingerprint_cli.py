"""Tiny CLI entry point for ``fingerprint.sh``.

Resolves the repo root via ``$PWD`` (the shell script ``cd``'s into it
before invoking us), loads ``fingerprint_manifest.json`` from the Skill
directory, and prints the outer fingerprint to stdout. The shell wrapper
captures stdout verbatim into the Skill body via ``!``…`` preprocessing.
"""

import json
import sys
from pathlib import Path

from utils.probe_audit.fingerprint import compute_fingerprint


def main() -> int:
    """Print the Skill fingerprint; exit nonzero on failure."""
    repo_root = Path.cwd()
    manifest_path = (
        repo_root / ".claude" / "skills" / "probe-audit" / "fingerprint_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    inputs = tuple(manifest["inputs"])
    result = compute_fingerprint(repo_root=repo_root, manifest_inputs=inputs)
    print(result.fingerprint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
