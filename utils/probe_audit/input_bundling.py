"""Input bundling for the two-phase fetch protocol.

Each bundle is the full payload one sub-agent rep sees at one of the
two phases — rationale fetch, then probe-output fetch. Both bundles are
SHA-256-fingerprinted in the audit log so a reviewer can confirm the
exact bytes a rep was shown at judgment time.

Each builder reads every source file's bytes EXACTLY ONCE, hashes
those bytes, verifies the hash against the caller-provided expected
SHA-256, then parses the bundle data from those same bytes. The
read-once + verify + parse sequence collapses into a single critical
section so the bytes the auditor sees are byte-identical to the bytes
the SHA-256 check passed against — eliminating any TOCTOU window
between integrity verification and bundle construction.
"""

import hashlib
import json
from pathlib import Path

from utils.experiment_analysis.stimulus_loading import parse_stimuli_jsonl
from utils.probe_audit.models import ProbeOutputBundle, RationaleBundle
from utils.probe_audit.sampling import check_rationale_source_shape
from utils.probe_audit.scenario_context import build_scenario_context
from utils.rationale_analysis.models import RATIONALE_ANALYSIS_FLAG_KEYS


class SourceDriftError(ValueError):
    """Raised when source-file bytes' SHA-256 disagrees with the expected pin.

    Subclasses ``ValueError`` so existing CLI exception-handling that
    catches ``ValueError`` continues to surface drift cleanly. CLI
    layers that want to distinguish drift from generic shape errors
    can catch ``SourceDriftError`` directly.
    """


def _canonical(payload: dict) -> str:
    """Sorted-key separator-compact JSON — used for both stdout and hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _verify_sha256(
    actual_bytes: bytes, expected: str, label: str, path: Path
) -> None:
    """Hash ``actual_bytes`` and raise ``SourceDriftError`` on mismatch.

    The ``label`` distinguishes which source drifted in the error
    message; ``path`` surfaces the on-disk location so the operator
    can locate and revert the mutated file.
    """
    actual = hashlib.sha256(actual_bytes).hexdigest()
    if actual != expected:
        raise SourceDriftError(
            f"{label} at {path} has been mutated since sample time — "
            f"SHA-256 {actual[:12]}… no longer matches pinned "
            f"{expected[:12]}…"
        )


def build_rationale_bundle(
    trial_key: tuple[str, str, int, int],
    rationale_source_path: Path,
    stimulus_source_path: Path,
    *,
    expected_rationale_source_sha256: str,
    expected_stimulus_source_sha256: str,
) -> tuple[RationaleBundle, str]:
    """Build the rationale-phase bundle for one trial.

    Reads each source file's bytes once, verifies the SHA-256 against
    the caller-provided expected pin, then parses the bundle data
    from those same bytes. A drift between the SHA check and the
    bundle parse is impossible because no second read happens.

    The rationale source is a JSON file with a ``trial_outcomes`` list
    whose entries each carry ``config_key``, ``example_id``, ``batch_index``,
    ``trial``, ``rationale``, ``predicted_score``, and ``ground_truth_score``.
    The stimulus source is the assembled JSONL whose ``custom_id`` records
    carry the grader's full structured feedback and whose static metadata +
    ground truth description fill the remaining scenario context fields.

    Returns the bundle and the SHA-256 of its canonical JSON.

    Raises:
        SourceDriftError: if the on-disk rationale or stimulus bytes'
            SHA-256 differs from the corresponding ``expected_*`` kwarg.
        ValueError: if the rationale file has the probe-snapshot shape
            (``trial_records`` instead of ``trial_outcomes``) — the
            shape check is delegated to
            :func:`check_rationale_source_shape` so the sample-time
            validator and the rep-time bundler raise identical wording.
        KeyError: if ``trial_key`` is absent from the rationale source.
    """
    rationale_bytes = Path(rationale_source_path).read_bytes()
    _verify_sha256(
        rationale_bytes,
        expected_rationale_source_sha256,
        "rationale source",
        Path(rationale_source_path),
    )
    payload = json.loads(rationale_bytes.decode("utf-8"))
    check_rationale_source_shape(payload, source_label=str(rationale_source_path))
    lookup = {
        (
            o["config_key"],
            o["example_id"],
            int(o["batch_index"]),
            int(o["trial"]),
        ): o
        for o in payload["trial_outcomes"]
    }
    outcome = lookup.get(trial_key)
    if outcome is None:
        raise KeyError(f"Trial {trial_key!r} not found in rationale source")

    stimulus_bytes = Path(stimulus_source_path).read_bytes()
    _verify_sha256(
        stimulus_bytes,
        expected_stimulus_source_sha256,
        "stimulus source",
        Path(stimulus_source_path),
    )
    # Parse from the same bytes used for the SHA check above; calling
    # ``load_stimuli(path)`` here would re-read the file, opening a
    # TOCTOU window between integrity verification and parse.
    stimuli = parse_stimuli_jsonl(stimulus_bytes.decode("utf-8"))
    scenario, condition = build_scenario_context(trial_key[1], stimuli)

    bundle = RationaleBundle(
        trial_key=trial_key,
        condition=condition,
        rationale=outcome["rationale"],
        predicted_score=outcome.get("predicted_score"),
        ground_truth_score=outcome["ground_truth_score"],
        current_value=scenario.current_value,
        proposed_value=scenario.proposed_value,
        historical_value=scenario.historical_value,
        domain_noun=scenario.domain_noun,
        draft_fallback_value=scenario.draft_fallback_value,
        grader_feedback=scenario.grader_feedback,
        ground_truth_description=scenario.ground_truth_description,
    )
    serialized = _canonical(bundle.model_dump(mode="json"))
    return bundle, _sha256_hex(serialized)


def build_probe_output_bundle(
    trial_key: tuple[str, str, int, int],
    probe_snapshot_path: Path,
    *,
    expected_probe_snapshot_sha256: str,
) -> tuple[ProbeOutputBundle, str]:
    """Build the probe-output-phase bundle for one trial.

    Reads the probe snapshot's bytes once, verifies the SHA-256
    against the caller-provided expected pin, then parses the trial
    record from those same bytes. Returns ``(bundle, sha256)`` — the
    hash is used in the audit log so reviewers can verify the exact
    bytes the auditor saw.

    Raises:
        SourceDriftError: if the on-disk probe snapshot bytes'
            SHA-256 differs from ``expected_probe_snapshot_sha256``.
        KeyError: if ``trial_key`` is absent from the snapshot.
    """
    probe_bytes = Path(probe_snapshot_path).read_bytes()
    _verify_sha256(
        probe_bytes,
        expected_probe_snapshot_sha256,
        "probe snapshot",
        Path(probe_snapshot_path),
    )
    payload = json.loads(probe_bytes.decode("utf-8"))
    lookup = {
        (
            r["config_key"],
            r["example_id"],
            int(r["batch_index"]),
            int(r["trial"]),
        ): r
        for r in payload["trial_records"]
    }
    rec = lookup.get(trial_key)
    if rec is None:
        raise KeyError(f"Trial {trial_key!r} not found in probe snapshot")

    bundle_fields: dict = {"trial_key": trial_key}
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        bundle_fields[f"aggregated_{flag}"] = rec.get(flag)
        bundle_fields[f"votes_{flag}"] = (
            rec.get(f"{flag}_votes_for", 0),
            rec.get(f"{flag}_votes_against", 0),
        )
    bundle = ProbeOutputBundle(**bundle_fields)
    serialized = _canonical(bundle.model_dump(mode="json"))
    return bundle, _sha256_hex(serialized)


def canonical_json_of_bundle(bundle) -> str:
    """Public accessor for the canonical JSON shown on stdout."""
    return _canonical(bundle.model_dump(mode="json"))
