"""Argparse-based CLI for ``probe-audit``.

Every subcommand reads and extends an append-only JSONL log under
``<audit-dir>/audit_log.jsonl``. The log enforces the two-phase protocol:
``record-independent-reading`` must precede ``fetch-probe-output`` for a
given ``(trial_id, rep_index)``, and ``record-judgment`` must follow
``fetch-probe-output`` for the same pair. Fingerprint drift between the
sample-time fingerprint and any later invocation is a hard refusal.
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError

from utils.probe_audit.aggregation import AggregateError, aggregate_probe_audit_records
from utils.probe_audit.args import SampleArgs, args_digest as _args_digest, canonical_json
from utils.probe_audit.fingerprint import (
    FingerprintResult,
    compute_fingerprint,
    git_describe_label,
)
from utils.probe_audit.input_bundling import (
    build_probe_output_bundle,
    build_rationale_bundle,
    canonical_json_of_bundle,
)
from utils.probe_audit.log import (
    EventRefused,
    LogValidationError,
    append_event,
    read_log,
    validate_log,
)
from utils.probe_audit.models import ProbeAuditResults
from utils.probe_audit.sampling import (
    SampleManifest,
    SelectionSource,
    SkillForensics,
    draw_sample,
    prepare_sample_sources,
)
from utils.probe_audit.storage import (
    ProbeAuditIndexRecord,
    compute_trial_set_sha,
    load_audit_record,
    persist_audit_draw,
    prior_audit_trial_keys,
    prune_missing_audits,
    resolve_base_dir,
    transition_record,
    update_audit_status,
)
from utils.rationale_analysis.models import RATIONALE_ANALYSIS_FLAG_KEYS


def _resolve_session_dir(audit_dir: Path, base: Path) -> str | None:
    """Return ``audit_dir`` relative to ``base``, or ``None`` if outside.

    ``None`` signals the caller to skip the index write with a stderr
    warning. Hard-failing would break researchers who legitimately want
    out-of-tree audit directories; silent skipping would hide the reason
    downstream discovery never sees the run.
    """
    try:
        return str(audit_dir.resolve().relative_to(base.resolve()))
    except ValueError:
        return None


def _now_utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_trial_id(trial_id: str) -> tuple[str, str, int, int]:
    """Parse ``config_key|example_id|batch_index|trial`` into a tuple."""
    parts = trial_id.split("|")
    if len(parts) != 4:
        raise ValueError(
            f"trial_id must be 'config_key|example_id|batch_index|trial', got {trial_id!r}"
        )
    return parts[0], parts[1], int(parts[2]), int(parts[3])


def _load_manifest(audit_dir: Path) -> SampleManifest:
    path = audit_dir / "sample_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No sample_manifest.json in {audit_dir}")
    return SampleManifest.model_validate_json(path.read_text())


def _log_path(audit_dir: Path) -> Path:
    return audit_dir / "audit_log.jsonl"


def _check_fingerprint(
    audit_dir: Path, caller_fingerprint: str | None
) -> SampleManifest | None:
    """Return the manifest when the fingerprint matches; ``None`` on drift.

    Returning ``None`` rather than raising ``SystemExit`` preserves the
    int return-code contract each subcommand promises — callers simply
    ``if manifest is None: return 2`` after the error has already been
    printed to stderr. This also means programmatic uses of ``main()``
    don't have to catch ``SystemExit`` for drift (argparse's own
    ``SystemExit`` path is a separate concern).
    """
    manifest = _load_manifest(audit_dir)
    if caller_fingerprint is not None and caller_fingerprint != manifest.skill_fingerprint:
        print(
            f"error: fingerprint drift — sample was drawn under "
            f"{manifest.skill_fingerprint[:12]}… but caller passed "
            f"{caller_fingerprint[:12]}…",
            file=sys.stderr,
        )
        return None
    return manifest


def _parse_bool(s: str) -> bool:
    low = s.lower()
    if low in {"true", "1", "yes", "t"}:
        return True
    if low in {"false", "0", "no", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {s!r}")


def _validate_log_or_print_error(log_path: Path) -> bool:
    """Run :func:`validate_log`; on failure, print stderr and return False.

    Commands that surface control-flow or status output (``status``,
    ``next-rep``, ``report``) run this at the top so they cannot
    display or advise on a tampered log. Returning a bool lets each
    caller map the result to its own exit semantics.
    """
    try:
        validate_log(log_path)
        return True
    except LogValidationError as exc:
        print(f"error: audit log validation failed: {exc}", file=sys.stderr)
        return False


def _resolve_repo_root(explicit: str | None) -> Path:
    """Resolve the repository root path.

    ``explicit`` is the optional ``--repo-root`` flag. When present it
    wins, so tests can point at a tmp directory without a Skill.
    Otherwise we invoke ``git rev-parse --show-toplevel`` from the
    current working directory — this is more reliable than
    ``Path.cwd()`` alone because the caller may be running from a
    subdirectory, and (per H7 from the adversarial review) using raw
    cwd is bypassable by invoking from outside the repo.
    """
    if explicit is not None:
        return Path(explicit)
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(completed.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path.cwd()


def _git_rev_parse_head(repo_root: Path) -> str | None:
    """Return the current HEAD commit SHA, or ``None`` outside a Git repo."""
    if not (repo_root / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return completed.stdout.strip() or None


def _git_tree_dirty_for(repo_root: Path, paths: tuple[str, ...]) -> bool:
    """True iff any of ``paths`` diverges from HEAD (uncommitted change).

    Uses ``git diff-index --quiet HEAD -- <paths>`` which exits 0 when
    clean and 1 when dirty. Anything else (e.g., no commits yet, not a
    Git repo) → treat as dirty so the flag is conservatively pessimistic.
    """
    if not (repo_root / ".git").exists() or not paths:
        return False
    try:
        completed = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--", *paths],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
    except OSError:
        return True
    return completed.returncode != 0


def _collect_skill_forensics(repo_root: Path) -> tuple[SkillForensics, str | None]:
    """Compute Skill forensics for the current working tree.

    Reads the Skill's ``fingerprint_manifest.json`` to discover its input
    set, then runs :func:`compute_fingerprint` and captures the Git
    describe label + HEAD SHA + dirty-tree flag. Returns the forensics
    bundle plus the computed fingerprint hex string so ``_cmd_sample``
    can verify it against the ``--skill-fingerprint`` the caller passed.
    Returns ``(SkillForensics(), None)`` when the manifest file is
    missing — that happens in tests where the fingerprint is injected by
    fixture rather than recomputed on disk.
    """
    manifest_path = (
        repo_root
        / ".claude"
        / "skills"
        / "probe-audit"
        / "fingerprint_manifest.json"
    )
    if not manifest_path.exists():
        return SkillForensics(), None
    manifest_data = json.loads(manifest_path.read_text())
    inputs = tuple(manifest_data["inputs"])
    result: FingerprintResult = compute_fingerprint(
        repo_root=repo_root, manifest_inputs=inputs
    )
    forensics = SkillForensics(
        fingerprint_method=result.method,
        fingerprint_inputs=result.resolved_inputs,
        per_file_hashes=dict(result.per_file_hashes),
        human_label=git_describe_label(repo_root),
        git_commit_sha=_git_rev_parse_head(repo_root),
        git_tree_dirty=_git_tree_dirty_for(repo_root, result.resolved_inputs),
    )
    return forensics, result.fingerprint


# ── subcommand: sample ───────────────────────────────────────────────────


def _parse_trial_keys_csv(csv: str) -> tuple[tuple[str, str, int, int], ...]:
    """Parse ``cfg|ex|bi|t,cfg|ex|bi|t,...`` into a tuple of trial keys.

    Raises ``ValueError`` naming the offending token if any piece is
    malformed. The error is caught by ``_resolve_pinning_flags`` and
    surfaced as an exit-2 with the offending token in the message.
    """
    result: list[tuple[str, str, int, int]] = []
    for token in csv.split(","):
        parts = token.split("|")
        if len(parts) != 4:
            raise ValueError(
                f"malformed --trial-keys token {token!r}: expected 4 "
                f"pipe-separated fields (config_key|example_id|batch_index|trial), "
                f"got {len(parts)}"
            )
        try:
            batch_index = int(parts[2])
            trial = int(parts[3])
        except ValueError:
            raise ValueError(
                f"malformed --trial-keys token {token!r}: batch_index and trial "
                "must be integers"
            ) from None
        result.append((parts[0], parts[1], batch_index, trial))
    return tuple(result)


def _resolve_pinning_flags(
    args: argparse.Namespace,
    repo_root: Path,
) -> (
    tuple[tuple[tuple[str, str, int, int], ...], SelectionSource | None] | int
):
    """Resolve ``--trials-from`` / ``--trial-keys`` into pinned keys + source.

    Returns either ``(pinned_trial_keys, selection_source)`` on success or
    an integer exit code (2) on any malformed input. All error messages
    are printed to stderr inside this function so the caller just returns
    the int. Exit conditions: missing source file, non-``SampleManifest``
    JSON, source outside ``repo_root``, malformed ``--trial-keys`` token.
    """
    if args.trials_from is None and args.trial_keys is None:
        return ((), None)
    if args.trials_from is not None:
        source_path = Path(args.trials_from)
        if not source_path.exists():
            print(
                f"error: --trials-from source not found: {source_path}",
                file=sys.stderr,
            )
            return 2
        try:
            raw = source_path.read_bytes()
        except OSError as exc:
            print(
                f"error: cannot read --trials-from source {source_path}: {exc}",
                file=sys.stderr,
            )
            return 2
        try:
            source_manifest = SampleManifest.model_validate_json(raw)
        except ValidationError as exc:
            print(
                f"error: --trials-from source {source_path} is not a valid "
                f"SampleManifest: {exc}",
                file=sys.stderr,
            )
            return 2
        sha256 = hashlib.sha256(raw).hexdigest()
        try:
            rel = source_path.resolve().relative_to(repo_root.resolve())
        except ValueError:
            print(
                f"error: --trials-from source {source_path} is outside "
                f"--repo-root {repo_root}; cannot record a stable "
                "repo-relative path for provenance",
                file=sys.stderr,
            )
            return 2
        pinned = tuple(tuple(k) for k in source_manifest.trial_keys)
        return (pinned, SelectionSource(path=str(rel), sha256=sha256))
    # --trial-keys path
    try:
        pinned = _parse_trial_keys_csv(args.trial_keys)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return (pinned, None)


def _cmd_sample(args: argparse.Namespace) -> int:
    audit_dir = Path(args.output_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = audit_dir / "sample_manifest.json"

    # Repo root is needed both to relativize ``--trials-from`` and (later)
    # for the skill-forensics sanity check, so resolve it once up front.
    repo_root = _resolve_repo_root(args.repo_root)

    resolved = _resolve_pinning_flags(args, repo_root)
    if isinstance(resolved, int):
        return resolved
    pinned_trial_keys, selection_source = resolved

    # Sample-size resolution: caller passes it, or we derive from pinned
    # count. Mismatch between the two is a CLI-layer error so the message
    # names both integers plus the source (vs. a generic Pydantic error).
    if args.sample_size is None:
        if not pinned_trial_keys:
            print(
                "error: --sample-size is required unless --trials-from or "
                "--trial-keys is provided",
                file=sys.stderr,
            )
            return 2
        effective_sample_size = len(pinned_trial_keys)
    else:
        if pinned_trial_keys and args.sample_size != len(pinned_trial_keys):
            src = (
                f"--trials-from ({args.trials_from})"
                if args.trials_from is not None
                else "--trial-keys"
            )
            print(
                f"error: --sample-size ({args.sample_size}) does not match "
                f"pinned trial count ({len(pinned_trial_keys)}) derived from "
                f"{src}",
                file=sys.stderr,
            )
            return 2
        effective_sample_size = args.sample_size

    # Normalize the rationale source to repo-relative. This repo-relative
    # string is what feeds both (a) the manifest's ``rationale_source.path``
    # field — the sole manifest authority for the stage-N snapshot path —
    # and (b) the sampling seed via ``seed_from_inputs``. The absolute
    # ``Path`` is used only for reading bytes; it never enters sample
    # identity. An out-of-tree rationale source is rejected with the
    # same exit-2 idiom the CLI uses elsewhere.
    rationale_source_cli = Path(args.rationale_source)
    try:
        rationale_source_relative = str(
            rationale_source_cli.resolve().relative_to(repo_root.resolve())
        )
    except ValueError:
        print(
            f"error: --rationale-source {rationale_source_cli} is outside "
            f"--repo-root {repo_root}; cannot record a stable repo-relative "
            "path for provenance",
            file=sys.stderr,
        )
        return 2
    # Anchor the byte-read path explicitly to ``repo_root / canonical-
    # relative`` rather than threading the raw CLI form through to
    # ``draw_sample``. Both the identity input
    # (``rationale_source_repo_relative``) and the I/O input
    # (``rationale_source_path``) are now structurally derived from the
    # same canonical form, so byte-read and seed identity point at the
    # same file by construction — not incidentally, via CWD-consistent
    # ``.resolve()`` / ``.read_bytes()`` side-effects.
    rationale_source_path = (repo_root / rationale_source_relative).resolve()

    # Stimulus JSONL mirrors the rationale-source path-normalization
    # contract: the repo-relative form pins identity on the manifest's
    # ``stimulus_source.path``, the absolute path is anchored via
    # ``(repo_root / relative).resolve()`` for byte I/O. Unlike rationale
    # source, the stimulus does NOT enter the sampling seed (Decision 2
    # in the plan): it augments scenario context on the auditor's bundle
    # but does not define which trials are eligible for sampling.
    stimulus_source_cli = Path(args.stimulus_jsonl)
    try:
        stimulus_source_relative = str(
            stimulus_source_cli.resolve().relative_to(repo_root.resolve())
        )
    except ValueError:
        print(
            f"error: --stimulus-jsonl {stimulus_source_cli} is outside "
            f"--repo-root {repo_root}; cannot record a stable repo-relative "
            "path for provenance",
            file=sys.stderr,
        )
        return 2
    stimulus_source_path = (repo_root / stimulus_source_relative).resolve()

    # Probe snapshot mirrors the rationale-source contract: the repo-
    # relative string is the identity form recorded on the manifest
    # (``sample_args.probe_snapshot_path``) and fed into ``canonical_json``
    # → ``args_digest`` → sampling seed; the absolute path flows
    # separately to ``draw_sample`` for byte-I/O. Without this
    # normalization, two invocations pointing at the same conceptual
    # snapshot via different spellings (absolute vs. relative, or from
    # different CWDs) would drift ``args_digest`` and the seed, breaking
    # reproducibility and falsely tripping the re-entry drift check below.
    probe_snapshot_cli = Path(args.probe_snapshot)
    try:
        probe_snapshot_relative = str(
            probe_snapshot_cli.resolve().relative_to(repo_root.resolve())
        )
    except ValueError:
        print(
            f"error: --probe-snapshot {probe_snapshot_cli} is outside "
            f"--repo-root {repo_root}; cannot record a stable repo-relative "
            "path for provenance",
            file=sys.stderr,
        )
        return 2
    probe_snapshot_path = (repo_root / probe_snapshot_relative).resolve()

    # Single-read snapshot of every source file. ``prepare_sample_sources``
    # reads each file's bytes exactly once and derives BOTH the SHA-256
    # and the parsed payload from those exact bytes; the resulting
    # ``SampleSourcePayloads`` is threaded into (a) the manifest-drift
    # refusal check below, (b) the prior-audit lookup via
    # ``prior_audit_trial_keys``, and (c) the manifest-producing
    # ``draw_sample`` call. With paths replaced by parsed payloads
    # downstream, no helper re-reads the files — closing both the
    # SHA-vs-SHA TOCTOU and the SHA-vs-parsed-content TOCTOU that earlier
    # implementations left open.
    try:
        source_payloads = prepare_sample_sources(
            probe_snapshot_path=probe_snapshot_path,
            rationale_source_path=rationale_source_path,
            stimulus_source_path=stimulus_source_path,
        )
    except FileNotFoundError as exc:
        print(
            f"error: source file missing: {exc}", file=sys.stderr
        )
        return 2
    except (json.JSONDecodeError, ValueError) as exc:
        print(
            f"error: source file malformed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 2
    probe_sha = source_payloads.probe_snapshot_sha256
    rationale_sha = source_payloads.rationale_source_sha256
    stimulus_sha = source_payloads.stimulus_source_sha256

    # Pinning bypasses the eligibility computation entirely, so
    # ``exclude_prior_audits`` is inert under either pinning flag. Coerce
    # to False and emit a stderr note when the user explicitly combined
    # the flags — the SampleArgs validator would otherwise reject with a
    # terser message. The note is only printed when the argparse default
    # or an explicit ``--exclude-prior-audits`` would have yielded
    # ``True``; ``--allow-prior-audit-overlap`` under pinning is already
    # a no-op and needs no commentary.
    if pinned_trial_keys and args.exclude_prior_audits:
        print(
            "note: --exclude-prior-audits is inert under --trials-from / "
            "--trial-keys (pinning bypasses the eligibility filter). The "
            "manifest will record exclude_prior_audits=False to reflect "
            "the actual draw semantics.",
            file=sys.stderr,
        )
    effective_exclude = args.exclude_prior_audits and not pinned_trial_keys

    # The coverage-convergent draw fixes its convergence partition to
    # ``condition`` (Q3); ``--stratify-by`` plays no role in it. Normalize so
    # the default 3-tuple does not trip the SampleArgs validator, emitting a
    # stderr note when the operator passed a different value so the override
    # is never silent. ``coverage_floor`` / ``coverage_rare_max_count`` are
    # ``None`` for the stratified draw (argparse default) and required for the
    # convergent draw (enforced by SampleArgs).
    if args.batch_allocation == "coverage_convergent":
        if tuple(args.stratify_by) != ("condition",):
            print(
                "note: --stratify-by is ignored under --batch-allocation "
                "coverage_convergent; the convergence partition is fixed to "
                "'condition'.",
                file=sys.stderr,
            )
        effective_stratify_by: tuple[str, ...] = ("condition",)
    else:
        effective_stratify_by = tuple(args.stratify_by)

    try:
        sample_args = SampleArgs(
            sample_size=effective_sample_size,
            stratify_by=effective_stratify_by,
            k=args.k,
            probe_snapshot_path=Path(probe_snapshot_relative),
            pinned_trial_keys=pinned_trial_keys,
            exclude_prior_audits=effective_exclude,
            batch_allocation=args.batch_allocation,
            coverage_floor=args.coverage_floor,
            coverage_rare_max_count=args.coverage_rare_max_count,
        )
    except ValidationError as exc:
        # Bare argparse ``type=int`` lets ``--k 0``, ``--k 2``, ``--k -1``
        # through; the SampleArgs validator catches them but raises
        # ``ValidationError`` which would otherwise escape as a traceback.
        # Surface as a clean exit-2, matching the idiom for malformed
        # ``--trials-from`` payloads above.
        print(f"error: invalid sample arguments: {exc}", file=sys.stderr)
        return 2

    if manifest_path.exists():
        existing = SampleManifest.model_validate_json(manifest_path.read_text())
        if existing.skill_fingerprint != args.skill_fingerprint:
            print(
                f"error: fingerprint drift — existing manifest in {audit_dir} "
                f"was drawn under {existing.skill_fingerprint[:12]}…, caller "
                f"passed {args.skill_fingerprint[:12]}…",
                file=sys.stderr,
            )
            return 2
        # Same fingerprint but different sampling-algorithm args would
        # silently overwrite the manifest while the append-only log
        # retained events bound to the prior trial set. Refuse so the
        # user uses a fresh ``--output-dir``. ``args_digest`` covers the
        # sampling-algorithm parameters; rationale-source drift is caught
        # by a separate check below because the rationale source no
        # longer lives on ``SampleArgs``.
        existing_digest = _args_digest(existing.sample_args)
        new_digest = _args_digest(sample_args)
        if existing_digest != new_digest:
            print(
                f"error: cannot re-sample in {audit_dir} with different "
                f"sample arguments — the existing manifest has args_digest "
                f"{existing_digest} but --sample-size / --stratify-by / "
                f"--probe-snapshot imply "
                f"{new_digest}. Use a fresh --output-dir.",
                file=sys.stderr,
            )
            return 2
        # K is deliberately excluded from ``args_digest`` (see
        # ``canonical_json`` in ``args.py``) so that bumping K does not
        # re-roll the trial set. That exclusion is correct for sample
        # identity, but it means the args_digest check above does NOT
        # catch ``--k`` drift. Without this explicit guard, re-running
        # ``sample`` in the same ``--output-dir`` with a different K
        # would silently rewrite the manifest while leaving rep events
        # (and the prior ``sample_drawn``) in ``audit_log.jsonl`` bound
        # to the old K — orphaning replicated reps on a K3 → K1 change
        # and making the audit incomplete on a K1 → K3 change. Refuse
        # and direct the user to a fresh ``--output-dir`` (with
        # ``--trials-from`` if they want the same trial set under a
        # different K).
        if existing.sample_args.k != sample_args.k:
            print(
                f"error: cannot re-sample in {audit_dir} with a different "
                f"K — the existing manifest pins k={existing.sample_args.k} "
                f"but --k={sample_args.k} was passed. K is excluded from "
                f"args_digest so that bumping K does not re-roll the trial "
                f"set, but rep events in audit_log.jsonl are bound to the "
                f"prior K. Use a fresh --output-dir (optionally with "
                f"--trials-from to reuse the same trial set under a "
                f"different K).",
                file=sys.stderr,
            )
            return 2
        # Rationale-source drift: the previous guard covered ``SampleArgs``
        # only. Since ``rationale_source.path`` is no longer on
        # ``SampleArgs``, we compare it explicitly here. Different stage-N
        # snapshots produce different joined eligible frames and different
        # seeds, so rerunning in the same output dir against a different
        # rationale source is the same class of error — an orphaned prior
        # run — and must be refused with the same exit-2 idiom.
        if existing.rationale_source.path != rationale_source_relative:
            print(
                f"error: cannot re-sample in {audit_dir} with a different "
                f"rationale source — the existing manifest pins "
                f"{existing.rationale_source.path!r} but --rationale-source "
                f"normalized to {rationale_source_relative!r}. Use a fresh "
                "--output-dir.",
                file=sys.stderr,
            )
            return 2
        # Stimulus-source path drift refusal — same idiom as rationale.
        if existing.stimulus_source.path != stimulus_source_relative:
            print(
                f"error: cannot re-sample in {audit_dir} with a different "
                f"stimulus source — the existing manifest pins "
                f"{existing.stimulus_source.path!r} but --stimulus-jsonl "
                f"normalized to {stimulus_source_relative!r}. Use a fresh "
                "--output-dir.",
                file=sys.stderr,
            )
            return 2
        # Byte-drift refusal for all three source files. The path/args
        # guards above only catch CLI-level drift; they do NOT catch
        # the case where the user re-runs ``sample`` with identical
        # flags after an in-place edit of probe-snapshot, rationale, or
        # stimulus bytes. The probe-snapshot and rationale SHAs feed
        # ``seed_from_inputs`` (so a mutation to either re-rolls the
        # eligible frame and the trial set), so silently overwriting the
        # manifest would orphan rep events keyed to the prior seed. The
        # stimulus SHA is intentionally excluded from the seed (Decision
        # 2: it augments per-trial context but does not define the
        # eligible frame), but mutating its bytes still invalidates the
        # auditor's pinned context — every downstream rep would render
        # against bytes the audit was not drawn against. All three
        # drifts therefore refuse here at re-entry. Symmetric with the
        # re-hash + refusal already applied at ``aggregate`` time in
        # ``aggregate_probe_audit_records``.
        #
        # The SHAs themselves come from the upfront hoist at the top of
        # ``_cmd_sample`` — re-computing here would open a TOCTOU window
        # vs. the prior-audit lookup and manifest write. ``FileNotFoundError``
        # handling lives at the hoist site (a missing file fails fast
        # before any of the manifest-drift, prior-audit, or draw_sample
        # paths run).
        if probe_sha != existing.probe_snapshot_sha256:
            print(
                f"error: cannot re-sample in {audit_dir} — probe snapshot at "
                f"{probe_snapshot_path} has been mutated since sample time "
                f"(SHA-256 {probe_sha[:12]}… no longer matches pinned "
                f"{existing.probe_snapshot_sha256[:12]}…). Use a fresh "
                "--output-dir, or revert the snapshot bytes.",
                file=sys.stderr,
            )
            return 2
        if rationale_sha != existing.rationale_source.sha256:
            print(
                f"error: cannot re-sample in {audit_dir} — rationale source at "
                f"{rationale_source_path} has been mutated since sample time "
                f"(SHA-256 {rationale_sha[:12]}… no longer matches "
                f"pinned {existing.rationale_source.sha256[:12]}…). Use a "
                "fresh --output-dir, or revert the rationale-source bytes.",
                file=sys.stderr,
            )
            return 2
        if stimulus_sha != existing.stimulus_source.sha256:
            print(
                f"error: cannot re-sample in {audit_dir} — stimulus source at "
                f"{stimulus_source_path} has been mutated since sample time "
                f"(SHA-256 {stimulus_sha[:12]}… no longer matches "
                f"pinned {existing.stimulus_source.sha256[:12]}…). Use a "
                "fresh --output-dir, or revert the stimulus-source bytes.",
                file=sys.stderr,
            )
            return 2

    # Repo root already resolved above (needed for pinning-path
    # relativization); forensics + fingerprint sanity check happen here,
    # after the manifest-drift check, per the existing ordering.
    forensics, recomputed_fp = _collect_skill_forensics(repo_root)

    # H8: When the Skill manifest is present on disk we can recompute
    # the fingerprint ourselves. A mismatch between that and the
    # caller-supplied ``--skill-fingerprint`` means either the Skill
    # was edited between the preprocessing that produced the flag and
    # this invocation, or the flag was fabricated — both cases should
    # fail loudly so drift detection can't be bypassed by passing an
    # arbitrary hex string.
    if recomputed_fp is not None and recomputed_fp != args.skill_fingerprint:
        print(
            f"error: --skill-fingerprint ({args.skill_fingerprint[:12]}…) "
            f"does not match the on-disk Skill fingerprint "
            f"({recomputed_fp[:12]}…). Either the Skill directory drifted "
            "between the preprocessing run and this call, or the flag was "
            "fabricated. Pass --repo-root <path-with-no-Skill> to bypass "
            "for tests.",
            file=sys.stderr,
        )
        return 2

    # Resolve prior-audit exclusion context when the flag is on. The
    # sampler is kept free of storage-layer coupling, so the CLI is the
    # sole call site for ``prior_audit_trial_keys`` and passes the result
    # (keys + contributing session_ids) into ``draw_sample`` as kwargs.
    # Pruning stale index rows first keeps the lookup honest — an index
    # row pointing at a deleted session directory can't contribute trial
    # keys because its manifest is gone. Both ``prune_missing_audits``
    # and ``prior_audit_trial_keys`` are wrapped in try/except so a
    # corrupt index (bad JSON blob, permission error) is surfaced as a
    # warning rather than blocking the primary sampling operation; the
    # conservative fallback is "no priors found" which matches how a
    # fresh setup would behave.
    prior_keys: frozenset[tuple[str, str, int, int]] | None = None
    prior_session_ids: tuple[str, ...] | None = None
    if sample_args.exclude_prior_audits:
        index_base_for_priors = resolve_base_dir()
        try:
            prune_missing_audits(base_dir=index_base_for_priors)
        except Exception as exc:
            print(
                f"warning: could not prune stale index rows before prior-"
                f"audit lookup ({type(exc).__name__}): {exc}; continuing with "
                "the unpruned index",
                file=sys.stderr,
            )
        try:
            # Exclude the current session's own index row from the
            # prior set. Re-running ``sample`` in an existing
            # ``--output-dir`` is the idempotent-drift-guard path above;
            # without this exclusion the helper would return the
            # current run's own trial_keys (persisted during the first
            # invocation) and ``draw_sample`` would subtract them from
            # the eligible frame, silently changing the draw on
            # re-entry even though every seed input is unchanged.
            # First-time invocations have no row under this name yet,
            # so the exclusion is a no-op for them.
            # Byte-aware identity lookup: same path with mutated bytes
            # is a different audit context, so the filter compares
            # rationale and stimulus SHAs alongside their paths. The
            # SHAs come from the upfront hoist above so the same byte
            # values feed both this lookup and the manifest write —
            # closing the TOCTOU window where re-reading per call site
            # could yield diverging identities for one ``sample`` run.
            prior_keys, prior_session_ids = prior_audit_trial_keys(
                skill_fingerprint=args.skill_fingerprint,
                probe_snapshot_sha256=probe_sha,
                rationale_source_path=rationale_source_relative,
                rationale_source_sha256=rationale_sha,
                stimulus_source_path=stimulus_source_relative,
                stimulus_source_sha256=stimulus_sha,
                batch_allocation=sample_args.batch_allocation,
                base_dir=index_base_for_priors,
                exclude_session_id=audit_dir.name,
            )
        except Exception as exc:
            print(
                f"error: failed to enumerate prior audits for exclusion "
                f"({type(exc).__name__}): {exc}. Rerun with "
                "--allow-prior-audit-overlap to skip this lookup.",
                file=sys.stderr,
            )
            return 2

    # ``draw_sample`` raises ValueError on the pinned path when a pinned
    # trial key is absent from the probe snapshot (researcher pointed
    # ``--trials-from`` at a manifest drawn against a different probe
    # snapshot, or passed ``--trial-keys`` naming a trial the snapshot
    # doesn't contain). Same idiom for rationale-source shape / coverage
    # errors raised by the sampler's validation, and for the reduced-
    # frame-below-sample-size error raised when exclusion leaves too few
    # trials. Surface all of these as exit-2 with the sampler's message
    # rather than letting the traceback escape.
    try:
        manifest = draw_sample(
            skill_fingerprint=args.skill_fingerprint,
            args=sample_args,
            forensics=forensics,
            source_payloads=source_payloads,
            rationale_source_repo_relative=rationale_source_relative,
            stimulus_source_repo_relative=stimulus_source_relative,
            selection_source=selection_source,
            prior_audit_keys=prior_keys,
            prior_audit_session_ids=prior_session_ids,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Ordering invariant: append the ``sample_drawn`` event BEFORE
    # writing the manifest. If the event append succeeds and the
    # manifest write fails, the log is self-consistent (the next
    # sample invocation sees no manifest, re-draws, and appends a
    # new ``sample_drawn``). The aggregator filters
    # ``sample_drawn`` events by ``manifest.seed`` so orphaned events
    # from crashed runs cannot hijack ``run_started_at``.
    #
    # Compute the sample timestamp once here so the ``sample_drawn``
    # event's ``recorded_at`` and the index row's ``sample_timestamp``
    # agree exactly. If a row were ever pruned and re-seeded via
    # ``backfill_from_filesystem``, that path derives the timestamp
    # from this same event — so storing the same value up front means
    # the sort key stays stable across a prune+backfill round trip.
    sample_drawn_at = _now_utc()
    append_event(
        _log_path(audit_dir),
        {
            "event_type": "sample_drawn",
            "trial_id": "-",
            "rep_index": 0,
            "recorded_at": sample_drawn_at,
            "sub_agent_session_id": None,
            "payload": {
                "seed": manifest.seed,
                "sample_size": sample_args.sample_size,
                "canonical_args_json": canonical_json(sample_args),
                "skill_fingerprint": args.skill_fingerprint,
            },
        },
    )
    # Atomic manifest write: render to a sibling tempfile on the same
    # filesystem, then ``os.replace`` — the only filesystem operation
    # that is guaranteed atomic on POSIX. A crash mid-write leaves the
    # sibling tempfile behind (cleaned up next run) but never a
    # torn manifest.
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_manifest.write_text(manifest.model_dump_json())
    os.replace(tmp_manifest, manifest_path)
    print(f"sampled {len(manifest.trial_keys)} trials → {manifest_path}")
    # Terminal-batch note: the coverage-convergent draw exhausts the residual
    # frame as its final batch rather than erroring, so a count below the
    # requested size is the convergence endpoint, not a shortfall. Surface it
    # explicitly (the stratified draw hard-errors on a short residual instead).
    if (
        sample_args.batch_allocation == "coverage_convergent"
        and len(manifest.trial_keys) < effective_sample_size
    ):
        print(
            f"note: terminal batch — the residual eligible frame was exhausted, "
            f"so {len(manifest.trial_keys)} trial(s) were drawn (fewer than the "
            f"requested --sample-size {effective_sample_size}). This is the "
            f"convergence endpoint, not a shortfall.",
            file=sys.stderr,
        )

    # Persist a ProbeAuditIndexRecord so the marimo discovery tab can
    # surface this run. Skipped (with a one-line warning) when the
    # audit directory is outside ``$PROBE_AUDIT_DIR``. A write failure
    # (DB locked, disk full, Pydantic validation mismatch) is wrapped
    # and surfaced as a warning rather than an error: the manifest and
    # ``sample_drawn`` event are already on disk, the sample has
    # semantically succeeded, so automated workflows reading the exit
    # code must not misread a non-blocking index hiccup as a sampling
    # failure. Symmetric with the wrapped writes in ``_cmd_next_rep``
    # and ``_cmd_aggregate``.
    index_base = resolve_base_dir()
    session_dir = _resolve_session_dir(audit_dir, index_base)
    if session_dir is None:
        print(
            f"warning: audit_dir {audit_dir} is outside PROBE_AUDIT_DIR "
            f"{index_base}; skipping index update",
            file=sys.stderr,
        )
    else:
        try:
            record = ProbeAuditIndexRecord(
                session_id=audit_dir.name,
                session_dir=session_dir,
                trial_set_sha=compute_trial_set_sha(manifest.trial_keys),
                fp_short=args.skill_fingerprint[:12],
                args_digest=_args_digest(sample_args),
                skill_fingerprint=args.skill_fingerprint,
                probe_snapshot_sha256=manifest.probe_snapshot_sha256,
                k=sample_args.k,
                selection_mode=manifest.selection_mode,
                batch_allocation=sample_args.batch_allocation,
                source_manifest_sha256=(
                    manifest.selection_source.sha256
                    if manifest.selection_source is not None
                    else None
                ),
                sample_timestamp=sample_drawn_at,
                audit_status="sampled",
                rationale_source_path=manifest.rationale_source.path,
                rationale_source_sha256=manifest.rationale_source.sha256,
                stimulus_source_path=manifest.stimulus_source.path,
                stimulus_source_sha256=manifest.stimulus_source.sha256,
            )
            persist_audit_draw(record, base_dir=index_base)
        except Exception as exc:
            print(
                f"warning: failed to persist index row for "
                f"{audit_dir.name!r} at 'sampled' ({type(exc).__name__}): "
                f"{exc}; sample_manifest.json is written and sampling "
                "succeeded",
                file=sys.stderr,
            )
    return 0


# ── subcommand: next-rep ────────────────────────────────────────────────


def _cmd_next_rep(args: argparse.Namespace) -> int:
    audit_dir = Path(args.audit_dir)
    if not _validate_log_or_print_error(_log_path(audit_dir)):
        return 2
    # Transition the index row from 'sampled' to 'in_progress' on first
    # call. Audits without an index row (out-of-tree audit dirs, or
    # audits that pre-date the index) are silently skipped.
    index_transition = _advance_index_to_in_progress(audit_dir)
    if index_transition != 0:
        return index_transition
    manifest = _load_manifest(audit_dir)
    events = read_log(_log_path(audit_dir))
    judged: set[tuple[str, int]] = {
        (e["trial_id"], int(e["rep_index"]))
        for e in events
        if e["event_type"] == "probe_judgment_recorded"
    }
    for trial_key in manifest.trial_keys:
        tid = "|".join(str(p) for p in trial_key)
        for rep in range(1, manifest.sample_args.k + 1):
            if (tid, rep) not in judged:
                print(f"trial_id={tid} rep={rep}")
                return 0
    print("all reps judged")
    return 0


def _advance_index_to_in_progress(audit_dir: Path) -> int:
    """Transition ``audit_status`` to ``in_progress`` on the first call.

    Skip silently when the session is out-of-tree or the index row is
    absent (no index row for this audit). Error out (exit 2) when the row is at
    ``aggregated`` — advancing a completed audit is ill-defined.
    Idempotent on ``in_progress`` (repeated calls are no-ops).

    The caller's primary operation (``next-rep``'s query) must not be
    blocked by index bookkeeping failures — if the write raises because
    of a TOCTOU race against a concurrent writer, a database lock, or a
    Pydantic validation error, we warn and continue rather than abort.
    """
    index_base = resolve_base_dir()
    if _resolve_session_dir(audit_dir, index_base) is None:
        return 0
    existing = load_audit_record(audit_dir.name, base_dir=index_base)
    if existing is None:
        return 0
    if existing.audit_status == "aggregated":
        print(
            f"error: cannot advance completed audit {audit_dir.name!r}; "
            "the audit has been aggregated and its snapshot is final",
            file=sys.stderr,
        )
        return 2
    if existing.audit_status == "in_progress":
        return 0
    try:
        update_audit_status(
            transition_record(existing, audit_status="in_progress"),
            base_dir=index_base,
        )
    except Exception as exc:
        print(
            f"warning: failed to transition index row for "
            f"{audit_dir.name!r} to 'in_progress' ({type(exc).__name__}): "
            f"{exc}",
            file=sys.stderr,
        )
    return 0


# ── subcommand: list-trial-ids ──────────────────────────────────────────


def _cmd_list_trial_ids(args: argparse.Namespace) -> int:
    """Print the manifest's trial_ids, one per line in draw order.

    SKILL.md step 2 points here to enumerate trial_ids before spawning K
    rep sub-agents per trial. Encapsulating the pipe-join + schema access
    behind the CLI keeps the orchestrator prompt free of manifest-schema
    details — a future schema change is absorbed here rather than drifting
    SKILL.md silently. Read-only: no audit event is emitted and the
    append-only log is not touched.
    """
    audit_dir = Path(args.audit_dir)
    manifest = _load_manifest(audit_dir)
    for trial_key in manifest.trial_keys:
        print("|".join(str(p) for p in trial_key))
    return 0


# ── subcommand: fetch-rationale ─────────────────────────────────────────


def _cmd_fetch_rationale(args: argparse.Namespace) -> int:
    audit_dir = Path(args.audit_dir)
    manifest = _check_fingerprint(audit_dir, args.skill_fingerprint)
    if manifest is None:
        return 2
    if not _check_rep_in_range(manifest, args.rep):
        return 2

    trial_key = _parse_trial_id(args.trial_id)
    if trial_key not in (tuple(k) for k in manifest.trial_keys):
        print(
            f"error: trial {args.trial_id} not found in sample manifest",
            file=sys.stderr,
        )
        return 2

    # Rationale and stimulus sources are pinned on the manifest at sample
    # time. Reps resolve them from the manifest rather than receiving CLI
    # args, so the orchestrator cannot accidentally substitute a path and
    # reps have nothing to second-guess. Both ``.path`` fields are
    # repo-relative; anchor against ``--repo-root`` (or the auto-detected
    # git root) so the same bytes resolve whether the rep is invoked from
    # the repo root or a subdirectory.
    repo_root = _resolve_repo_root(args.repo_root)
    rationale_source_path = repo_root / manifest.rationale_source.path
    stimulus_source_path = repo_root / manifest.stimulus_source.path

    # Drift refusal: ``build_rationale_bundle`` reads each source file's
    # bytes exactly once, hashes them, verifies the hash against the
    # manifest-pinned SHA, then parses the bundle from those same bytes.
    # The single read-then-verify-then-parse window eliminates the TOCTOU
    # gap a separate pre-check would open, where a file could be swapped
    # between the SHA verify and the parse. Mirrored by the probe-snapshot
    # guard in ``_cmd_fetch_probe_output`` and the manifest-rehash guard
    # in ``aggregate_probe_audit_records``, so every rep-time and
    # aggregate-time byte read either matches the manifest pin or refuses.
    # ``FileNotFoundError`` is caught here (the file may have been deleted
    # between aggregate and fetch); ``SourceDriftError`` is the SHA-mismatch
    # path; both surface as exit-2 with the bundle builder's wording.
    try:
        bundle, sha = build_rationale_bundle(
            trial_key,
            rationale_source_path,
            stimulus_source_path,
            expected_rationale_source_sha256=manifest.rationale_source.sha256,
            expected_stimulus_source_sha256=manifest.stimulus_source.sha256,
        )
    except FileNotFoundError as exc:
        print(f"error: source file missing: {exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    serialized = canonical_json_of_bundle(bundle)

    # Duplicate-event check lives in an in-lock guard so concurrent
    # retries of the same (trial_id, rep) cannot both append a
    # rationale_fetched event — aggregation uses [0] and would silently
    # drop the second, hiding the anomaly.
    def _guard(events: list[dict]) -> str | None:
        if _has_event(events, "rationale_fetched", args.trial_id, args.rep):
            return "rationale already fetched for this rep"
        return None

    try:
        append_event(
            _log_path(audit_dir),
            {
                "event_type": "rationale_fetched",
                "trial_id": args.trial_id,
                "rep_index": args.rep,
                "recorded_at": _now_utc(),
                "sub_agent_session_id": args.sub_agent_session_id,
                "sub_agent_model": args.sub_agent_model,
                "rationale_source": manifest.rationale_source.path,
                "stimulus_source": manifest.stimulus_source.path,
                "payload": {"bundle_sha256": sha},
            },
            guard=_guard,
        )
    except EventRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(serialized)
    return 0


# ── subcommand: record-independent-reading ──────────────────────────────


def _has_event(
    events: list[dict], event_type: str, trial_id: str, rep: int
) -> bool:
    return any(
        e["event_type"] == event_type
        and e["trial_id"] == trial_id
        and int(e["rep_index"]) == rep
        for e in events
    )


def _check_rep_in_range(manifest: SampleManifest, rep: int) -> bool:
    """Return True if ``rep`` is a valid 1-based index for this audit's K.

    A rep outside ``[1..K]`` would append events the aggregator (which
    iterates ``range(1, K + 1)``) never reads, silently stranding the
    work and surfacing only as an opaque incomplete-coverage refusal at
    aggregate time. Rejecting up front fails fast with the admissible
    range; prints the error to stderr on failure.
    """
    k = manifest.sample_args.k
    if not (1 <= rep <= k):
        print(
            f"error: --rep {rep} is out of range; manifest K={k} admits "
            f"reps 1..{k}",
            file=sys.stderr,
        )
        return False
    return True


def _check_reasonings_nonempty(args: argparse.Namespace) -> bool:
    """Return True if every ``--{flag}-reasoning`` is non-empty after strip.

    The audit's value is the qualitative justification behind each flag;
    an empty (or whitespace-only) reasoning is a hollow record that no
    later phase can reconstruct. Enforced at the CLI boundary (not as a
    model validator) so it never retroactively rejects already-recorded
    logs at aggregate time. Prints the error to stderr on failure.
    """
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        if not getattr(args, f"{flag}_reasoning").strip():
            flag_cli = flag.replace("_", "-")
            print(
                f"error: --{flag_cli}-reasoning must be non-empty",
                file=sys.stderr,
            )
            return False
    return True


def _cmd_record_independent_reading(args: argparse.Namespace) -> int:
    audit_dir = Path(args.audit_dir)
    manifest = _check_fingerprint(audit_dir, args.skill_fingerprint)
    if manifest is None:
        return 2
    if not _check_rep_in_range(manifest, args.rep):
        return 2
    if not _check_reasonings_nonempty(args):
        return 2

    # Guard evaluated INSIDE the log's lock window so the prerequisite
    # and duplicate checks cannot race against a concurrent append for
    # the same (trial_id, rep). A separate read_log() outside the lock
    # would let two writers both see "no prior reading" and both append.
    def _guard(events: list[dict]) -> str | None:
        if not _has_event(events, "rationale_fetched", args.trial_id, args.rep):
            return "cannot record reading before fetch-rationale for this rep"
        if _has_event(events, "probe_output_fetched", args.trial_id, args.rep):
            return (
                "cannot record reading after probe output has been fetched "
                "for this rep — the reading would no longer be independent"
            )
        if _has_event(
            events, "independent_reading_recorded", args.trial_id, args.rep
        ):
            return "independent reading already recorded for this rep"
        return None

    # One ``{flag}`` bool + one ``{flag}_reasoning`` string per flag,
    # collected by looping the single-source flag-key tuple so the
    # recorded payload cannot drift from the probe's seven-flag schema.
    payload: dict = {}
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        payload[flag] = getattr(args, flag)
        payload[f"{flag}_reasoning"] = getattr(args, f"{flag}_reasoning")

    try:
        append_event(
            _log_path(audit_dir),
            {
                "event_type": "independent_reading_recorded",
                "trial_id": args.trial_id,
                "rep_index": args.rep,
                "recorded_at": _now_utc(),
                "sub_agent_session_id": args.sub_agent_session_id,
                "sub_agent_model": args.sub_agent_model,
                "payload": payload,
            },
            guard=_guard,
        )
    except EventRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


# ── subcommand: fetch-probe-output ──────────────────────────────────────


def _cmd_fetch_probe_output(args: argparse.Namespace) -> int:
    audit_dir = Path(args.audit_dir)
    manifest = _check_fingerprint(audit_dir, args.skill_fingerprint)
    if manifest is None:
        return 2
    if not _check_rep_in_range(manifest, args.rep):
        return 2

    trial_key = _parse_trial_id(args.trial_id)
    # ``sample_args.probe_snapshot_path`` is repo-relative (pinned at
    # sample time by the CLI's ``relative_to(repo_root)`` normalization),
    # so anchor against ``--repo-root`` for byte reads — mirroring the
    # treatment of ``manifest.rationale_source.path`` in
    # ``_cmd_fetch_rationale``. Without the anchor, a rep invoked from
    # a subdirectory (or tests operating in tmp_path) would read a
    # CWD-relative path and silently miss the snapshot.
    repo_root = _resolve_repo_root(args.repo_root)
    probe_snapshot_path = repo_root / manifest.sample_args.probe_snapshot_path

    # Drift refusal: ``build_probe_output_bundle`` reads the snapshot
    # bytes once, hashes them, verifies the hash against the manifest
    # pin, then parses the bundle from those same bytes. A pre-check
    # would open a TOCTOU window where the file could be swapped
    # between the SHA verify and the parse — combining rep judgments
    # with a probe-output view they never saw. Mirrors the
    # ``_cmd_fetch_rationale`` integrity contract.
    try:
        bundle, sha = build_probe_output_bundle(
            trial_key,
            probe_snapshot_path,
            expected_probe_snapshot_sha256=manifest.probe_snapshot_sha256,
        )
    except FileNotFoundError as exc:
        print(f"error: probe snapshot file missing: {exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    serialized = canonical_json_of_bundle(bundle)

    # Prerequisite (reading must exist) + duplicate-event check both
    # evaluated inside the log's lock window. A separate read_log()
    # outside the lock would let two concurrent retries both see
    # ``no prior probe_output_fetched`` and both append.
    def _guard(events: list[dict]) -> str | None:
        if not _has_event(
            events, "independent_reading_recorded", args.trial_id, args.rep
        ):
            return (
                "cannot fetch probe output before independent reading is "
                "recorded for this rep"
            )
        if _has_event(events, "probe_output_fetched", args.trial_id, args.rep):
            return "probe output already fetched for this rep"
        return None

    try:
        append_event(
            _log_path(audit_dir),
            {
                "event_type": "probe_output_fetched",
                "trial_id": args.trial_id,
                "rep_index": args.rep,
                "recorded_at": _now_utc(),
                "sub_agent_session_id": args.sub_agent_session_id,
                "sub_agent_model": args.sub_agent_model,
                "payload": {"bundle_sha256": sha},
            },
            guard=_guard,
        )
    except EventRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(serialized)
    return 0


# ── subcommand: record-judgment ─────────────────────────────────────────


def _cmd_record_judgment(args: argparse.Namespace) -> int:
    audit_dir = Path(args.audit_dir)
    manifest = _check_fingerprint(audit_dir, args.skill_fingerprint)
    if manifest is None:
        return 2
    if not _check_rep_in_range(manifest, args.rep):
        return 2
    if not _check_reasonings_nonempty(args):
        return 2

    # Argument-only checks (not stateful) can stay outside the lock. A
    # corrected classification must be present exactly when the rep judged
    # the probe wrong: REQUIRED when probe-correct is false (else the
    # aggregator has no rep-supplied value to compare against the probe),
    # and FORBIDDEN when probe-correct is true (a flag the rep agrees the
    # probe got right has nothing to correct — supplying one is an
    # internally contradictory verdict). The flags that survive as
    # probe-wrong-with-corrected are the only ones the coherence guard
    # below has to scrutinize.
    incorrect_flags: list[str] = []
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        flag_cli = flag.replace("_", "-")
        if getattr(args, f"{flag}_probe_correct") is False:
            if getattr(args, f"{flag}_corrected_classification") is None:
                print(
                    f"error: --{flag_cli}-corrected-classification is required "
                    f"when --{flag_cli}-probe-correct=false",
                    file=sys.stderr,
                )
                return 2
            incorrect_flags.append(flag)
        elif getattr(args, f"{flag}_corrected_classification") is not None:
            print(
                f"error: --{flag_cli}-corrected-classification must be omitted "
                f"when --{flag_cli}-probe-correct=true",
                file=sys.stderr,
            )
            return 2

    # Coherence: a verdict that marks the probe wrong on a flag yet supplies
    # a corrected classification IDENTICAL to the probe's own classification
    # is internally contradictory — "the probe is wrong, and the right answer
    # is exactly what the probe said." Catch it at record time by loading the
    # same probe-output bundle the rep saw: the snapshot path + SHA are
    # pinned on the manifest, so this re-reads and re-verifies the exact
    # bytes ``fetch-probe-output`` served (mirroring that command's integrity
    # contract). The read happens only when at least one flag was judged
    # probe-wrong, so an all-correct judgment needs no probe-snapshot access
    # and stays repo-root-independent.
    probe_classification: dict[str, bool | None] = {}
    if incorrect_flags:
        repo_root = _resolve_repo_root(args.repo_root)
        probe_snapshot_path = repo_root / manifest.sample_args.probe_snapshot_path
        try:
            bundle, _bundle_sha = build_probe_output_bundle(
                _parse_trial_id(args.trial_id),
                probe_snapshot_path,
                expected_probe_snapshot_sha256=manifest.probe_snapshot_sha256,
            )
        except FileNotFoundError as exc:
            print(f"error: probe snapshot file missing: {exc}", file=sys.stderr)
            return 2
        except (KeyError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        probe_classification = {
            flag: getattr(bundle, f"aggregated_{flag}") for flag in incorrect_flags
        }

    # State-dependent checks (prerequisite + duplicate) must share the
    # log's lock window to be race-free under concurrent retry spawns. The
    # contradiction check runs last so a duplicate or out-of-order call is
    # refused on those grounds before its payload is scrutinized — and the
    # probe-snapshot read above stays outside the lock.
    def _guard(events: list[dict]) -> str | None:
        if not _has_event(events, "probe_output_fetched", args.trial_id, args.rep):
            return "cannot record judgment before fetch-probe-output for this rep"
        if _has_event(
            events, "probe_judgment_recorded", args.trial_id, args.rep
        ):
            return "probe-auditor alignment judgment already recorded for this rep"
        for flag in incorrect_flags:
            corrected = getattr(args, f"{flag}_corrected_classification")
            if corrected == probe_classification[flag]:
                flag_cli = flag.replace("_", "-")
                return (
                    f"--{flag_cli}-corrected-classification="
                    f"{str(corrected).lower()} contradicts the probe's own "
                    f"classification: when --{flag_cli}-probe-correct=false the "
                    f"corrected classification must differ from the probe's "
                    f"classification ({str(probe_classification[flag]).lower()}) "
                    f"for this flag"
                )
        return None

    # Per flag: a ``{flag}_probe_correct`` bool, a ``{flag}_reasoning``
    # string, and a ``{flag}_corrected_classification`` (None unless the
    # rep judged the probe wrong). Looped over the single-source flag-key
    # tuple so the recorded payload tracks the probe's seven-flag schema.
    payload: dict = {}
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        payload[f"{flag}_probe_correct"] = getattr(
            args, f"{flag}_probe_correct"
        )
        payload[f"{flag}_reasoning"] = getattr(args, f"{flag}_reasoning")
        payload[f"{flag}_corrected_classification"] = getattr(
            args, f"{flag}_corrected_classification"
        )

    try:
        append_event(
            _log_path(audit_dir),
            {
                "event_type": "probe_judgment_recorded",
                "trial_id": args.trial_id,
                "rep_index": args.rep,
                "recorded_at": _now_utc(),
                "sub_agent_session_id": args.sub_agent_session_id,
                "sub_agent_model": args.sub_agent_model,
                "payload": payload,
            },
            guard=_guard,
        )
    except EventRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


# ── subcommand: status ──────────────────────────────────────────────────


def _cmd_status(args: argparse.Namespace) -> int:
    audit_dir = Path(args.audit_dir)
    if not _validate_log_or_print_error(_log_path(audit_dir)):
        return 2
    manifest = _load_manifest(audit_dir)
    events = read_log(_log_path(audit_dir))
    trial_keys = [tuple(k) for k in manifest.trial_keys]
    # Valid (trial_id, rep) pairs for the CURRENT manifest. Re-sampling
    # into an existing audit directory (same fingerprint, different
    # args) leaves stale judgment events behind in the append-only log;
    # restricting ``judged`` to this set keeps the counter honest and
    # stops the "(complete)" label from turning on prematurely.
    valid_pairs = {
        ("|".join(str(p) for p in tk), rep)
        for tk in trial_keys
        for rep in range(1, manifest.sample_args.k + 1)
    }
    judged: set[tuple[str, int]] = {
        (e["trial_id"], int(e["rep_index"]))
        for e in events
        if e["event_type"] == "probe_judgment_recorded"
    } & valid_pairs
    total = len(trial_keys) * manifest.sample_args.k
    remaining = total - len(judged)
    # The ``(complete)`` label is load-bearing — the SKILL.md orchestrator
    # reads this line to decide whether to keep spawning retry sub-agents.
    # Emit a count of the outstanding reps when the run is not yet done
    # so the label can never mislead a caller into stopping early.
    suffix = "(complete)" if remaining == 0 else f"({remaining} remaining)"
    print(f"coverage: {len(judged)}/{total} reps judged {suffix}")
    incomplete = []
    for trial_key in trial_keys:
        tid = "|".join(str(p) for p in trial_key)
        missing = [
            rep
            for rep in range(1, manifest.sample_args.k + 1)
            if (tid, rep) not in judged
        ]
        if missing:
            incomplete.append((tid, missing))
    for tid, missing in incomplete[:10]:
        print(f"  missing trial_id={tid} reps={missing}")
    if len(incomplete) > 10:
        print(f"  … and {len(incomplete) - 10} more incomplete trials")
    return 0


# ── argument parser ─────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probe-audit")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # sample
    p = sub.add_parser("sample")
    p.add_argument("--skill-fingerprint", required=True)
    p.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "trial count; required unless --trials-from or --trial-keys is "
            "passed (in which case it's derived from the pinned set)"
        ),
    )
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--probe-snapshot", required=True)
    p.add_argument(
        "--rationale-source",
        required=True,
        help=(
            "path to the stage-N snapshot (with 'trial_outcomes') whose "
            "rationales the probe classified; pinned on the manifest so "
            "reps resolve it deterministically from --audit-dir"
        ),
    )
    p.add_argument(
        "--stimulus-jsonl",
        required=True,
        help=(
            "path to the assembled stimulus JSONL (with 'custom_id', "
            "'system_prompt', 'messages') used for scenario context "
            "extraction on the auditor's bundle; pinned on the manifest "
            "so reps resolve it deterministically from --audit-dir. "
            "Excluded from the sampling seed — augments scenario context "
            "but does not change which trials are eligible."
        ),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--stratify-by",
        nargs="+",
        default=["condition", "articulation", "split_vote"],
    )
    p.add_argument(
        "--batch-allocation",
        choices=["stratified", "coverage_convergent"],
        default="stratified",
        help=(
            "parametric draw algorithm: 'stratified' (default, equal "
            "allocation across occupied strata) or 'coverage_convergent' "
            "(iterative breadth-first floor on rare flag classes then "
            "proportional convergence to the pool; requires "
            "--exclude-prior-audits, --coverage-floor, --coverage-rare-max-count)"
        ),
    )
    p.add_argument(
        "--coverage-floor",
        type=int,
        default=None,
        help=(
            "coverage-convergent only: per-rare-flag early-coverage target F "
            "(positive int). Required when --batch-allocation coverage_convergent"
        ),
    )
    p.add_argument(
        "--coverage-rare-max-count",
        type=int,
        default=None,
        help=(
            "coverage-convergent only: a flag is 'rare' (and floored) iff its "
            "eligible-pool TRUE count is at most this threshold C (positive "
            "int; recommend ~100). Required when --batch-allocation "
            "coverage_convergent"
        ),
    )
    p.add_argument(
        "--repo-root",
        default=None,
        help=(
            "repository root used for forensics + fingerprint sanity "
            "check (defaults to `git rev-parse --show-toplevel`)"
        ),
    )
    # Trial-pinning flags — mutually exclusive; neither flag falls through
    # to the existing stratified draw so parametric callers keep working.
    pin = p.add_mutually_exclusive_group(required=False)
    pin.add_argument(
        "--trials-from",
        default=None,
        help=(
            "path to a prior sample_manifest.json; pins the same trial set "
            "and records its repo-relative path + SHA-256 on the new manifest"
        ),
    )
    pin.add_argument(
        "--trial-keys",
        default=None,
        help=(
            "comma-separated trial keys, each formatted as "
            "'config_key|example_id|batch_index|trial'"
        ),
    )
    # Prior-audit exclusion — mutually exclusive flags that set the
    # ``SampleArgs.exclude_prior_audits`` boolean. Default is exclusion-on
    # so the iterative pooled-audit workflow (the primary use case) works
    # without ceremony; ``--allow-prior-audit-overlap`` is the opt-out for
    # niche cases like auditor-reliability studies on the same trials.
    exclusion = p.add_mutually_exclusive_group(required=False)
    exclusion.add_argument(
        "--exclude-prior-audits",
        dest="exclude_prior_audits",
        action="store_true",
        help=(
            "subtract trial keys drawn by prior audits under the same "
            "byte-aware pooling identity tuple (skill_fingerprint, "
            "probe_snapshot_sha256, rationale_source_path/_sha256, "
            "stimulus_source_path/_sha256, batch_allocation) from the "
            "eligible frame before the draw (default)"
        ),
    )
    exclusion.add_argument(
        "--allow-prior-audit-overlap",
        dest="exclude_prior_audits",
        action="store_false",
        help=(
            "draw from the full eligible frame without subtracting prior-"
            "audit trial keys — use for auditor-reliability studies that "
            "deliberately re-audit the same trials"
        ),
    )
    p.set_defaults(exclude_prior_audits=True, func=_cmd_sample)

    # next-rep
    p = sub.add_parser("next-rep")
    p.add_argument("--audit-dir", required=True)
    p.set_defaults(func=_cmd_next_rep)

    # list-trial-ids
    p = sub.add_parser("list-trial-ids")
    p.add_argument("--audit-dir", required=True)
    p.set_defaults(func=_cmd_list_trial_ids)

    # fetch-rationale
    p = sub.add_parser("fetch-rationale")
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--trial-id", required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--skill-fingerprint")
    p.add_argument("--sub-agent-session-id")
    p.add_argument("--sub-agent-model", required=True)
    p.add_argument(
        "--repo-root",
        default=None,
        help=(
            "repository root used to anchor the manifest-pinned "
            "rationale_source AND stimulus_source paths (defaults to "
            "`git rev-parse --show-toplevel`); tests that build audits "
            "in a tmp dir should point this at the same root passed to "
            "`sample`"
        ),
    )
    p.set_defaults(func=_cmd_fetch_rationale)

    # record-independent-reading
    p = sub.add_parser("record-independent-reading")
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--trial-id", required=True)
    p.add_argument("--rep", type=int, required=True)
    # One --{flag} bool + one --{flag}-reasoning string per flag, looped
    # over the single-source flag-key tuple so the CLI surface cannot
    # drift from the probe's seven-flag schema.
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        flag_cli = flag.replace("_", "-")
        p.add_argument(f"--{flag_cli}", type=_parse_bool, required=True)
        p.add_argument(f"--{flag_cli}-reasoning", required=True)
    p.add_argument("--skill-fingerprint")
    p.add_argument("--sub-agent-session-id")
    p.add_argument("--sub-agent-model", required=True)
    p.set_defaults(func=_cmd_record_independent_reading)

    # fetch-probe-output
    p = sub.add_parser("fetch-probe-output")
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--trial-id", required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--skill-fingerprint")
    p.add_argument("--sub-agent-session-id")
    p.add_argument("--sub-agent-model", required=True)
    p.add_argument(
        "--repo-root",
        default=None,
        help=(
            "repository root used to anchor the manifest-pinned "
            "probe-snapshot path (defaults to `git rev-parse "
            "--show-toplevel`); tests that build audits in a tmp dir "
            "should point this at the same root passed to `sample`"
        ),
    )
    p.set_defaults(func=_cmd_fetch_probe_output)

    # record-judgment
    p = sub.add_parser("record-judgment")
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--trial-id", required=True)
    p.add_argument("--rep", type=int, required=True)
    # Per flag: a --{flag}-probe-correct bool, a --{flag}-reasoning
    # string, and an optional --{flag}-corrected-classification bool
    # (required by _cmd_record_judgment only when probe-correct is false).
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        flag_cli = flag.replace("_", "-")
        p.add_argument(
            f"--{flag_cli}-probe-correct", type=_parse_bool, required=True
        )
        p.add_argument(f"--{flag_cli}-reasoning", required=True)
        p.add_argument(
            f"--{flag_cli}-corrected-classification", type=_parse_bool
        )
    p.add_argument("--skill-fingerprint")
    p.add_argument("--sub-agent-session-id")
    p.add_argument("--sub-agent-model", required=True)
    p.add_argument(
        "--repo-root",
        default=None,
        help=(
            "repository root used to anchor the manifest-pinned "
            "probe-snapshot path when a flag is judged probe-incorrect "
            "(defaults to `git rev-parse --show-toplevel`); needed so the "
            "coherence guard can compare each corrected classification "
            "against the probe's own classification. Tests that build "
            "audits in a tmp dir should point this at the same root passed "
            "to `sample`"
        ),
    )
    p.set_defaults(func=_cmd_record_judgment)

    # status
    p = sub.add_parser("status")
    p.add_argument("--audit-dir", required=True)
    p.set_defaults(func=_cmd_status)

    # aggregate
    p = sub.add_parser("aggregate")
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--auditor-model-version", default="unknown")
    p.add_argument("--skill-fingerprint")
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="allow trials that lack a full five-event trace per rep",
    )
    p.add_argument(
        "--repo-root",
        help=(
            "repository root used to anchor the rationale source path "
            "stored on the manifest; defaults to CWD"
        ),
    )
    p.set_defaults(func=_cmd_aggregate)

    p = sub.add_parser("report")
    p.add_argument("--audit-dir", required=True)
    p.add_argument(
        "--repo-root",
        help=(
            "repository root used to anchor the rationale source path "
            "stored on the manifest; defaults to CWD"
        ),
    )
    p.set_defaults(func=_cmd_report)

    return parser


def _cmd_aggregate(args: argparse.Namespace) -> int:
    audit_dir = Path(args.audit_dir)
    if args.skill_fingerprint is not None and _check_fingerprint(
        audit_dir, args.skill_fingerprint
    ) is None:
        return 2
    try:
        results = aggregate_probe_audit_records(
            audit_dir,
            args.auditor_model_version,
            allow_incomplete=args.allow_incomplete,
            repo_root=_resolve_repo_root(getattr(args, "repo_root", None)),
        )
    except AggregateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    (audit_dir / "snapshot.json").write_text(results.model_dump_json())
    print(f"aggregated {len(results.records)} trial records → snapshot.json")

    # Transition the index row to 'aggregated' and stamp the completion
    # fields. Out-of-tree audit dirs and audits without an index row are
    # silently skipped — aggregation itself is unaffected.
    # A write failure (TOCTOU race, DB lock, validation mismatch) is
    # wrapped and surfaced as a warning rather than an error: snapshot.
    # json is already on disk and the primary `aggregate` operation
    # succeeded, so automated workflows relying on the exit code must
    # not misread a non-blocking index hiccup as an aggregation failure.
    index_base = resolve_base_dir()
    if _resolve_session_dir(audit_dir, index_base) is not None:
        try:
            existing = load_audit_record(audit_dir.name, base_dir=index_base)
            if existing is not None:
                update_audit_status(
                    transition_record(
                        existing,
                        audit_status="aggregated",
                        snapshot_path=f"{existing.session_dir}/snapshot.json",
                        auditor_model_versions=results.provenance.auditor_model_versions,
                        audit_completion_timestamp=results.provenance.run_completed_at,
                    ),
                    base_dir=index_base,
                )
        except Exception as exc:
            print(
                f"warning: failed to transition index row for "
                f"{audit_dir.name!r} to 'aggregated' "
                f"({type(exc).__name__}): {exc}; snapshot.json is "
                "written and aggregation succeeded",
                file=sys.stderr,
            )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    audit_dir = Path(args.audit_dir)
    # Always validate the log up front — even if we read from a prior
    # snapshot.json, a tampered log between aggregate and report should
    # be surfaced rather than silently ignored.
    if not _validate_log_or_print_error(_log_path(audit_dir)):
        return 2
    snap_path = audit_dir / "snapshot.json"
    if snap_path.exists():
        results = ProbeAuditResults.model_validate_json(snap_path.read_text())
    else:
        # Mirror _cmd_aggregate's error contract — surface log
        # validation failures and incomplete-coverage refusals as a
        # clean CLI error instead of an unhandled traceback.
        try:
            results = aggregate_probe_audit_records(
                audit_dir,
                "unknown",
                repo_root=_resolve_repo_root(getattr(args, "repo_root", None)),
            )
        except AggregateError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    n = len(results.records)
    if n == 0:
        print("no records")
        return 0
    print(f"trials: {n}")
    # One alignment line per flag: the count of trials whose aggregated
    # probe-correct judgment resolved True for that flag. Flags the
    # coverage-convergent draw floored as rare (``classified_rare_flags``)
    # are marked base-rate-driven — the probe rarely classifies them
    # positive, so the shared-negative class dominates the agreement and the
    # rate is NOT a reliability estimate on the probe's positive calls. The
    # de-pooled per-value-class view lives in the notebook / summary report;
    # ``None`` (stratified draw) leaves every line unmarked.
    rare_flags = set(results.provenance.classified_rare_flags or ())
    for flag in RATIONALE_ANALYSIS_FLAG_KEYS:
        n_correct = sum(
            1
            for r in results.records
            if getattr(r, f"aggregated_probe_correct_{flag}") is True
        )
        marker = "  [rare flag — base-rate-driven]" if flag in rare_flags else ""
        print(f"{flag} alignment: {n_correct}/{n}{marker}")
    if rare_flags:
        print(
            "note: flags marked base-rate-driven are rarely classified "
            "positive by the probe, so their alignment is dominated by the "
            "shared negative class and is not a reliability estimate on the "
            "probe's positive calls; see the de-pooled per-value-class table "
            "in the review notebook / summary report."
        )
    # Replication line. Distinct from the K row in the snapshot's
    # provenance — the report is the only signal a non-notebook caller
    # sees, so spell out what K=1 means rather than printing a bare
    # ``K=1`` and leaving the reviewer to infer "no replication evidence".
    if results.provenance.replication_evidence == "none":
        print("replication: 1 rep (no replication evidence)")
    else:
        print(f"replication: K={results.provenance.k} replicated reps")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. ``argv`` defaults to ``sys.argv[1:]``."""
    parser = _build_parser()
    ns = parser.parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":
    sys.exit(main())
