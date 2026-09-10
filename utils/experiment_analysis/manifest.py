"""Experiment manifest — the single authority on which jobs enter the analysis.

Each entry declares a configuration and the ordered list of job_ids for that
configuration's batches. List position determines batch order; the pipeline
validates this against created_at timestamps at load time.
"""

from pydantic import BaseModel, ConfigDict, model_validator


class ManifestEntry(BaseModel):
    """One configuration's batch history in the experiment.

    The config_key is fixed at skeleton generation time. The job_ids tuple
    grows as the researcher appends job IDs after each batch submission.
    Position is the batch index: job_ids[0] = batch 0, etc.
    """

    model_config = ConfigDict(frozen=True)

    config_key: str  # e.g. "openai--gpt-5.2--xhigh"
    job_ids: tuple[str, ...]  # ordered by batch index; empty = unfilled skeleton


class ExperimentManifest(BaseModel):
    """Declares which ProcessedJob outputs constitute this experiment's data.

    Each entry maps a config_key to an ordered list of job_ids. The list
    position is the batch index; the pipeline validates created_at
    monotonicity at load time. Config keys are fixed at skeleton generation
    time; job_ids are appended by the researcher after each batch submission.
    """

    model_config = ConfigDict(frozen=True)

    stage: str  # e.g. "stage_01"
    dataset_sha256: str  # ties results to a specific dataset version
    entries: tuple[ManifestEntry, ...]

    @model_validator(mode="after")
    def validate_entries(self) -> "ExperimentManifest":
        """Ensure at least one entry, no duplicate config_keys, no duplicate job_ids."""
        if not self.entries:
            raise ValueError("manifest must contain at least one entry")

        # No duplicate config_keys
        config_keys = [e.config_key for e in self.entries]
        if len(config_keys) != len(set(config_keys)):
            dupes = [k for k in config_keys if config_keys.count(k) > 1]
            raise ValueError(f"duplicate config_key(s): {sorted(set(dupes))}")

        # No duplicate job_ids across all entries
        all_ids: list[str] = []
        for e in self.entries:
            all_ids.extend(jid for jid in e.job_ids if jid)
        seen: set[str] = set()
        for jid in all_ids:
            if jid in seen:
                raise ValueError(f"duplicate job_id across entries: {jid!r}")
            seen.add(jid)

        return self
