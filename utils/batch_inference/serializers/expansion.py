"""Trial expansion for batch inference datasets.

Replicates each example N times with suffixed custom_id values,
enabling repeated evaluations of the same inputs.
"""

from utils.batch_inference.dataset import Dataset
from utils.batch_inference.example import Example


def expand_trials(dataset: Dataset, num_trials: int) -> Dataset:
    """Replicate each example in a dataset with trial-suffixed custom_ids.

    Always suffixes IDs — even when num_trials=1 — to keep the format
    consistent across all batch jobs (e.g., "eval-001_trial_001").

    Zero-pads the trial number to the width of num_trials
    (e.g., num_trials=100 produces _trial_001 through _trial_100).
    """
    if num_trials < 1:
        raise ValueError(f"num_trials must be >= 1, got {num_trials}")

    # Minimum 3-digit padding (e.g., _trial_001); widens for num_trials >= 1000
    width = max(3, len(str(num_trials)))
    expanded: list[Example] = [
        example.model_copy(
            update={"custom_id": f"{example.custom_id}_trial_{trial:0{width}d}"}
        )
        for example in dataset.examples
        for trial in range(1, num_trials + 1)
    ]
    return Dataset(examples=expanded)
