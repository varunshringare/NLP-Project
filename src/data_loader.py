"""
Data loading utilities for the AmbiStory dataset.

The dataset is stored as a single JSON object whose top-level keys are
string integers ("0", "1", …).  Each value is a sample dict containing
the story components and human annotation fields.
"""

import json
from pathlib import Path
from typing import Any


def load_json_dataset(file_path: Path) -> dict[str, dict[str, Any]]:
    """
    Load an AmbiStory JSON file and return it as a plain Python dict.

    Parameters
    ----------
    file_path : Path
        Path to a train.json or dev.json file.

    Returns
    -------
    dict[str, dict]
        Mapping from string sample IDs to sample dicts.

    Raises
    ------
    FileNotFoundError
        If the given path does not exist.
    ValueError
        If the file cannot be parsed as a JSON object.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object at the top level of {file_path}, "
            f"got {type(data).__name__}."
        )

    return data


def extract_samples(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Flatten the top-level dict into a list of sample dicts, each carrying
    the string key as a ``"id"`` field for downstream convenience.

    Parameters
    ----------
    data : dict[str, dict]
        Raw dataset dict as returned by :func:`load_json_dataset`.

    Returns
    -------
    list[dict]
        List of sample dicts, each augmented with an ``"id"`` key.
    """
    samples = []
    for key, sample in data.items():
        entry = dict(sample)   # shallow copy so we do not mutate the original
        entry["id"] = str(key)
        samples.append(entry)
    return samples
