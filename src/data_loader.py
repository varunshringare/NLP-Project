
import json
from pathlib import Path
from typing import Any


def load_json_dataset(file_path: Path) -> dict[str, dict[str, Any]]:
    
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
    
    samples = []
    for key, sample in data.items():
        entry = dict(sample)   # shallow copy so we do not mutate the original
        entry["id"] = str(key)
        samples.append(entry)
    return samples
