from __future__ import annotations

import csv
import json
import os
from typing import Iterable, Mapping, Any


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def append_records_csv(path: str, records: Iterable[Mapping[str, Any]]) -> None:
    records = list(records)
    if not records:
        return
    ensure_dir(path)
    exists = os.path.exists(path)

    # Stable union of keys.
    keys = []
    for rec in records:
        for key in rec.keys():
            if key not in keys:
                keys.append(key)

    if exists:
        # Preserve old header if file already exists.
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            try:
                old_keys = next(reader)
                for k in old_keys:
                    if k not in keys:
                        keys.insert(0, k)
            except StopIteration:
                pass

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not exists or os.path.getsize(path) == 0:
            writer.writeheader()
        for rec in records:
            writer.writerow({k: serialize(rec.get(k, "")) for k in keys})


def serialize(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return value
