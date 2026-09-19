"""Validate and merge generated SFT JSON files into one training file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Set, Tuple


REQUIRED_FIELDS = ("instruction", "input", "output")


def read_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield records from one generated JSON file."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("the file must contain a JSON object or list")
    for record in value:
        if isinstance(record, dict):
            yield record


def valid_record(record: Dict[str, Any]) -> bool:
    """Check the minimum instruction-tuning schema."""
    return all(isinstance(record.get(field), str) and record[field].strip() for field in REQUIRED_FIELDS)


def record_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return the exact-match key used by optional deduplication."""
    return tuple(str(record[field]).strip() for field in REQUIRED_FIELDS)


def collect_records(input_dir: Path, deduplicate: bool) -> Iterable[Dict[str, str]]:
    """Collect valid final outputs and ignore partial checkpoint files."""
    seen: Set[Tuple[str, str, str]] = set()
    for path in sorted(input_dir.rglob("*_generated.json")):
        if "_partial" in path.stem:
            continue
        try:
            records = list(read_records(path))
        except Exception as exc:
            print(f"Skipped {path}: {exc}")
            continue
        for record in records:
            if not valid_record(record):
                print(f"Skipped invalid record in {path}.")
                continue
            key = record_key(record)
            if deduplicate and key in seen:
                continue
            seen.add(key)
            yield {field: key[index] for index, field in enumerate(REQUIRED_FIELDS)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing generated JSON files.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON array path.")
    parser.add_argument("--deduplicate", action="store_true", help="Remove exact duplicate triples.")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error("Input directory does not exist.")
    if args.output.exists():
        parser.error("Output exists; choose a new output file.")

    records = list(collect_records(args.input_dir, args.deduplicate))
    if not records:
        parser.error("No valid generated samples found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} valid SFT records to {args.output}.")


if __name__ == "__main__":
    main()
