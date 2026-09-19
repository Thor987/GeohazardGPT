"""Combine processed source text into a corpus for SFT sample generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def clean_text(value: object, min_chars: int) -> Optional[str]:
    """Return normalized text, or None when the value is not usable."""
    if not isinstance(value, str):
        return None
    text = str(value).strip()
    return text if len(text) >= min_chars else None


def choose_text_column(frame: pd.DataFrame, path: Path) -> Optional[str]:
    """Choose the text column used by a processed source file."""
    name = path.name.lower()
    if "source_papers" in name:
        candidates = ("Abstract1", "Abstract", "text", "Full Text1", "Full Text")
    else:
        candidates = ("Full Text1", "Full Text", "text", "Abstract1", "Abstract")
    return next((column for column in candidates if column in frame.columns), None)


def iter_parquet_texts(source_dir: Path, min_chars: int) -> Iterator[str]:
    """Yield text records from all Parquet files below source_dir."""
    for path in sorted(source_dir.rglob("*.parquet")):
        frame = pd.read_parquet(path)
        column = choose_text_column(frame, path)
        if column is None:
            print(f"Skipped {path}: no supported text column.")
            continue
        count = 0
        for value in frame[column].tolist():
            text = clean_text(value, min_chars)
            if text is not None:
                count += 1
                yield text
        print(f"Read {count} records from {path} ({column}).")


def iter_json_texts(c4_dir: Path, min_chars: int, max_files: Optional[int]) -> Iterator[str]:
    """Yield text fields from JSONL files below a filtered C4 directory."""
    paths = sorted({*c4_dir.rglob("*.json"), *c4_dir.rglob("*.jsonl")})
    if max_files is not None:
        paths = paths[:max_files]

    for path in paths:
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipped invalid JSON: {path}:{line_number}")
                    continue
                if not isinstance(record, dict):
                    continue
                text = clean_text(record.get("text"), min_chars)
                if text is not None:
                    count += 1
                    yield text
        print(f"Read {count} records from {path}.")


def write_jsonl(records: Iterable[str], output: Path, batch_size: int) -> int:
    """Write records as JSONL without storing the complete corpus in memory."""
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for text in records:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            if count % batch_size == 0:
                print(f"Written {count} records.")
    return count


def write_parquet(records: Iterable[str], output: Path, batch_size: int) -> int:
    """Write records as Parquet in batches."""
    writer: Optional[pq.ParquetWriter] = None
    count = 0
    batch = []
    try:
        for text in records:
            batch.append(text)
            if len(batch) >= batch_size:
                table = pa.table({"text": batch})
                if writer is None:
                    writer = pq.ParquetWriter(output, table.schema)
                writer.write_table(table)
                count += len(batch)
                batch.clear()
                print(f"Written {count} records.")
        if batch:
            table = pa.table({"text": batch})
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema)
            writer.write_table(table)
            count += len(batch)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pq.write_table(pa.table({"text": pa.array([], type=pa.string())}), output)
    return count


def build_records(
    source_dir: Path,
    c4_dir: Optional[Path],
    min_chars: int,
    max_source_records: Optional[int],
    max_c4_records: Optional[int],
    max_c4_files: Optional[int],
) -> Iterator[str]:
    """Yield source records followed by optional filtered C4 records."""
    source_count = 0
    for text in iter_parquet_texts(source_dir, min_chars):
        if max_source_records is not None and source_count >= max_source_records:
            break
        source_count += 1
        yield text

    if c4_dir is None:
        return

    c4_count = 0
    for text in iter_json_texts(c4_dir, min_chars, max_c4_files):
        if max_c4_records is not None and c4_count >= max_c4_records:
            break
        c4_count += 1
        yield text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing processed literature and book Parquet files.",
    )
    parser.add_argument(
        "--c4-dir",
        type=Path,
        help="Optional directory containing filtered C4 JSONL files.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL or Parquet path.")
    parser.add_argument(
        "--format",
        choices=("jsonl", "parquet"),
        help="Output format; inferred from the output suffix when omitted.",
    )
    parser.add_argument("--min-chars", type=int, default=10, help="Minimum text length.")
    parser.add_argument("--max-source-records", type=int, help="Optional source-record limit.")
    parser.add_argument("--max-c4-records", type=int, help="Optional C4-record limit.")
    parser.add_argument("--max-c4-files", type=int, help="Optional C4-file limit.")
    parser.add_argument("--batch-size", type=int, default=10000, help="Write batch size.")
    args = parser.parse_args()

    if not args.source_dir.is_dir() or (args.c4_dir is not None and not args.c4_dir.is_dir()):
        parser.error("Source directories must exist.")
    if args.min_chars < 1 or args.batch_size < 1:
        parser.error("min-chars and batch-size must be positive.")
    for limit in (args.max_source_records, args.max_c4_records, args.max_c4_files):
        if limit is not None and limit < 1:
            parser.error("Record and file limits must be positive.")

    output_format = args.format
    if output_format is None:
        output_format = "parquet" if args.output.suffix.lower() == ".parquet" else "jsonl"
    output = args.output.with_suffix("." + output_format)
    if output.exists():
        parser.error("Output exists; choose a new output file.")
    if output.resolve().is_relative_to(args.source_dir.resolve()):
        parser.error("Output must be outside the source directory.")
    if args.c4_dir and output.resolve().is_relative_to(args.c4_dir.resolve()):
        parser.error("Output must be outside the C4 directory.")
    output.parent.mkdir(parents=True, exist_ok=True)

    records = build_records(
        source_dir=args.source_dir,
        c4_dir=args.c4_dir,
        min_chars=args.min_chars,
        max_source_records=args.max_source_records,
        max_c4_records=args.max_c4_records,
        max_c4_files=args.max_c4_files,
    )
    if output_format == "jsonl":
        count = write_jsonl(records, output, args.batch_size)
    else:
        count = write_parquet(records, output, args.batch_size)
    print(f"Saved {count} records to {output}.")


if __name__ == "__main__":
    main()
