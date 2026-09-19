"""Extract PDF text, attach article metadata, and write a Parquet corpus."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

try:
    import pymupdf
except ImportError:  # pragma: no cover - compatibility with older installations
    import fitz as pymupdf

try:
    from multi_column import column_boxes
except ImportError:
    column_boxes = None


METADATA_FIELDS = [
    "Full Text",
    "Article Title",
    "Source Title",
    "Keywords Plus",
    "Abstract",
    "Publisher",
    "Publication Year",
    "DOI",
    "Open Access Designations",
]


def extract_pdf_text(pdf_path: Path, footer_margin: int = 50) -> str:
    """Extract text in reading order, using the optional column helper when available."""
    parts: List[str] = []
    document = pymupdf.open(pdf_path)
    try:
        for page in document:
            if column_boxes is None:
                parts.append(page.get_text("text", sort=True))
                continue
            try:
                boxes = column_boxes(page, footer_margin=footer_margin, no_image_text=True)
            except Exception:
                boxes = []
            if boxes:
                parts.append("".join(page.get_text("text", clip=box, sort=True) for box in boxes))
            else:
                parts.append(page.get_text("text", sort=True))
    finally:
        document.close()
    return "\n".join(parts)


def convert_pdfs(pdf_dir: Path, txt_dir: Path) -> None:
    """Convert all PDFs in a directory, preserving existing text files."""
    txt_dir.mkdir(parents=True, exist_ok=True)
    processed = skipped = 0
    started = time.time()
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        txt_path = txt_dir / f"{pdf_path.stem}.txt"
        if txt_path.exists():
            skipped += 1
            continue
        txt_path.write_text(extract_pdf_text(pdf_path), encoding="utf-8")
        processed += 1
        print(f"Extracted {pdf_path.name}.")
    print(f"PDF extraction finished: {processed} converted, {skipped} skipped, {time.time() - started:.1f}s.")


def safe_value(row: pd.Series, key: str) -> Any:
    value = row.get(key, "")
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float, bool)):
        return value
    return str(value)


def attach_metadata(
    txt_dir: Path,
    metadata_path: Path,
    json_dir: Path,
    failed_report: Path,
    fuzzy_threshold: int,
) -> List[Dict[str, Any]]:
    """Match extracted text files to article metadata and write JSON records."""
    frame = pd.read_excel(metadata_path, engine="openpyxl")
    if "Article Title" not in frame.columns:
        raise KeyError("The metadata table must contain an 'Article Title' column.")
    frame["Article Title"] = frame["Article Title"].fillna("").astype(str)
    titles = frame["Article Title"].tolist()
    json_dir.mkdir(parents=True, exist_ok=True)
    failed: List[str] = []
    records: List[Dict[str, Any]] = []

    for txt_path in sorted(txt_dir.glob("*.txt")):
        base_name = txt_path.stem
        match = frame[frame["Article Title"] == base_name]
        if match.empty:
            best = process.extractOne(base_name, titles, scorer=fuzz.ratio)
            if best is None or best[1] < fuzzy_threshold:
                failed.append(base_name)
                continue
            match = frame.iloc[[best[2]]]

        full_text = txt_path.read_text(encoding="utf-8")
        # Preserve heading boundaries for downstream reference removal.
        row = match.iloc[0]
        record = {field: safe_value(row, field) for field in METADATA_FIELDS}
        record["Full Text"] = full_text
        output_path = json_dir / f"{base_name}.json"
        output_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        records.append(record)

    failed_report.parent.mkdir(parents=True, exist_ok=True)
    failed_report.write_text("\n".join(failed) + ("\n" if failed else ""), encoding="utf-8")
    print(f"Metadata matching finished: {len(records)} matched, {len(failed)} unmatched.")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--txt-dir", required=True)
    parser.add_argument("--metadata", required=True, help="Excel metadata table.")
    parser.add_argument("--json-dir", required=True)
    parser.add_argument("--parquet-output", required=True)
    parser.add_argument("--failed-report", required=True)
    parser.add_argument("--fuzzy-threshold", type=int, default=95)
    args = parser.parse_args()
    if not 0 <= args.fuzzy_threshold <= 100:
        parser.error("--fuzzy-threshold must be between 0 and 100")

    convert_pdfs(Path(args.pdf_dir), Path(args.txt_dir))
    records = attach_metadata(
        Path(args.txt_dir),
        Path(args.metadata),
        Path(args.json_dir),
        Path(args.failed_report),
        args.fuzzy_threshold,
    )
    if records:
        output_path = Path(args.parquet_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(records)), output_path)
        print(f"Saved Parquet corpus to {output_path}.")


if __name__ == "__main__":
    main()
