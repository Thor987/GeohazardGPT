"""Merge processed literature and web-corpus records into JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--literature", required=True, help="Processed literature Parquet file.")
    parser.add_argument("--web", required=True, help="Filtered web-corpus JSONL file.")
    parser.add_argument("--output", required=True, help="Merged JSONL output path.")
    args = parser.parse_args()

    literature = pd.read_parquet(args.literature)
    web = pd.read_json(args.web, lines=True)
    if "text" not in literature.columns:
        column = next((name for name in ("Full Text1", "Full Text", "Abstract1", "Abstract") if name in literature.columns), None)
        if column is None:
            raise ValueError("Literature has no supported text column.")
        literature = literature.rename(columns={column: "text"})
    if "text" not in web.columns:
        raise ValueError("Web corpus must contain a text field.")
    literature["source"] = "literature"
    web["source"] = "web"
    merged = pd.concat([literature, web], ignore_index=True)
    merged = merged[merged["text"].map(lambda value: isinstance(value, str) and bool(value.strip()))]
    merged = merged[["text", "source"]]

    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError("Output already exists; choose a new output file.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in merged.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(merged)} records to {output_path}.")


if __name__ == "__main__":
    main()
