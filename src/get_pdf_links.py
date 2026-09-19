"""Retrieve legally available open-access PDF links from DOI metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests


def query_unpaywall(doi: str, email: str, timeout: int = 20) -> Dict[str, Any]:
    """Query the Unpaywall API for one DOI."""
    url = f"https://api.unpaywall.org/v2/{doi}"
    response = requests.get(
        url,
        params={"email": email},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Excel file containing a DOI column.")
    parser.add_argument("--email", required=True, help="Contact email required by Unpaywall.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--doi-column", default="DOI")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_excel(metadata_path, engine="openpyxl")
    if args.doi_column not in frame.columns:
        raise KeyError(f"Missing DOI column '{args.doi_column}' in {metadata_path}.")

    links: List[str] = []
    success_dois: List[str] = []
    failed_dois: List[str] = []
    titles: List[str] = []

    for raw_doi in frame[args.doi_column].dropna().unique():
        doi = str(raw_doi).strip()
        if not doi:
            continue
        try:
            payload = query_unpaywall(doi, args.email)
            best_oa = payload.get("best_oa_location") or {}
            pdf_url = best_oa.get("url_for_pdf")
            title = payload.get("title") or ""
            if not pdf_url:
                failed_dois.append(doi)
                continue
            links.append(str(pdf_url))
            success_dois.append(doi)
            titles.append(str(title))
            print(f"Found open-access PDF: {doi}")
        except (requests.RequestException, ValueError) as exc:
            failed_dois.append(doi)
            print(f"Failed to query {doi}: {exc}")

    (output_dir / "all_pdf_links.txt").write_text("\n".join(links) + "\n", encoding="utf-8")
    (output_dir / "success_pdfs.txt").write_text("\n".join(success_dois) + "\n", encoding="utf-8")
    (output_dir / "failed_dois.txt").write_text("\n".join(failed_dois) + "\n", encoding="utf-8")
    (output_dir / "titles.txt").write_text("\n".join(titles) + "\n", encoding="utf-8")
    print(f"Saved {len(links)} links to {output_dir}.")


if __name__ == "__main__":
    main()
