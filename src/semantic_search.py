"""Filter a large text corpus with a FAISS semantic-search index.

The script indexes a reviewed seed corpus and retrieves the most similar
segments from the processed geohazard corpus. It is the semantic-filtering
stage of GeoInstruct.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def load_json_samples(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load JSON samples grouped by category and file name."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    if not root.exists():
        return grouped
    for category in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(category.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, list):
                value = [value]
            grouped[f"{category.name}_{path.stem}"] = value
    return grouped


class SemanticSearcher:
    """Build and query an inner-product FAISS index over a seed corpus."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        LOGGER.info("Loading sentence-embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.corpus_info: List[Dict[str, Any]] = []

    def load_annotation_data(
        self,
        expert_dir: str,
        llm_dir: str,
        *,
        max_samples_per_group: int = 205,
    ) -> List[Dict[str, Any]]:
        """Merge expert and teacher-generated seed records by category."""
        expert = load_json_samples(Path(expert_dir))
        generated = load_json_samples(Path(llm_dir))
        if not expert and not generated:
            raise FileNotFoundError(
                "Neither the expert seed directory nor the generated seed "
                "directory contains JSON files."
            )

        all_items: List[Dict[str, Any]] = []
        for key in sorted(set(expert) | set(generated)):
            combined = expert.get(key, []) + generated.get(key, [])
            all_items.extend(combined[:max_samples_per_group])
        LOGGER.info("Loaded %d seed segments.", len(all_items))
        return all_items

    def build_index(self, annotation_items: List[Dict[str, Any]]) -> None:
        """Encode seed segments and build a cosine-similarity index."""
        texts = [str(item.get("text", "")).strip() for item in annotation_items]
        if not texts or any(not text for text in texts):
            raise ValueError("Every seed record must contain a non-empty 'text' field.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.corpus_info = [
            {
                "text": text,
                "major_category": item.get("major categories", "Unknown"),
                "sub_category": str(item.get("sub-categories", "Unknown")).strip(),
            }
            for item, text in zip(annotation_items, texts)
        ]
        LOGGER.info("Built FAISS index with %d seed vectors.", self.index.ntotal)

    def search(
        self,
        merged_data_path: str,
        *,
        top_k: int,
        query_batch_size: int = 64,
    ) -> List[Dict[str, Any]]:
        """Retrieve and rank corpus segments by their best seed similarity."""
        if self.index is None:
            raise RuntimeError("Build the FAISS index before searching.")

        query_items: List[Dict[str, Any]] = []
        with Path(merged_data_path).open("r", encoding="utf-8") as handle:
            for line in tqdm(handle, desc="Loading corpus records"):
                if line.strip():
                    query_items.append(json.loads(line))
        query_items = [item for item in query_items if isinstance(item.get("text"), str) and item["text"].strip()]
        query_texts = [item["text"].strip() for item in query_items]
        if not query_texts:
            raise ValueError("The query corpus contains no non-empty 'text' fields.")

        embeddings = self.model.encode(
            query_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=query_batch_size,
            show_progress_bar=True,
        ).astype("float32")
        distances, indices = self.index.search(embeddings, 1)

        results = []
        for index, (text, item) in enumerate(zip(query_texts, query_items)):
            results.append(
                {
                    "score": float(distances[index, 0]),
                    "text": text,
                    "source_data": item,
                    "best_corpus_idx": int(indices[index, 0]),
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def save_index(self, index_path: str, info_path: str) -> None:
        """Save the FAISS index and source metadata."""
        if self.index is None:
            raise RuntimeError("No FAISS index is available.")
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(info_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with Path(info_path).open("wb") as handle:
            pickle.dump(self.corpus_info, handle)

    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """Save ranked matches together with their best seed annotation."""
        formatted = []
        for rank, result in enumerate(results, start=1):
            item = {
                "rank": rank,
                "score": result["score"],
                "text": result["text"],
                "source_info": result["source_data"],
            }
            matched_index = result["best_corpus_idx"]
            if 0 <= matched_index < len(self.corpus_info):
                item["matched_annotation"] = self.corpus_info[matched_index]
            formatted.append(item)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(formatted, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-dir", required=True)
    parser.add_argument("--llm-dir", required=True)
    parser.add_argument("--merged-data", required=True)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--info-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=140180)
    parser.add_argument("--max-samples-per-group", type=int, default=205)
    parser.add_argument("--query-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.top_k < 1 or args.max_samples_per_group < 1 or args.query_batch_size < 1:
        parser.error("top-k, max-samples-per-group, and query-batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    start = time.time()
    searcher = SemanticSearcher(args.model)
    annotations = searcher.load_annotation_data(
        args.expert_dir,
        args.llm_dir,
        max_samples_per_group=args.max_samples_per_group,
    )
    searcher.build_index(annotations)
    searcher.save_index(args.index_path, args.info_path)
    results = searcher.search(
        args.merged_data,
        top_k=args.top_k,
        query_batch_size=args.query_batch_size,
    )
    searcher.save_results(results, args.output)
    LOGGER.info("Saved %d matches to %s.", len(results), args.output)
    LOGGER.info("Finished in %.2f minutes.", (time.time() - start) / 60)


if __name__ == "__main__":
    main()
