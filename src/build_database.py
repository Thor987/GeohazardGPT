"""Build a persistent Chroma index for clause-level engineering standards.

The builder uses Qwen3-Embedding with last-token pooling. Documents are split
before encoding so that an overlong section is never silently truncated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


MAX_EMBEDDING_TOKENS = 4096


class Qwen3Embedder:
    """Encode documents and queries with a Qwen3 embedding checkpoint."""

    def __init__(self, model_path: str, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_name = device
        dtype = torch.float16 if device != "cpu" else torch.float32

        print(f"Loading Qwen3-Embedding from '{model_path}' on {device}.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs).eval()
        if device == "cpu":
            self.model.to("cpu")
        print("Embedding model loaded.")

    @staticmethod
    def _last_token_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Use the hidden state of the last non-padding token."""
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def token_lengths(self, texts: List[str]) -> List[int]:
        encoded = self.tokenizer(
            texts,
            truncation=False,
            add_special_tokens=True,
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def encode(
        self,
        texts: str | List[str],
        *,
        normalize_embeddings: bool = True,
        is_query: bool = False,
        batch_size: int = 4,
        task: str = (
            "Retrieve engineering standard provisions, formulas, and "
            "parameter tables relevant to the question."
        ),
    ):
        """Encode one or more texts without silent truncation."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("At least one text is required.")

        prepared = texts
        if is_query:
            prepared = [f"Instruct: {task}\nQuery: {text}" for text in texts]

        all_embeddings = []
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            lengths = self.token_lengths(batch)
            if max(lengths) > MAX_EMBEDDING_TOKENS:
                raise ValueError(
                    "An embedding input exceeds 4096 tokens. Split the source "
                    "section before encoding; silent truncation is disabled. "
                    f"Maximum length in batch: {max(lengths)}."
                )

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=False,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.inference_mode():
                outputs = self.model(**encoded)

            embeddings = self._last_token_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )
            if normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().float())

        return torch.cat(all_embeddings, dim=0).numpy()


def load_and_chunk_documents(json_dir: str) -> List[Dict[str, Any]]:
    """Load JSON standards and create one record per source section."""
    root = Path(json_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Standards directory does not exist: {root}")

    documents: List[Dict[str, Any]] = []
    id_counts: Dict[str, int] = {}
    print(f"Loading standards from {root}.")

    for path in sorted(root.glob("*.json")):
        match = re.match(r"^(\d+)", path.name)
        file_id = match.group(1) if match else path.stem

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Could not parse {path}") from exc

        title = str(data.get("title", ""))
        sections = data.get("sections", [])
        if not isinstance(sections, list):
            raise ValueError(f"Expected a list of sections in {path}")

        for section in sections:
            section_id = str(section.get("id", "")).strip()
            content = str(section.get("content", "")).strip()
            if not section_id or not content:
                continue

            base_id = f"{file_id}.{section_id}"
            occurrence = id_counts.get(base_id, 0)
            chunk_id = base_id if occurrence == 0 else f"{base_id}.{occurrence}"
            id_counts[base_id] = occurrence + 1
            documents.append(
                {
                    "content": content,
                    "chunk_id": chunk_id,
                    "metadata": {
                        "source_file": path.name,
                        "title": title,
                        "original_section_id": section_id,
                    },
                }
            )

    print(f"Loaded {len(documents)} source sections.")
    return documents


def split_long_sections(
    documents: List[Dict[str, Any]],
    tokenizer,
    *,
    max_tokens: int = 3500,
    overlap_chars: int = 200,
) -> List[Dict[str, Any]]:
    """Split long sections while preserving source and parent-section metadata."""
    if not 1 <= max_tokens <= MAX_EMBEDDING_TOKENS:
        raise ValueError("max_tokens must be between 1 and 4096")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative")

    def count(text: str) -> int:
        return len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])

    output: List[Dict[str, Any]] = []
    for document in documents:
        text = document["content"]
        if count(text) <= max_tokens:
            output.append(document)
            continue

        start = 0
        part_index = 0
        while start < len(text):
            low, high = start + 1, len(text)
            end = start
            while low <= high:
                middle = (low + high) // 2
                if count(text[start:middle]) <= max_tokens:
                    end = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if end == start:
                raise ValueError(f"Could not split source section {document['chunk_id']}.")

            if end < len(text):
                newline = text.rfind("\n", start + (end - start) // 2, end)
                if newline >= 0 and count(text[start : newline + 1]) <= max_tokens:
                    end = newline + 1

            part_index += 1
            output.append(
                {
                    "content": text[start:end],
                    "chunk_id": f"{document['chunk_id']}::part{part_index}",
                    "metadata": {
                        **document["metadata"],
                        "parent_chunk_id": document["chunk_id"],
                        "part_index": part_index,
                        "char_start": start,
                        "char_end": end,
                    },
                }
            )
            if end == len(text):
                break
            start = max(start + 1, end - min(overlap_chars, (end - start) // 2))

        print(
            f"Split {document['chunk_id']} ({count(text)} tokens) into "
            f"{part_index} chunks."
        )

    ids = [item["chunk_id"] for item in output]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs were produced.")
    return output


def build_vector_database(
    documents: List[Dict[str, Any]],
    *,
    embedding_model_path: str,
    db_path: str,
    collection_name: str,
    batch_size: int = 4,
    chunk_tokens: int = 3500,
    device: str = "auto",
) -> None:
    """Encode source sections and store them in a persistent Chroma collection."""
    embedder = Qwen3Embedder(embedding_model_path, device=device)
    documents = split_long_sections(
        documents,
        embedder.tokenizer,
        max_tokens=chunk_tokens,
    )

    print(f"Creating Chroma collection '{collection_name}' at '{db_path}'.")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count():
        raise ValueError(
            "The target collection is not empty. Use a new database path or "
            "collection name to avoid mixing corpora."
        )

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        embeddings = embedder.encode(
            [item["content"] for item in batch],
            batch_size=batch_size,
        ).tolist()
        collection.add(
            ids=[item["chunk_id"] for item in batch],
            embeddings=embeddings,
            documents=[item["content"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )
        print(f"Indexed {min(start + batch_size, len(documents))}/{len(documents)} sections.")

    print(f"Finished. Collection size: {collection.count()}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing standard JSON files.")
    parser.add_argument("--embedding-model", required=True, help="Qwen3-Embedding checkpoint path or model ID.")
    parser.add_argument("--db-path", required=True, help="Persistent Chroma directory.")
    parser.add_argument("--collection", default="engineering_specs_qwen3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-tokens", type=int, default=3500)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 1 <= args.chunk_tokens <= MAX_EMBEDDING_TOKENS:
        parser.error("--chunk-tokens must be between 1 and 4096")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    return args


def main() -> None:
    args = parse_args()
    documents = load_and_chunk_documents(args.data_dir)
    if not documents:
        raise SystemExit("No source sections were found.")
    build_vector_database(
        documents,
        embedding_model_path=args.embedding_model,
        db_path=args.db_path,
        collection_name=args.collection,
        batch_size=args.batch_size,
        chunk_tokens=args.chunk_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    main()
