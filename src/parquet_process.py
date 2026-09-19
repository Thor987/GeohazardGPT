"""Dataverse/Spark ETL functions used to clean and chunk article text.

Importing this module registers the custom ETL functions but does not execute
the pipeline. Run the file directly with explicit input and output paths.
"""

from __future__ import annotations

import argparse
import re
from typing import Any, List

from omegaconf import OmegaConf
from pyspark.sql.types import ArrayType, StringType

from dataverse.etl import register_etl


@register_etl
def custom___text___remove_abstract_and_references(
    spark,
    data,
    subset: str = "Full Text",
    *args: Any,
    **kwargs: Any,
):
    """Remove material before the abstract and after the references section."""
    from pyspark.sql import DataFrame
    import pyspark.sql.functions as F

    if not isinstance(data, DataFrame):
        data = spark.createDataFrame(data)

    abstract_pattern = re.compile(r"^Abstract|^ABSTRACT", re.MULTILINE)
    conclusion_pattern = re.compile(
        r"^Conclusions\b|^Summary\b",
        re.MULTILINE | re.IGNORECASE,
    )
    reference_pattern = re.compile(
        r"^References?\b|^Bibliography\b|^Acknowledgements?\b|"
        r"^Data availability\b|^Supplemental material\b",
        re.MULTILINE | re.IGNORECASE,
    )

    def process_text(text: str | None) -> str | None:
        if not text:
            return text
        abstract_matches = list(abstract_pattern.finditer(text))
        start = min((match.start() for match in abstract_matches), default=0)

        conclusion_matches = list(conclusion_pattern.finditer(text))
        start_point = 0
        if conclusion_matches:
            last = max(conclusion_matches, key=lambda match: match.start())
            if last.start() >= len(text) * 0.6:
                start_point = last.end()

        reference_matches = [
            match
            for match in reference_pattern.finditer(text)
            if match.start() > start_point and match.start() > len(text) * 0.67
        ]
        end = min((match.start() for match in reference_matches), default=len(text))
        return text[start:end]

    return data.withColumn(
        subset,
        F.udf(process_text, returnType=StringType())(F.col(subset)),
    )


@register_etl
def custom___text___split_by_sentence_block(
    spark,
    data,
    subset: str = "Full Text",
    min_words: int = 10,
    *args: Any,
    **kwargs: Any,
):
    """Split text into sentence blocks containing at least ``min_words`` words."""
    from pyspark.sql import DataFrame
    import pyspark.sql.functions as F

    if not isinstance(data, DataFrame):
        data = spark.createDataFrame(data)

    def split_text(text: str | None) -> List[str]:
        if not text:
            return []

        protected = text
        protected = re.sub(
            r"(https?://[^\s]+|doi\.org/[^\s]+)",
            lambda match: match.group(1).replace(".", "[DOT]"),
            protected,
        )
        protected = re.sub(
            r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
            lambda match: match.group(1).replace(".", "[DOT]"),
            protected,
        )
        protected = re.sub(r"(\d)\.(\d)", r"\1[DOT]\2", protected)
        abbreviations = [
            "e.g",
            "i.e",
            "S.A",
            "U.S.A",
            "Dr",
            "Mr",
            "Ms",
            "Prof",
            "Inc",
            "Ltd",
            "Jr",
            "Sr",
            "vs",
            "Fig",
            "No",
            "Vol",
            "pp",
            "et al",
            "etc",
        ]
        for abbreviation in abbreviations:
            protected = re.sub(
                rf"\b{re.escape(abbreviation)}\.",
                f"{abbreviation}[DOT]",
                protected,
                flags=re.IGNORECASE,
            )
        protected = re.sub(
            r"(\[\d+\]\.|\(\d+\)\.|^\d+\.)",
            lambda match: match.group(1).replace(".", "[DOT]"),
            protected,
        )
        sentences = re.split(r"(?<=[.!?…])\s+(?=[A-Z])", protected)
        sentences = [sentence.replace("[DOT]", ".").strip() for sentence in sentences]
        sentences = [sentence for sentence in sentences if sentence]

        blocks: List[str] = []
        block: List[str] = []
        word_count = 0
        for sentence in sentences:
            block.append(sentence)
            word_count += len(sentence.split())
            if word_count >= min_words:
                candidate = " ".join(block).strip()
                if candidate and re.match(r"^[A-Za-z]", candidate):
                    blocks.append(candidate)
                block = []
                word_count = 0
        if block:
            candidate = " ".join(block).strip()
            if candidate and re.match(r"^[A-Za-z]", candidate):
                blocks.append(candidate)
        return blocks

    split_udf = F.udf(split_text, returnType=ArrayType(StringType()))
    return (
        data.withColumn("blocks", split_udf(F.col(subset)))
        .withColumn(subset, F.explode("blocks"))
        .drop("blocks")
    )


@register_etl
def custom___text___filter_en(
    spark,
    data,
    subset: str = "Full Text",
    model_path: str | None = None,
    threshold: float = 0.65,
    *args: Any,
    **kwargs: Any,
):
    """Keep sentences classified as English by a fastText language model."""
    from pyspark.sql import DataFrame
    import pyspark.sql.functions as F

    if not model_path:
        raise ValueError("model_path is required when the English filter is enabled")
    if not isinstance(data, DataFrame):
        data = spark.createDataFrame(data)

    model_cache = {"value": None}

    def filter_text(text: str | None) -> List[str]:
        if not text:
            return []
        if model_cache["value"] is None:
            import fasttext

            model_cache["value"] = fasttext.load_model(model_path)
        model = model_cache["value"]
        selected = []
        for sentence in (part.strip() for part in text.split(".") if part.strip()):
            labels, probabilities = model.predict(" ".join(sentence.split()))
            if labels[0] == "__label__en" and probabilities[0] >= threshold:
                selected.append(sentence)
        return selected

    return data.withColumn(
        subset,
        F.concat_ws(". ", F.udf(filter_text, returnType=ArrayType(StringType()))(F.col(subset))),
    ).filter(F.length(F.trim(F.col(subset))) > 0)


def build_config(input_path: str, output_path: str, use_language_filter: bool = False, fasttext_model: str | None = None):
    """Create a Dataverse ETL configuration for the public command-line entry point."""
    transforms = [
        {
            "name": "deduplication___minhash___lsh_jaccard",
            "args": {"threshold": 0.75, "ngram_size": 5, "subset": "Full Text"},
        },
        {"name": "cleaning___char___remove_accent", "args": {"subset": "Full Text"}},
        {"name": "cleaning___char___normalize_whitespace", "args": {"subset": "Full Text"}},
        {"name": "cleaning___char___remove_unprintable", "args": {"subset": "Full Text"}},
        {
            "name": "custom___text___remove_abstract_and_references",
            "args": {"subset": "Full Text"},
        },
        {
            "name": "cleaning___document___split_by_word",
            "args": {"subset": "Full Text", "word_per_chunk": 300},
        },
    ]
    if use_language_filter:
        if not fasttext_model:
            raise ValueError("fasttext_model is required when language filtering is enabled")
        transforms.insert(
            4,
            {
                "name": "custom___text___filter_en",
                "args": {
                    "subset": "Full Text",
                    "model_path": fasttext_model,
                    "threshold": 0.65,
                },
            },
        )
    transforms.extend(
        [
            {
                "name": "data_save___parquet___ufl2parquet",
                "args": {"save_path": output_path},
            }
        ]
    )
    return OmegaConf.create(
        {
            "spark": {"appname": "GeohazardGPT-ETL", "driver": {"memory": "32g"}},
            "etl": [
                {
                    "name": "data_ingestion___parquet___pq2raw",
                    "args": {"path": [input_path]},
                },
                *transforms,
            ],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input Parquet path.")
    parser.add_argument("--output", required=True, help="Output Parquet path.")
    parser.add_argument("--language-model", help="fastText language-ID model path.")
    args = parser.parse_args()
    from dataverse.etl import ETLPipeline

    config = build_config(
        args.input,
        args.output,
        use_language_filter=bool(args.language_model),
        fasttext_model=args.language_model,
    )
    ETLPipeline().run(config=config, verbose=True)


if __name__ == "__main__":
    main()
