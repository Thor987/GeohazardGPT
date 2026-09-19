"""Generate geohazard SFT samples from seed and semantically retrieved text."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from llm_client import OpenAICompatibleClient
except ImportError:  # pragma: no cover - supports package-style imports
    from .llm_client import OpenAICompatibleClient


QUESTION_TYPES = (
    "factoid_QA",
    "open-ended_question",
    "summary_question",
    "recommendation_question",
)

QUESTION_TYPE_NAMES = {
    "factoid_QA": "factoid QA",
    "open-ended_question": "open-ended question",
    "summary_question": "summary question",
    "recommendation_question": "recommendation question",
}




def parse_json_object(response: str) -> Dict[str, str]:
    """Parse and validate one JSON object returned by the teacher model."""
    cleaned = response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("The teacher response does not contain a JSON object.")
    value, _ = decoder.raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("The teacher response is not a JSON object.")

    required = ("instruction", "input", "output")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"The teacher response is missing fields: {missing}")
    if any(not isinstance(value[key], str) for key in required):
        raise ValueError("Required fields must be strings.")
    result = {key: value[key].strip() for key in required}
    if any(not result[key] for key in required):
        raise ValueError("The teacher response contains an empty required field.")
    return result


def read_json_records(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON file containing either one record or a list of records."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Unsupported JSON structure: {path}")


def add_record(
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]],
    record: Dict[str, Any],
    source: str,
    major_category: str,
    sub_category: str,
) -> None:
    """Add one source record to a category/subcategory group."""
    text = str(record.get("text", "")).strip()
    if not text:
        return
    groups[major_category][sub_category].append(
        {"text": text, "source": source, "original_data": record}
    )


def load_seed_directory(
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]],
    directory: Optional[Path],
    source: str,
) -> int:
    """Load expert or teacher-generated seed samples from category folders."""
    if directory is None or not directory.exists():
        return 0

    loaded = 0
    for category_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        for json_path in sorted(category_dir.glob("*.json")):
            try:
                records = read_json_records(json_path)
            except Exception as exc:
                print(f"Skipped {json_path}: {exc}")
                continue
            sub_category = json_path.stem
            for record in records:
                before = len(groups[category_dir.name][sub_category])
                add_record(groups, record, source, category_dir.name, sub_category)
                loaded += int(len(groups[category_dir.name][sub_category]) > before)
    return loaded


def nested_value(record: Dict[str, Any], keys: Iterable[str]) -> str:
    """Read a category field from a record or one of its nested metadata objects."""
    containers = [
        record,
        record.get("source_info", {}),
        record.get("source_data", {}),
        record.get("matched_annotation", {}),
    ]
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in keys:
            value = container.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return "unknown"


def load_retrieved_file(
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]],
    path: Optional[Path],
    min_rank: int,
) -> int:
    """Load semantic-search results while supporting both old and new schemas."""
    if path is None or not path.exists():
        return 0
    records = read_json_records(path)
    loaded = 0
    for record in records:
        try:
            rank = int(record.get("rank", 0))
        except (TypeError, ValueError):
            rank = 0
        if rank < min_rank:
            continue
        major = nested_value(record, ("major_category", "major categories", "major_category_name"))
        sub = nested_value(record, ("sub_category", "sub-categories", "sub_category_name"))
        before = len(groups[major][sub])
        add_record(groups, record, "retrieved", major, sub)
        loaded += int(len(groups[major][sub]) > before)
    return loaded


def build_prompt(question_type: str, major_category: str, sub_category: str, text: str) -> str:
    """Render the complete original notebook prompt verbatim."""
    if question_type == 'factoid_QA':
        prompt = f'You are a professional geological disaster expert. Generate a high-quality factoid QA based on the following geological disaster text about {major_category} - {sub_category}.\n\nExample:\n```json\n{{\n  "instruction": "You are a helpful geological disaster assistant. This is a factoid QA task. Provide a concise, itemized factual answer. Use only essential facts and measurements, avoid lengthy explanations.",\n    "input": "What are the main causes of coastal cliff erosion?",\n    "output": "The main causes are wave action (hydraulic action, abrasion), weathering (chemical and physical), groundwater seepage, gravity, and human activities."\n}}\n```\n\nContext Text: {text}\n\n**Question Type Guidelines:**\n\n- **question & answer**  \n  - Input: A short, factual question derived from the text (who, what, where, when, how much).  \n  - Output: A concise, itemized factual answer supported directly by the text. Provide only essential facts and measurements; avoid lengthy explanations.\n\n**Requirements:**\n1. The question should be scientifically accurate and professionally relevant\n2. The answer should be comprehensive and based on the provided text\n3. Use appropriate geological and disaster management terminology\n4. Ensure the difficulty level is suitable for geological professionals\n\n**Output Format:**\nGenerate exactly ONE factoid QA question-answer pair in the following JSON format:\n```json\n{{\n  "instruction": "You are a helpful geological disaster assistant. This is a factoid QA task. Provide a concise, itemized factual answer. Use only essential facts and measurements, avoid lengthy explanations.",\n  "input": "[Your generated question here]",\n  "output": "[Your generated comprehensive answer here]"\n}}\n```\n\nPlease ensure the JSON is valid and properly formatted.\n\n'
    elif question_type == 'open-ended_question':
        prompt = f'You are a professional geological disaster expert. Generate a high-quality open-ended_question based on the following geological disaster text about {major_category} - {sub_category}.\n\nExample:\n```json\n{{\n    "instruction": "You are a helpful geological disaster assistant. This is an open-ended question task. Please output a comprehensive explanation that considers multiple perspectives or consequences, not just facts.",\n    "input": "What are the environmental and social consequences of coastal cliff erosion?",\n    "output": "Environmental consequences:\n\n- Habitat loss and fragmentation: nesting sites, coastal wetlands and cliff-top vegetation are destroyed.\n- Reduced biodiversity and disruption of ecological networks (shorebirds, intertidal communities, dune plants).\n- Increased sediment load and turbidity in nearshore waters, affecting seagrass, coral and fisheries.\n- Changes to coastal geomorphology and sediment budgets (shoreline retreat, altered beaches and nearshore bars).\n- Water-quality impacts where eroded material contains contaminants.\n\nSocial and economic consequences:\n\n- Loss or damage to property, infrastructure (roads, utilities, seawalls, ports) and cultural sites located on cliff tops.\n- Economic impacts on tourism, recreation and local fisheries from beach loss and degraded coastal scenery.\n- Forced relocation or land-use restrictions for vulnerable communities; increased insurance and repair costs.\n- Safety hazards (sudden cliff collapses, increased coastal flooding) and stress on emergency services.\n- Legal and governance conflicts over shoreline management, compensation and coastal defence choices.\n\nOverall, cliff erosion creates cascading environmental damages and direct human costs — from ecosystem degradation to loss of land, livelihoods and safety — often concentrated in coastal communities that rely on the shoreline for economy and culture."\n}}\n```\n\nContext Text: {text}\n\n**Question Type Guidelines:**\n\n- **open-ended question**  \n  - Input: A broad, exploratory question that invites discussion (why, how, implications, comparisons).  \n  - Output: A comprehensive explanation that considers multiple perspectives or consequences, not just facts.  \n\n**Requirements:**\n1. The question should be scientifically accurate and professionally relevant\n2. The answer should be comprehensive and based on the provided text\n3. Use appropriate geological and disaster management terminology\n4. Ensure the difficulty level is suitable for geological professionals\n\n**Output Format:**\nGenerate exactly ONE open-ended question-answer pair in the following JSON format:\n```json\n{{\n  "instruction": "You are a helpful geological disaster assistant. This is an open-ended question task. Please output a comprehensive explanation that considers multiple perspectives or consequences, not just facts.",\n  "input": "[Your generated question here]",\n  "output": "[Your generated comprehensive answer here]"\n}}\n\nPlease ensure the JSON is valid and properly formatted.\n\n'
    elif question_type == 'summary_question':
        prompt = f'You are a professional geological disaster expert. Generate a high-quality summary_question based on the following geological disaster text about {major_category} - {sub_category}.\n\nExample:\n```json\n{{\n    "instruction": "You are a helpful geological disaster assistant. This is a summary question task. Provide a concise and professional summary of the context text, highlighting key findings, drivers, and implications.",\n    "input": "[The original Context Text here]",\n    "output": "Summary:\n[Generate a concise summary of the provided geological disaster text, highlighting main findings, trends, and key drivers without reframing it as a question.]"\n}}\n```\n\nContext Text: {text}\n\n**Question Type Guidelines:**\n\n- **summary question**  \n  - Input: A passage of geological disaster text (e.g., survey data, case description, model results).\n  - Output: A concise summary of the passage, highlighting the main findings, trends, and key drivers.\n  - Important: Do not reframe the passage as a question. Use the passage itself ([The original Context Text here]) as the input. The model should directly summarize the text provided. \n\n**Requirements:**\n1. The question should be scientifically accurate and professionally relevant\n2. The answer should be comprehensive and based on the provided text\n3. Use appropriate geological and disaster management terminology\n4. Ensure the difficulty level is suitable for geological professionals\n\n**Output Format:**\nGenerate exactly ONE summary question-answer pair in the following JSON format:\n```json\n{{\n  "instruction": "You are a helpful geological disaster assistant. This is a summary_question task. Provide a concise and professional summary of the context text, highlighting key findings, drivers, and implications.",\n  "input": "[The original Context Text here]",\n  "output": "[Your generated comprehensive answer here]"\n}}\n\nPlease ensure the JSON is valid and properly formatted.\n\n'
    elif question_type == 'recommendation_question':
        prompt = f'You are a professional geological disaster expert. Generate a high-quality recommendation_question based on the following geological disaster text about {major_category} - {sub_category}.\n\nExample:\n```json\n{{\n  "instruction": "You are a helpful geological disaster assistant. This is a recommendation question task. Provide practical engineering or management measures to mitigate or control the described geological disaster.",\n  "input": "What engineering measures can be applied to mitigate coastal cliff erosion?",\n  "output": "Recommended measures:\n\n- Hard engineering: seawalls, revetments, groynes to reduce direct wave attack.\n- Soft engineering: beach nourishment, dune stabilization, re-vegetation to enhance natural buffers.\n- Drainage and groundwater control: surface drains, sub-surface drains to reduce pore-water pressure.\n- Monitoring and early warning systems: LiDAR surveys, crack meters, slope stability sensors.\n- Managed retreat in high-risk areas where engineering protection is unsustainable.\n\nOverall, a combination of hard and soft measures, adapted to local geological and socioeconomic conditions, is essential for effective long-term risk reduction."\n}}\n```\n\nContext Text: {text}\n\n**Question Type Guidelines:**\n\n- **recommendation question**  \n  - Input: A question about what mitigation, adaptation, or engineering measures can be applied.  \n  - Output: A list of feasible, professional recommendations, grounded in geological disaster management practice.\n\n**Requirements:**\n1. The question should be scientifically accurate and professionally relevant\n2. The answer should be comprehensive and based on the provided text\n3. Use appropriate geological and disaster management terminology\n4. Ensure the difficulty level is suitable for geological professionals\n\n**Output Format:**\nGenerate exactly ONE reasoning question-answer pair in the following JSON format:\n```json\n{{\n  "instruction": "You are a helpful geological disaster assistant. This is a recommendation question task. Provide practical engineering or management measures to mitigate or control the described geological disaster.",\n  "input": "[Your generated question here]",\n  "output": "[Your generated list of recommendations here]"\n}}\n\n\nPlease ensure the JSON is valid and properly formatted.\n\n'
    else:
        prompt = f'Error: unknown question type {question_type}'
    return prompt


def safe_name(value: str) -> str:
    """Make a category name safe for a checkpoint filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


class SFTGenerator:
    """Generate samples with deterministic shuffling and resumable checkpoints."""

    def __init__(
        self,
        *,
        output_dir: Path,
        checkpoint_dir: Path,
        client: OpenAICompatibleClient,
        seed: int,
        max_input_chars: int,
        checkpoint_interval: int,
        max_retries: int,
        debug: bool,
    ) -> None:
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.client = client
        self.seed = seed
        self.max_input_chars = max_input_chars
        self.checkpoint_interval = checkpoint_interval
        self.max_retries = max_retries
        self.debug = debug
        self.stats = defaultdict(int)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, major: str, sub: str) -> Path:
        identity = hashlib.sha256(json.dumps([major, sub]).encode()).hexdigest()[:16]
        return self.checkpoint_dir / f"{safe_name(major)}__{safe_name(sub)}_{identity}.json"

    def load_checkpoint(self, major: str, sub: str) -> Dict[str, Any]:
        path = self.checkpoint_path(major, sub)
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            except Exception as exc:
                raise ValueError(f"Invalid checkpoint {path}; refusing to restart silently.") from exc
        return {"processed_count": 0, "questions_by_type": {kind: [] for kind in QUESTION_TYPES}}

    def save_checkpoint(
        self,
        major: str,
        sub: str,
        processed_count: int,
        questions_by_type: Dict[str, List[Dict[str, str]]],
    ) -> None:
        payload = {
            "processed_count": processed_count,
            "questions_by_type": questions_by_type,
            "fingerprint": self.fingerprint,
        }
        destination = self.checkpoint_path(major, sub)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(destination)

    def write_results(
        self,
        major: str,
        sub: str,
        questions_by_type: Dict[str, List[Dict[str, str]]],
    ) -> None:
        output_dir = self.output_dir / safe_name(major) / safe_name(sub)
        output_dir.mkdir(parents=True, exist_ok=True)
        for question_type, records in questions_by_type.items():
            if records:
                path = output_dir / f"{question_type}_generated.json"
                path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    def generate_one(self, major: str, sub: str, question_type: str, text: str) -> Dict[str, str]:
        if len(text) > self.max_input_chars:
            text = text[: self.max_input_chars] + "..."
        prompt = build_prompt(question_type, major, sub, text)
        if self.debug:
            print(f"\n--- Prompt: {major}/{sub}/{question_type} ---\n{prompt}\n--- End prompt ---")

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.generate(prompt, temperature=0.7, max_tokens=2048)
                result = parse_json_object(response)
                self.stats["successful_generations"] += 1
                return result
            except Exception as exc:
                print(f"Generation attempt {attempt}/{self.max_retries} failed: {exc}")
                if attempt < self.max_retries:
                    time.sleep(min(2 * attempt, 10))
        self.stats["failed_generations"] += 1
        raise RuntimeError(f"Generation failed after {self.max_retries} attempts.")

    def process_group(self, major: str, sub: str, items: List[Dict[str, Any]]) -> None:
        """Generate all samples for one category/subcategory group."""
        items = list(items)
        random.Random(self.seed).shuffle(items)
        self.fingerprint = hashlib.sha256(json.dumps(
            [items, self.seed, self.max_input_chars, getattr(self.client, "model", ""),
             [build_prompt(kind, "MAJOR", "SUB", "CONTEXT") for kind in QUESTION_TYPES]], sort_keys=True, ensure_ascii=False
        ).encode()).hexdigest()
        checkpoint = self.load_checkpoint(major, sub)
        if checkpoint.get("processed_count", 0) and checkpoint.get("fingerprint") != self.fingerprint:
            raise ValueError("Checkpoint inputs or settings changed; use a new checkpoint directory.")
        processed = int(checkpoint.get("processed_count", 0))
        questions_by_type = {
            kind: list(checkpoint.get("questions_by_type", {}).get(kind, []))
            for kind in QUESTION_TYPES
        }

        if processed >= len(items):
            self.write_results(major, sub, questions_by_type)
            return

        for index in range(processed, len(items)):
            item = items[index]
            text = str(item.get("text", "")).strip()
            if not text:
                processed += 1
                continue
            selector = random.Random(self.seed + index)
            question_type = selector.choice(QUESTION_TYPES)
            try:
                result = self.generate_one(major, sub, question_type, text)
            except Exception:
                self.save_checkpoint(major, sub, processed, questions_by_type)
                raise
            questions_by_type[question_type].append(result)
            processed += 1
            self.stats[f"{question_type}_generated"] += 1
            if processed % self.checkpoint_interval == 0:
                self.save_checkpoint(major, sub, processed, questions_by_type)

        self.write_results(major, sub, questions_by_type)
        self.save_checkpoint(major, sub, processed, questions_by_type)

    def run(self, groups: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
        """Generate samples for every loaded group."""
        started = time.time()
        total_groups = sum(len(subcategories) for subcategories in groups.values())
        completed = 0
        for major in sorted(groups):
            for sub in sorted(groups[major]):
                items = groups[major][sub]
                print(f"Processing {major}/{sub}: {len(items)} source records.")
                self.process_group(major, sub, items)
                completed += 1
                print(f"Completed {completed}/{total_groups} groups.")
        print(
            f"Generated {sum(self.stats[f'{kind}_generated'] for kind in QUESTION_TYPES)} samples "
            f"in {(time.time() - started) / 60:.1f} minutes."
        )
        print(f"Successful calls: {self.stats['successful_generations']}; failed calls: {self.stats['failed_generations']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True, help="Directory containing the seed inputs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated SFT JSON files.")
    parser.add_argument("--checkpoint-dir", type=Path, help="Checkpoint directory; defaults to output-dir/checkpoints.")
    parser.add_argument("--expert-dir", type=Path, help="Expert seed directory; defaults to base-dir/annotation_data_expert.")
    parser.add_argument("--llm-dir", type=Path, help="Teacher-generated seed directory; defaults to base-dir/annotation_data_llm.")
    parser.add_argument("--retrieved-file", type=Path, help="Semantic-search result JSON; defaults to base-dir/top_semantic_matches.json.")
    parser.add_argument("--min-retrieval-rank", type=int, default=3)
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument("--base-url")
    parser.add_argument("--max-input-chars", type=int, default=3000)
    parser.add_argument("--checkpoint-interval", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.min_retrieval_rank < 0 or args.max_input_chars < 1 or args.checkpoint_interval < 1 or args.max_retries < 1:
        parser.error("rank, input length, checkpoint interval, and retry count are invalid")
    return args


def main() -> None:
    args = parse_args()
    expert_dir = args.expert_dir or args.base_dir / "annotation_data_expert"
    llm_dir = args.llm_dir or args.base_dir / "annotation_data_llm"
    retrieved_file = args.retrieved_file or args.base_dir / "top_semantic_matches.json"
    checkpoint_dir = args.checkpoint_dir or args.output_dir / "checkpoints"

    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    counts = {
        "expert": load_seed_directory(groups, expert_dir, "expert"),
        "llm": load_seed_directory(groups, llm_dir, "llm"),
        "retrieved": load_retrieved_file(groups, retrieved_file, args.min_retrieval_rank),
    }
    print(f"Loaded records: {counts}")
    if not groups:
        raise SystemExit("No input records were found.")

    client = OpenAICompatibleClient(model=args.model, base_url=args.base_url)
    generator = SFTGenerator(
        output_dir=args.output_dir,
        checkpoint_dir=checkpoint_dir,
        client=client,
        seed=args.seed,
        max_input_chars=args.max_input_chars,
        checkpoint_interval=args.checkpoint_interval,
        max_retries=args.max_retries,
        debug=args.debug,
    )
    generator.run(groups)


if __name__ == "__main__":
    main()
