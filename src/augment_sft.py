"""Augment geohazard instruction samples using existing examples.

The input directory is expected to contain one directory per major hazard
category. Each category directory contains JSON files for its subcategories,
with records containing ``instruction``, ``input``, and ``output`` fields.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

try:
    from llm_client import OpenAICompatibleClient
except ImportError:  # pragma: no cover - supports package-style imports
    from .llm_client import OpenAICompatibleClient


QUESTION_TYPES = [
    "question & answer",
    "open-ended question",
    "summary question",
    "reasoning question",
]


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object even when the model surrounds it with a code fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Teacher output is not a JSON object.")
    required = {"instruction", "input", "output"}
    if not required.issubset(value):
        raise ValueError(f"Teacher output is missing fields: {sorted(required - set(value))}")
    if any(not isinstance(value[key], str) or not value[key].strip() for key in required):
        raise ValueError("Teacher fields must be non-empty strings.")
    return {key: value[key].strip() for key in required}


class SampleAmplifier:
    """Expand expert-written samples with a user-supplied teacher model."""

    def __init__(
        self,
        *,
        data_dir: str,
        output_dir: str,
        client: OpenAICompatibleClient,
        samples_per_type: int = 10,
        generated_example_threshold: int = 3,
        rng: random.Random | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.client = client
        self.samples_per_type = samples_per_type
        self.generated_example_threshold = generated_example_threshold
        self.rng = rng or random.Random()

    def load_existing_samples(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Load category and subcategory samples from JSON files."""
        all_samples: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for category_dir in sorted(path for path in self.data_dir.iterdir() if path.is_dir()):
            category: Dict[str, List[Dict[str, Any]]] = {}
            for path in sorted(category_dir.glob("*.json")):
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, list):
                    value = [value]
                category[path.stem.replace("_en_en", "")] = value
            if category:
                all_samples[category_dir.name] = category
        return all_samples

    @staticmethod
    def by_question_type(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {question_type: [] for question_type in QUESTION_TYPES}
        for sample in samples:
            instruction = str(sample.get("instruction", ""))
            for question_type in QUESTION_TYPES:
                if question_type in instruction:
                    grouped[question_type].append(sample)
                    break
        return grouped

    def generated_path(self, category: str, subcategory: str, question_type: str) -> Path:
        return (
            self.output_dir
            / category
            / subcategory
            / f"{question_type.replace(' ', '_')}_generated.json"
        )

    def generated_samples(self, category: str, subcategory: str, question_type: str) -> List[Dict[str, Any]]:
        path = self.generated_path(category, subcategory, question_type)
        if not path.exists():
            return []
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []

    def select_examples(
        self,
        all_samples: Dict[str, Dict[str, List[Dict[str, Any]]]],
        category: str,
        subcategory: str,
        question_type: str,
    ) -> List[Dict[str, Any]]:
        """Select one local, one cross-category, and optionally one generated example."""
        examples: List[Dict[str, Any]] = []
        local = self.by_question_type(all_samples[category][subcategory])[question_type]
        if local:
            examples.append(self.rng.choice(local))

        other: List[Dict[str, Any]] = []
        for other_category, subcategories in all_samples.items():
            for other_subcategory, samples in subcategories.items():
                if (other_category, other_subcategory) != (category, subcategory):
                    other.extend(self.by_question_type(samples)[question_type])
        if other:
            examples.append(self.rng.choice(other))

        generated = self.generated_samples(category, subcategory, question_type)
        if len(generated) >= self.generated_example_threshold:
            examples.append(self.rng.choice(generated))
        return examples

    @staticmethod
    def create_prompt(disaster_type, sub_disaster_type, question_type, examples):
        """Render the original example-conditioned prompt verbatim."""
        prompt = f'You are a professional geological disaster expert tasked with creating training samples for a geological disaster assistant. \n\n**Task**: Generate a high-quality sample for the disaster type "{disaster_type}" - "{sub_disaster_type}" with question type "{question_type}".\n\n**Format Requirements**:\n- instruction: "You are a helpful geological disaster assistant. This is a {question_type} task."\n- input: A relevant question/request about {sub_disaster_type}\n- output: A comprehensive, accurate, and professional response\n\n**Question Type Guidelines**:\n- question & answer: Direct factual questions requiring precise answers\n- open-ended question: Questions allowing for detailed explanations and multiple perspectives  \n- summary question: Questions requiring summarization of complex geological information\n- reasoning question: Questions requiring analysis, cause-effect reasoning, or problem-solving\n\n**Examples of similar samples**:\n'
        for i, example in enumerate(examples, 1):
            prompt += f'\nExample {i}:\n'
            prompt += f"instruction: {example['instruction']}\n"
            prompt += f"input: {example['input']}\n"
            prompt += f"output: {example['output']}\n"
        prompt += f'\n\n**Requirements**:\n1. The content must be scientifically accurate and professionally written\n2. Focus specifically on {sub_disaster_type} within the {disaster_type} category\n3. Maintain the exact instruction format with "{question_type} task"\n4. Provide professional, comprehensive, accurate, and non-fictional responses\n5. Use proper geological terminology\n6. Return only the JSON object with instruction, input, and output fields\n\nGenerate one sample now:'
        return prompt

    def generate_one(
        self,
        category: str,
        subcategory: str,
        question_type: str,
        examples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = self.create_prompt(category, subcategory, question_type, examples)
        return parse_json_object(self.client.generate(prompt))

    def append_sample(
        self,
        category: str,
        subcategory: str,
        question_type: str,
        sample: Dict[str, Any],
    ) -> int:
        path = self.generated_path(category, subcategory, question_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = self.generated_samples(category, subcategory, question_type)
        samples.append(sample)
        path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(samples)

    def amplify_all(self) -> None:
        all_samples = self.load_existing_samples()
        for category, subcategories in all_samples.items():
            for subcategory in subcategories:
                for question_type in QUESTION_TYPES:
                    existing = len(self.generated_samples(category, subcategory, question_type))
                    while existing < self.samples_per_type:
                        examples = self.select_examples(
                            all_samples, category, subcategory, question_type
                        )
                        if not examples:
                            print(f"No examples available for {category}/{subcategory}/{question_type}.")
                            break
                        try:
                            sample = self.generate_one(
                                category, subcategory, question_type, examples
                            )
                        except Exception as exc:
                            print(f"Generation failed for {category}/{subcategory}: {exc}")
                            break
                        existing = self.append_sample(
                            category, subcategory, question_type, sample
                        )
                        print(
                            f"Generated {category}/{subcategory}/{question_type}: "
                            f"{existing}/{self.samples_per_type}"
                        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument("--base-url")
    parser.add_argument("--samples-per-type", type=int, default=10)
    parser.add_argument("--generated-example-threshold", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.samples_per_type < 1 or args.generated_example_threshold < 0:
        parser.error("sample counts must be non-negative, and samples-per-type must be positive")

    client = OpenAICompatibleClient(model=args.model, base_url=args.base_url)
    amplifier = SampleAmplifier(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        client=client,
        samples_per_type=args.samples_per_type,
        generated_example_threshold=args.generated_example_threshold,
        rng=random.Random(args.seed),
    )
    amplifier.amplify_all()


if __name__ == "__main__":
    main()
