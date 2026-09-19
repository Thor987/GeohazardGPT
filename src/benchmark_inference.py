"""Shared inference utilities for GeohazardBench and GeohazardExam."""

import argparse
import json
import os
import re
from pathlib import Path


def exam_prompt(row):
    """Preserve prepared prompts, or construct the original Chinese exam prompt."""
    if row.get("prompt"):
        return row["prompt"]
    single = "select one" in row["question_type"].lower() or "单选" in row["question_type"]
    kind = "单选题" if single else "多选题"
    options = "".join(f"{key}: {value}\n" for key, value in row["options"].items())
    instruction = (
        "请仔细分析题目并选择一个正确答案。请按以下格式回答：\n\n答案：[选项字母]\n\n解析：[根据你的专业知识详细解释为什么选择这个答案，分析每个选项的正确性]"
        if single else
        "请仔细分析题目并选择所有正确答案。请按以下格式回答：\n\n答案：[选项字母，多个用|分隔，如A|B|C]\n\n解析：[根据你的专业知识详细解释为什么选择这些答案，分析每个选项的正确性]"
    )
    return (
        "你是一位精通中国地质灾害与岩土工程国家标准、行业标准的注册岩土工程师。请严格依据提供的规范条款，用准确、专业的技术语言回答问题。\n\n"
        f"题目类型：{kind}\n题目：{row['question_content']}\n\n选项：\n{options}\n\n{instruction}"
    )


def extract_answer(response, single):
    """Parse an explicit answer; never collect option letters from the explanation."""
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.S)
    pattern = r"(?:最终答案|正确答案|答案|final answer|answer)\s*[:：]\s*\**\s*\[?\s*([A-D](?:[\s|、,，]*[A-D])*)(?![A-Za-z])"
    matches = re.findall(pattern, response, flags=re.I)
    if matches:
        value = matches[-1]
    elif re.fullmatch(r"\s*[A-D](?:[\s|、,，]*[A-D])*\s*", response, flags=re.I):
        value = response
    else:
        return "NO_ANSWER_FOUND"
    letters = sorted(set(re.findall(r"[A-D]", value.upper())))
    if single and len(letters) != 1:
        return "NO_ANSWER_FOUND"
    return "|".join(letters)


def save_json(path, value):
    """Atomically save a complete JSON document after each prediction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def local_generator(args, task):
    """Load a base or merged model and an optional LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=args.trust_remote_code
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def generate(prompt):
        if task == "bench" or args.chat_template:
            messages = [{"role": "user", "content": prompt}]
            if task == "bench":
                messages.insert(0, {"role": "system", "content": "You are a helpful geological disaster assistant."})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        else:
            encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        settings = dict(max_new_tokens=args.max_new_tokens, do_sample=args.sample,
                        temperature=args.temperature,
                        repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id)
        if args.sample:
            settings.update(top_p=args.top_p, top_k=50)
        # Temperature is retained for experiment traceability but ignored by greedy decoding.
        with torch.inference_mode():
            output = model.generate(**encoded, **settings)
        tokens = output[0, encoded["input_ids"].shape[1]:]
        return tokenizer.decode(tokens, skip_special_tokens=True).strip()

    return generate


def api_generator(args):
    """Use an OpenAI-compatible endpoint configured through environment variables."""
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY before API inference.")
    client = OpenAI(api_key=key, base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"))

    def generate(prompt):
        response = client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature, max_tokens=args.max_new_tokens,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("Response truncated; increase --max-new-tokens.")
        if not choice.message.content:
            raise RuntimeError("Empty API response.")
        return choice.message.content.strip()

    return generate


def main(task, backend):
    parser = argparse.ArgumentParser(description=f"Geohazard{task.title()} {backend} inference.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7 if task == "bench" else 0.3)
    if backend == "local":
        parser.add_argument("--adapter", help="Optional LoRA adapter directory.")
        parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--sample", action=argparse.BooleanOptionalAction, default=task == "bench")
        parser.add_argument("--top-p", type=float, default=0.9)
        parser.add_argument("--chat-template", action="store_true", help="Apply a chat template to Exam prompts.")
        parser.add_argument("--trust-remote-code", action="store_true")
    else:
        parser.add_argument("--base-url", help="Optional OpenAI-compatible endpoint.")
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error("Input and output must be different files.")
    if args.output.exists():
        parser.error("Output already exists; choose a new output file.")
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        parser.error("Input must be a non-empty JSON array.")
    generate = local_generator(args, task) if backend == "local" else api_generator(args)
    results = []
    for index, row in enumerate(rows):
        prompt = exam_prompt(row) if task == "exam" else row.get("instruction", "") + ("\n" if backend == "local" else "\n\n") + row["input"]
        response = generate(prompt)
        if task == "bench":
            result = {"instruction": row.get("instruction", ""), "input": row["input"], "output": response}
        else:
            kind = row.get("question_type", "")
            single = "select one" in kind.lower() or "单选" in kind or (not kind and "单选题" in prompt)
            answer = extract_answer(response, single)
            gold = "|".join(sorted(set(re.findall(r"[A-D]", str(row["correct_answer"]).upper()))))
            analysis = re.search(r"解析[：:]([\s\S]*)", response)
            result = dict(row, question_number=row.get("question_number", index + 1), prompt=prompt,
                          raw_response=response, model_answer=answer,
                          model_analysis=analysis.group(1).strip() if analysis else "",
                          is_correct=bool(gold) and answer == gold)
        results.append(result)
        payload = results
        if task == "exam":
            correct = sum(item["is_correct"] for item in results)
            payload = dict(model=Path(args.model).name, total_questions=len(rows),
                           total_questions_processed=len(results), correct_answers=correct,
                           accuracy=correct / len(results), results=results)
        save_json(args.output, payload)
        print(f"Saved {index + 1}/{len(rows)} predictions.", flush=True)
