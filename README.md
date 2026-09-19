# GeohazardGPT

GeohazardGPT is an 8B geohazard-domain language-model system for professional question answering, engineering reasoning, and standards-grounded analysis.

The released model is available at [Hugging Face](https://huggingface.co/pengfali/GeohazardGPT).

The project develops a geohazard-focused language model for technical question answering and engineering reasoning. Its workflow combines a 12-category geohazard taxonomy, processed literature and web or book text, expert and teacher-assisted SFT construction, Qwen3-8B LoRA adaptation, and standards-grounded retrieval augmented generation. General instruction mixing is used to retain broad instruction-following ability during domain adaptation.

## Overview

- **GeoInstruct:** taxonomy-guided semantic filtering and instruction generation for geohazard domain adaptation.
- **LoRA fine-tuning:** Qwen3-8B is adapted with the configuration in `src/qwen_3_8b_lora_sft.yaml`.
- **Standards RAG:** Qwen3-Embedding-4B retrieves clause-level evidence, Qwen3-Reranker-4B reranks candidates, and the fine-tuned model generates the answer.

The project covers 12 geohazard categories, including Coastal Hazards, Crustal Activity, Land Degradation, Mining and Underground Engineenng, River, Lake, and Reservoir Hazards, Slope Mass Movement, Special Geotechnical Hazards, Ground Defomation, Marine Geological Hazards, Soil and Water Pollution and Geochemical Anomalies, Water Source Depletion, Urban Geological Hazards

## Main components

```text
src/
├── get_pdf_links.py                         # Collect open-access links
├── pdf2txt2parquet.py                       # Extract PDF text and metadata
├── parquet_process.py                       # Clean and chunk the corpus
├── build_sft_corpus.py                      # Combine source text for SFT sample generation
├── merge_all_training_data.py               # Merge processed corpora
├── semantic_search.py                       # FAISS semantic filtering
├── generate_sft_data.py                     # Generate SFT data from retrieved and seed text
├── merge_sft_samples.py                     # Validate and merge generated SFT samples
├── augment_sft.py                          # Alternative seed-example augmentation
├── qwen_3_8b_lora_sft.yaml                  # LoRA configuration
├── build_database.py                         # Build the Chroma index
└── query_rag.py                              # Retrieve, rerank, assemble, and infer
```

## Key settings

- Teacher model: `GPT-4o-2024-11-20`
- Generator model: `Qwen3-8B`
- Embedding model: `Qwen3-Embedding-4B`
- Reranker model: `Qwen3-Reranker-4B`
- Retrieval: top 30 candidates
- Reranking: retain top 15 candidates
- LoRA rank/alpha/dropout: `128/256/0.05`
- Learning rate: `2e-4`

## Basic workflow

Run commands from the repository root. Replace paths with local paths.

```bash
python src/build_sft_corpus.py \
  --source-dir /path/to/processed_sources \
  --c4-dir /path/to/filtered_c4 \
  --output /path/to/sft_corpus.jsonl

python src/semantic_search.py \
  --expert-dir /path/to/expert_seed \
  --llm-dir /path/to/generated_seed \
  --merged-data /path/to/merged_data.jsonl \
  --index-path /path/to/seed.faiss \
  --info-path /path/to/seed_metadata.pkl \
  --output /path/to/top_semantic_matches.json

python src/generate_sft_data.py \
  --base-dir /path/to/sft_inputs \
  --retrieved-file /path/to/top_semantic_matches.json \
  --output-dir /path/to/generated_sft \
  --model gpt-4o-2024-11-20

python src/merge_sft_samples.py \
  --input-dir /path/to/generated_sft \
  --output /path/to/final_data.json \
  --deduplicate

python src/augment_sft.py \
  --data-dir /path/to/seed_samples \
  --output-dir /path/to/generated_samples \
  --model gpt-4o-2024-11-20

llamafactory-cli train src/qwen_3_8b_lora_sft.yaml

python src/build_database.py \
  --data-dir /path/to/standards_json \
  --embedding-model /path/to/Qwen3-Embedding-4B \
  --db-path /path/to/chroma_db \
  --collection engineering_specs_qwen3

python src/query_rag.py retrieve \
  --embedding-model /path/to/Qwen3-Embedding-4B \
  --reranker-model /path/to/Qwen3-Reranker-4B \
  --db-path /path/to/chroma_db \
  --input /path/to/test.json \
  --output /path/to/retrieval_evidence.json \
  --retrieve-k 30 \
  --rerank-k 15

python src/query_rag.py assemble \
  --input /path/to/retrieval_evidence.json \
  --output /path/to/rag_prompts.json

python src/query_rag.py infer \
  --model /path/to/GeohazardGPT \
  --input /path/to/rag_prompts.json \
  --output /path/to/rag_results.json
```

## Benchmark inference

Use `bench_local.py` or `exam_local.py` for local models.
Pass `--adapter` for LoRA inference; omit it for base or merged models.
Bench inputs contain `instruction` and `input`. Exam inputs contain `correct_answer`
and either `prompt` or `question_type`, `question_content`, and `options`.

```bash
python src/bench_local.py --model /path/to/model --input /path/to/bench.json --output /path/to/bench_predictions.json
python src/exam_local.py --model /path/to/base_model --adapter /path/to/adapter --input /path/to/exam.json --output /path/to/exam_predictions.json
python src/bench_api.py --model MODEL_ID --input /path/to/bench.json --output /path/to/bench_api_predictions.json
python src/exam_api.py --model MODEL_ID --input /path/to/exam.json --output /path/to/exam_api_predictions.json
```

API inference reads `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`.
Bench saves response records for downstream scoring; Exam saves raw responses,
parsed options, correctness, and accuracy. Exam parsing uses explicit answer labels
or standalone options instead of collecting letters from the explanation.
Local decoding defaults (base and LoRA models):

- GeohazardBench: `max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1`.
- GeohazardExam: `max_new_tokens=1024, do_sample=False, temperature=0.3, repetition_penalty=1.1`. Temperature has no effect during greedy decoding.

## License

The source code is released under the MIT License. Third-party models and source materials follow their own licenses.
