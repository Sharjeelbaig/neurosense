# Neurosense

Neurosense is an embedding SLM pipeline for semantic search.

It includes:
- Training from query-positive or query-positive-negative JSONL.
- Retrieval evaluation (queries/corpus JSONL).
- Embedding generation for local text files.
- Publishing trained checkpoints to Hugging Face.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## 2) Training Data Format

`train.jsonl` lines support either pair or triplet:

```json
{"query":"what is migraine aura?","positive":"Migraine aura is a sensory disturbance.","negative":"A diesel engine uses compression ignition."}
{"query":"symptoms of sleep apnea","positive":"Sleep apnea often causes loud snoring and daytime fatigue."}
```

## 3) Train Neurosense

```bash
neurosense-train \
  --train-file data/sample/train.jsonl \
  --output-dir models/Neurosense \
  --base-model BAAI/bge-small-en-v1.5 \
  --epochs 1 \
  --batch-size 16
```

## 4) Evaluate Retrieval Quality

```bash
neurosense-eval \
  --model-dir models/Neurosense \
  --corpus-file data/sample/corpus.jsonl \
  --queries-file data/sample/queries.jsonl
```

## 5) Embed Texts

```bash
neurosense-embed \
  --model-dir models/Neurosense \
  --input-file data/sample/texts.txt \
  --output-file outputs/embeddings.npy
```

## 6) Publish to Hugging Face

```bash
export HF_TOKEN=your_hf_write_token
neurosense-publish \
  --model-dir models/Neurosense \
  --repo-id YOUR_USERNAME/Neurosense \
  --private false
```

## Notes

- For production quality, use domain-specific pairs/triplets from your own corpus and queries.
- Good first target: 100k+ quality query-document pairs.
- If you do not have labels yet, start with weak supervision (BM25 hard-negative mining + click logs if available).
