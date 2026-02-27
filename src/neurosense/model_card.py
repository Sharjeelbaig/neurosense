from __future__ import annotations

from pathlib import Path
from typing import Any


def write_model_card(output_dir: str, context: dict[str, Any]) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    card_path = target / "README.md"

    base_model = context.get("base_model", "unknown")
    train_file = context.get("train_file", "unknown")
    train_pairs = context.get("train_pairs", 0)
    train_triplets = context.get("train_triplets", 0)
    epochs = context.get("epochs", "unknown")
    batch_size = context.get("batch_size", "unknown")
    max_seq_length = context.get("max_seq_length", "unknown")

    card = f"""---
language:
- en
license: apache-2.0
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- embeddings
- semantic-search
- neurosense
base_model: {base_model}
---

# Neurosense

Neurosense is an embedding model optimized for semantic retrieval.

## Training Summary

- Base model: `{base_model}`
- Training file: `{train_file}`
- Pair examples: `{train_pairs}`
- Triplet examples: `{train_triplets}`
- Epochs: `{epochs}`
- Batch size: `{batch_size}`
- Max sequence length: `{max_seq_length}`

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{context.get('hf_repo_id', 'path-or-repo-id')}")
embeddings = model.encode([
    "example query",
    "example document"
], normalize_embeddings=True)
```

## Intended Use

- Semantic search
- Dense retrieval / RAG
- Similarity matching
"""

    card_path.write_text(card, encoding="utf-8")
    return card_path
