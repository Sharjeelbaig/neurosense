from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sentence_transformers import InputExample


class DataFormatError(ValueError):
    pass


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with src.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise DataFormatError(f"Invalid JSON at {path}:{idx}: {exc}") from exc
            rows.append(row)

    if not rows:
        raise DataFormatError(f"No usable JSONL rows found in: {path}")
    return rows


def load_training_examples(path: str) -> tuple[list[InputExample], list[InputExample]]:
    rows = _read_jsonl(path)
    pairs: list[InputExample] = []
    triplets: list[InputExample] = []

    for idx, row in enumerate(rows, start=1):
        query = str(row.get("query", "")).strip()
        positive = str(row.get("positive", "")).strip()
        negative = str(row.get("negative", "")).strip()

        if not query or not positive:
            raise DataFormatError(
                f"Row {idx} in {path} must include non-empty 'query' and 'positive'."
            )

        if negative:
            triplets.append(InputExample(texts=[query, positive, negative]))
        else:
            pairs.append(InputExample(texts=[query, positive]))

    return pairs, triplets


def load_ir_data(corpus_file: str, queries_file: str) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    corpus_rows = _read_jsonl(corpus_file)
    query_rows = _read_jsonl(queries_file)

    corpus: dict[str, str] = {}
    for idx, row in enumerate(corpus_rows, start=1):
        doc_id = str(row.get("id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not doc_id or not text:
            raise DataFormatError(
                f"Corpus row {idx} in {corpus_file} must include non-empty 'id' and 'text'."
            )
        corpus[doc_id] = text

    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    for idx, row in enumerate(query_rows, start=1):
        query_id = str(row.get("id", "")).strip()
        query_text = str(row.get("query", "")).strip()
        positives = row.get("positive_ids", [])

        if not query_id or not query_text:
            raise DataFormatError(
                f"Query row {idx} in {queries_file} must include non-empty 'id' and 'query'."
            )
        if not isinstance(positives, list) or not positives:
            raise DataFormatError(
                f"Query row {idx} in {queries_file} must include non-empty list 'positive_ids'."
            )

        positive_set = {str(x).strip() for x in positives if str(x).strip()}
        if not positive_set:
            raise DataFormatError(
                f"Query row {idx} in {queries_file} has invalid 'positive_ids'."
            )

        queries[query_id] = query_text
        relevant_docs[query_id] = positive_set

    return corpus, queries, relevant_docs


def read_text_lines(path: str) -> list[str]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {path}")

    lines = [line.strip() for line in src.read_text(encoding="utf-8").splitlines()]
    items = [line for line in lines if line]
    if not items:
        raise DataFormatError(f"No non-empty lines found in: {path}")
    return items
