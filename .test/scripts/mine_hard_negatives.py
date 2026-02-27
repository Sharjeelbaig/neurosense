from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DATASET_HF = {
    "scifact": "mteb/scifact",
    "fiqa": "mteb/fiqa",
    "nfcorpus": "mteb/nfcorpus",
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def doc_text(row: dict) -> str:
    title = str(row.get("title", "") or "").strip()
    text = str(row.get("text", "") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def load_corpus(name: str, bench_root: str | None = None) -> tuple[list[str], list[str]]:
    if bench_root:
        bench_file = Path(bench_root) / name / "corpus.jsonl"
        if bench_file.exists():
            ids: list[str] = []
            texts: list[str] = []
            for row in read_jsonl(bench_file):
                raw_id = str(row.get("id", "")).strip()
                doc_id = raw_id.split(":", 1)[1] if ":" in raw_id else raw_id
                text = str(row.get("text", "") or "").strip()
                if doc_id and text:
                    ids.append(doc_id)
                    texts.append(text)
            if ids:
                return ids, texts

    ds = load_dataset(DATASET_HF[name], "corpus", split="corpus")
    ids: list[str] = []
    texts: list[str] = []
    for row in ds:
        did = str(row.get("id", row.get("_id", ""))).strip()
        text = doc_text(row)
        if did and text:
            ids.append(did)
            texts.append(text)
    return ids, texts


def load_query_positives(name: str) -> dict[str, set[str]]:
    ds = load_dataset(DATASET_HF[name], "default", split="train")
    out: dict[str, set[str]] = {}
    for row in ds:
        score = float(row.get("score", 0.0) or 0.0)
        if score <= 0:
            continue
        qid = str(row.get("query-id", "")).strip()
        did = str(row.get("corpus-id", "")).strip()
        if qid and did:
            out.setdefault(qid, set()).add(did)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives for round2 triplet training")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-pairs", default=".test/data/train_pairs_round1.jsonl")
    parser.add_argument("--out-file", default=".test/data/train_triplets_round2.jsonl")
    parser.add_argument("--max-triplets", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument("--bench-root", default=None, help="Optional benchmark root for sampled corpora")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    rows = read_jsonl(Path(args.train_pairs))
    if not rows:
        raise ValueError("No rows in train pairs file")

    by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        name = str(row.get("dataset", "")).strip()
        if name in DATASET_HF:
            by_dataset.setdefault(name, []).append(row)

    model = SentenceTransformer(args.model)

    triplets: list[dict] = []
    mined_count: dict[str, int] = {}

    for name, dataset_rows in by_dataset.items():
        print(f"Mining negatives for {name} ({len(dataset_rows)} pairs)...")

        corpus_ids, corpus_texts = load_corpus(name, bench_root=args.bench_root)
        positives_by_query = load_query_positives(name)
        doc_text_by_id = {did: txt for did, txt in zip(corpus_ids, corpus_texts)}

        corpus_emb = model.encode(
            corpus_texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        qid_to_text: dict[str, str] = {}
        for row in dataset_rows:
            qid = str(row["query_id"])
            qid_to_text[qid] = row["query"]

        qids = list(qid_to_text.keys())
        qtexts = [qid_to_text[qid] for qid in qids]
        qemb = model.encode(
            qtexts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        neg_by_qid: dict[str, str] = {}
        corpus_size = len(corpus_ids)
        k = min(args.top_k, corpus_size)

        for start in range(0, len(qids), args.query_chunk):
            end = min(start + args.query_chunk, len(qids))
            chunk_emb = qemb[start:end]
            sims = np.matmul(chunk_emb, corpus_emb.T)

            for local_idx, qid in enumerate(qids[start:end]):
                row = sims[local_idx]
                idx = np.argpartition(row, -k)[-k:]
                idx = idx[np.argsort(row[idx])[::-1]]

                forbidden = positives_by_query.get(qid, set())
                chosen = None
                for j in idx:
                    did = corpus_ids[int(j)]
                    if did not in forbidden:
                        chosen = did
                        break

                if chosen is None:
                    candidates = [did for did in corpus_ids if did not in forbidden]
                    chosen = rng.choice(candidates) if candidates else corpus_ids[0]

                neg_by_qid[qid] = chosen

        for row in dataset_rows:
            qid = str(row["query_id"])
            neg_id = neg_by_qid[qid]
            negative_text = doc_text_by_id.get(neg_id)
            if not negative_text:
                continue
            triplets.append(
                {
                    "dataset": name,
                    "query_id": qid,
                    "query": row["query"],
                    "positive": row["positive"],
                    "negative": negative_text,
                }
            )

        mined_count[name] = len(dataset_rows)

    rng.shuffle(triplets)
    triplets = triplets[: args.max_triplets]
    write_jsonl(Path(args.out_file), triplets)

    summary = {
        "model": args.model,
        "input_pairs": len(rows),
        "triplets_written": len(triplets),
        "by_dataset_input": {k: len(v) for k, v in by_dataset.items()},
        "by_dataset_mined": mined_count,
        "out_file": args.out_file,
    }
    summary_path = Path(args.out_file).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote triplets: {args.out_file}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
