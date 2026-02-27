from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

DATASETS = {
    "scifact": {
        "hf_name": "mteb/scifact",
        "train_split": "train",
        "eval_split": "test",
        "eval_queries": 50,
        "train_cap": 2000,
        "corpus_cap": 1200,
    },
    "fiqa": {
        "hf_name": "mteb/fiqa",
        "train_split": "train",
        "eval_split": "test",
        "eval_queries": 50,
        "train_cap": 12000,
        "corpus_cap": 1200,
    },
    "nfcorpus": {
        "hf_name": "mteb/nfcorpus",
        "train_split": "train",
        "eval_split": "test",
        "eval_queries": 50,
        "train_cap": 20000,
        "corpus_cap": 1200,
    },
}


def _doc_text(row: dict) -> str:
    title = str(row.get("title", "") or "").strip()
    text = str(row.get("text", "") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def _load_corpus(hf_name: str) -> dict[str, str]:
    ds = load_dataset(hf_name, "corpus", split="corpus")
    out: dict[str, str] = {}
    for row in ds:
        doc_id = str(row.get("id", row.get("_id", ""))).strip()
        text = _doc_text(row)
        if doc_id and text:
            out[doc_id] = text
    return out


def _load_queries(hf_name: str) -> dict[str, str]:
    ds = load_dataset(hf_name, "queries", split="queries")
    out: dict[str, str] = {}
    for row in ds:
        qid = str(row.get("id", row.get("_id", ""))).strip()
        text = str(row.get("text", "") or "").strip()
        if qid and text:
            out[qid] = text
    return out


def _load_qrels(hf_name: str, split: str) -> dict[str, set[str]]:
    ds = load_dataset(hf_name, "default", split=split)
    out: dict[str, set[str]] = {}
    for row in ds:
        score = float(row.get("score", 0.0) or 0.0)
        if score <= 0:
            continue
        qid = str(row.get("query-id", "")).strip()
        did = str(row.get("corpus-id", "")).strip()
        if not qid or not did:
            continue
        out.setdefault(qid, set()).add(did)
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare internet corpora benchmark + training pairs")
    parser.add_argument("--out-dir", default=".test/data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    bench_dir = out_dir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    all_train_pairs: list[dict] = []
    summary: dict[str, dict] = {}

    for name, cfg in DATASETS.items():
        hf_name = cfg["hf_name"]
        print(f"\\nPreparing dataset: {name} ({hf_name})")

        corpus = _load_corpus(hf_name)
        queries = _load_queries(hf_name)
        train_qrels = _load_qrels(hf_name, cfg["train_split"])
        eval_qrels = _load_qrels(hf_name, cfg["eval_split"])

        usable_eval_qids = [
            qid
            for qid, pos in eval_qrels.items()
            if qid in queries and any(pid in corpus for pid in pos)
        ]
        rng.shuffle(usable_eval_qids)
        picked_eval_qids = usable_eval_qids[: int(cfg["eval_queries"])]

        eval_queries_rows = []
        eval_positive_doc_ids: set[str] = set()
        for qid in picked_eval_qids:
            pos_ids = [pid for pid in sorted(eval_qrels[qid]) if pid in corpus]
            if not pos_ids:
                continue
            eval_positive_doc_ids.update(pos_ids)
            eval_queries_rows.append(
                {
                    "id": f"{name}:{qid}",
                    "query": queries[qid],
                    "positive_ids": [f"{name}:{pid}" for pid in pos_ids],
                }
            )

        corpus_cap = int(cfg["corpus_cap"])
        all_doc_ids = list(corpus.keys())
        kept_doc_ids: set[str] = set(all_doc_ids)
        if len(all_doc_ids) > corpus_cap:
            must_keep = set(eval_positive_doc_ids)
            remaining = [did for did in all_doc_ids if did not in must_keep]
            rng.shuffle(remaining)
            budget = max(0, corpus_cap - len(must_keep))
            kept_doc_ids = must_keep.union(remaining[:budget])

        eval_corpus_rows = [
            {"id": f"{name}:{doc_id}", "text": text}
            for doc_id, text in corpus.items()
            if doc_id in kept_doc_ids
        ]

        dataset_dir = bench_dir / name
        _write_jsonl(dataset_dir / "corpus.jsonl", eval_corpus_rows)
        _write_jsonl(dataset_dir / "queries.jsonl", eval_queries_rows)

        # Build train pairs from train qrels
        train_pairs = []
        for qid, pos_ids in train_qrels.items():
            q_text = queries.get(qid)
            if not q_text:
                continue
            for did in pos_ids:
                d_text = corpus.get(did)
                if not d_text:
                    continue
                train_pairs.append(
                    {
                        "dataset": name,
                        "query_id": qid,
                        "doc_id": did,
                        "query": q_text,
                        "positive": d_text,
                    }
                )

        rng.shuffle(train_pairs)
        train_pairs = train_pairs[: int(cfg["train_cap"])]
        all_train_pairs.extend(train_pairs)

        summary[name] = {
            "hf_name": hf_name,
            "corpus_docs_full": len(corpus),
            "corpus_docs_benchmark": len(eval_corpus_rows),
            "queries_total": len(queries),
            "train_qrels_queries": len(train_qrels),
            "eval_qrels_queries": len(eval_qrels),
            "eval_queries_selected": len(eval_queries_rows),
            "train_pairs_selected": len(train_pairs),
            "benchmark_dir": str(dataset_dir),
        }

        print(
            f"{name}: docs_full={len(corpus)} docs_bench={len(eval_corpus_rows)} "
            f"eval_queries={len(eval_queries_rows)} train_pairs={len(train_pairs)}"
        )

    rng.shuffle(all_train_pairs)
    train_path = out_dir / "train_pairs_round1.jsonl"
    _write_jsonl(train_path, all_train_pairs)

    summary["overall"] = {
        "total_train_pairs": len(all_train_pairs),
        "train_pairs_file": str(train_path),
        "benchmarks_dir": str(bench_dir),
    }

    summary_path = out_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\\nWrote:")
    print(f"- {train_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
