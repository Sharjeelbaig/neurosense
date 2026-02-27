from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def metric_for_dataset(
    model: SentenceTransformer,
    dataset_dir: Path,
    batch_size: int,
    query_prefix: str = "",
    corpus_prefix: str = "",
) -> dict:
    corpus_rows = read_jsonl(dataset_dir / "corpus.jsonl")
    query_rows = read_jsonl(dataset_dir / "queries.jsonl")

    corpus_ids = [row["id"] for row in corpus_rows]
    corpus_texts = [f"{corpus_prefix}{row['text']}" for row in corpus_rows]

    query_ids = [row["id"] for row in query_rows]
    query_texts = [f"{query_prefix}{row['query']}" for row in query_rows]
    qrels = [set(row["positive_ids"]) for row in query_rows]

    corpus_emb = model.encode(
        corpus_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    query_emb = model.encode(
        query_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    sims = np.matmul(query_emb, corpus_emb.T)

    hit1 = 0.0
    hit5 = 0.0
    mrr10 = 0.0
    recall10 = 0.0
    ap10 = 0.0

    per_query = []
    for i, qid in enumerate(query_ids):
        positives = qrels[i]
        scores = sims[i]
        top10_idx = np.argpartition(scores, -10)[-10:]
        top10_idx = top10_idx[np.argsort(scores[top10_idx])[::-1]]
        top10_doc_ids = [corpus_ids[j] for j in top10_idx]

        top1_doc = top10_doc_ids[0]
        top5_docs = top10_doc_ids[:5]

        q_hit1 = 1.0 if top1_doc in positives else 0.0
        q_hit5 = 1.0 if any(doc in positives for doc in top5_docs) else 0.0

        rr = 0.0
        correct_in_top10 = 0
        precision_sum = 0.0
        for rank, doc_id in enumerate(top10_doc_ids, start=1):
            if doc_id in positives:
                correct_in_top10 += 1
                if rr == 0.0:
                    rr = 1.0 / rank
                precision_sum += correct_in_top10 / rank

        q_recall10 = correct_in_top10 / max(1, len(positives))
        q_ap10 = precision_sum / min(len(positives), 10)

        hit1 += q_hit1
        hit5 += q_hit5
        mrr10 += rr
        recall10 += q_recall10
        ap10 += q_ap10

        per_query.append(
            {
                "query_id": qid,
                "hit@1": q_hit1,
                "hit@5": q_hit5,
                "mrr@10": rr,
                "recall@10": q_recall10,
                "ap@10": q_ap10,
            }
        )

    n = max(1, len(query_ids))
    return {
        "dataset": dataset_dir.name,
        "queries": len(query_ids),
        "docs": len(corpus_ids),
        "hit@1": hit1 / n,
        "hit@5": hit5 / n,
        "mrr@10": mrr10 / n,
        "recall@10": recall10 / n,
        "map@10": ap10 / n,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate embeddings on multi-corpus retrieval benchmark")
    parser.add_argument("--model", required=True, help="Model path or HF repo id")
    parser.add_argument("--bench-root", default=".test/data/benchmarks")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu, cuda, mps")
    parser.add_argument("--query-prefix", default="", help="Optional prefix for each query text")
    parser.add_argument("--corpus-prefix", default="", help="Optional prefix for each corpus text")
    args = parser.parse_args()

    bench_root = Path(args.bench_root)
    dataset_dirs = sorted([p for p in bench_root.iterdir() if p.is_dir()])
    if not dataset_dirs:
        raise FileNotFoundError(f"No benchmark datasets found under: {bench_root}")

    print(f"Loading model: {args.model} on device={args.device}")
    model = SentenceTransformer(args.model, device=args.device)

    results = []
    for dataset_dir in dataset_dirs:
        print(f"Evaluating {dataset_dir.name}...")
        metrics = metric_for_dataset(
            model,
            dataset_dir,
            args.batch_size,
            query_prefix=args.query_prefix,
            corpus_prefix=args.corpus_prefix,
        )
        results.append(metrics)
        print(
            f"{dataset_dir.name}: hit@1={metrics['hit@1']:.4f} "
            f"mrr@10={metrics['mrr@10']:.4f} recall@10={metrics['recall@10']:.4f}"
        )

    total_queries = sum(r["queries"] for r in results)
    weighted = {
        "hit@1": sum(r["hit@1"] * r["queries"] for r in results) / total_queries,
        "hit@5": sum(r["hit@5"] * r["queries"] for r in results) / total_queries,
        "mrr@10": sum(r["mrr@10"] * r["queries"] for r in results) / total_queries,
        "recall@10": sum(r["recall@10"] * r["queries"] for r in results) / total_queries,
        "map@10": sum(r["map@10"] * r["queries"] for r in results) / total_queries,
    }

    report = {
        "model": args.model,
        "benchmark_root": str(bench_root),
        "datasets": results,
        "aggregate_weighted": {"queries": total_queries, **weighted},
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nAggregate:")
    print(
        f"queries={total_queries} hit@1={weighted['hit@1']:.4f} "
        f"mrr@10={weighted['mrr@10']:.4f} recall@10={weighted['recall@10']:.4f}"
    )
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
