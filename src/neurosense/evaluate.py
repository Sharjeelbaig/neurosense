from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from neurosense.data import load_ir_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Neurosense retrieval quality")
    parser.add_argument("--model-dir", required=True, help="Path to trained model")
    parser.add_argument("--corpus-file", required=True, help="JSONL corpus: id,text")
    parser.add_argument("--queries-file", required=True, help="JSONL queries: id,query,positive_ids")
    parser.add_argument("--name", default="neurosense-eval", help="Evaluation name")
    parser.add_argument("--output-dir", default="outputs/eval", help="Where evaluator CSV is written")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus, queries, relevant_docs = load_ir_data(args.corpus_file, args.queries_file)
    print(f"Loaded corpus={len(corpus)} docs, queries={len(queries)}")

    model = SentenceTransformer(args.model_dir)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=args.name,
    )

    score = evaluator(model, output_path=str(out))

    if isinstance(score, dict):
        metrics_file = out / f"{args.name}_metrics.json"
        metrics_file.write_text(json.dumps(score, indent=2), encoding="utf-8")

        main_score = None
        for key in ("cosine_ndcg@10", "cosine_map@100", "dot_ndcg@10", "dot_map@100"):
            value = score.get(key)
            if isinstance(value, (int, float)):
                main_score = float(value)
                break
        if main_score is None:
            numeric_values = [float(v) for v in score.values() if isinstance(v, (int, float))]
            main_score = numeric_values[0] if numeric_values else 0.0

        print(f"Main retrieval score ({args.name}): {main_score:.6f}")
        print(f"Raw metric dict written to: {metrics_file}")
    else:
        print(f"Main retrieval score ({args.name}): {score:.6f}")
    print(f"Detailed metrics written under: {out}")


if __name__ == "__main__":
    main()
