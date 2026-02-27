from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from neurosense.data import read_text_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create embeddings with Neurosense")
    parser.add_argument("--model-dir", required=True, help="Path or HF repo id of model")
    parser.add_argument("--input-file", required=True, help="Plain text file, one text per line")
    parser.add_argument("--output-file", required=True, help="Output .npy file")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--normalize", action="store_true", help="L2 normalize embeddings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    texts = read_text_lines(args.input_file)
    model = SentenceTransformer(args.model_dir)

    vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=args.normalize,
    )

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, vectors)

    print(f"Encoded {len(texts)} texts")
    print(f"Embedding shape: {vectors.shape}")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
