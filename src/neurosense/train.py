from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

from neurosense.data import load_training_examples
from neurosense.model_card import write_model_card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Neurosense embedding model")
    parser.add_argument("--train-file", required=True, help="JSONL with query/positive[/negative]")
    parser.add_argument("--output-dir", default="models/Neurosense", help="Output directory")
    parser.add_argument("--base-model", default="BAAI/bge-small-en-v1.5", help="HF base embedding model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pairs, triplets = load_training_examples(args.train_file)
    if not pairs and not triplets:
        raise ValueError("No training examples loaded.")

    print(f"Loaded {len(pairs)} pair examples and {len(triplets)} triplet examples.")
    print(f"Loading base model: {args.base_model}")

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_length

    train_objectives = []
    total_steps_per_epoch = 0

    if pairs:
        pair_loader = DataLoader(pairs, shuffle=True, batch_size=args.batch_size, drop_last=False)
        train_objectives.append((pair_loader, losses.MultipleNegativesRankingLoss(model)))
        total_steps_per_epoch += len(pair_loader)

    if triplets:
        triplet_loader = DataLoader(triplets, shuffle=True, batch_size=args.batch_size)
        train_objectives.append((triplet_loader, losses.TripletLoss(model=model)))
        total_steps_per_epoch += len(triplet_loader)

    warmup_steps = int(total_steps_per_epoch * args.epochs * args.warmup_ratio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=train_objectives,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(output_dir),
        show_progress_bar=True,
        use_amp=args.use_amp,
    )

    write_model_card(
        output_dir=str(output_dir),
        context={
            "base_model": args.base_model,
            "train_file": args.train_file,
            "train_pairs": len(pairs),
            "train_triplets": len(triplets),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
        },
    )

    print(f"Saved Neurosense model to: {output_dir}")


if __name__ == "__main__":
    main()
