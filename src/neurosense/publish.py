from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Neurosense model to Hugging Face")
    parser.add_argument("--model-dir", required=True, help="Directory that contains model files")
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. username/Neurosense")
    parser.add_argument(
        "--private",
        default="false",
        help="true/false (default: false)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token. If omitted, uses HF_TOKEN env var.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Missing Hugging Face token. Pass --hf-token or set HF_TOKEN.")

    private = str_to_bool(args.private)

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, private=private, exist_ok=True)

    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"Published Neurosense model to: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
