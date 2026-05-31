from __future__ import annotations

import argparse

from huggingface_hub import model_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve a Hugging Face model id to its current commit SHA.")
    parser.add_argument("model_id")
    args = parser.parse_args()
    print(model_info(args.model_id).sha)


if __name__ == "__main__":
    main()
