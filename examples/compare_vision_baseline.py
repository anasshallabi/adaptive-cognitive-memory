"""Matched real-image comparison using one frozen OpenCLIP representation.

Example:
    python -m examples.compare_vision_baseline \
      --manifest datasets/vision_v10_test.csv --threshold 0.85 --device auto

The threshold must be frozen from separate validation data before final test.
"""
import argparse
import json

from acm.benchmark import load_manifest
from acm.vision import OpenCLIPEncoder
from acm.vision_compare import evaluate_matched_one_shot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ACM to direct cosine 1-NN on identical OpenCLIP embeddings"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    encoder = OpenCLIPEncoder(
        device=args.device,
        model_name=args.model,
        pretrained=args.pretrained,
    )
    report = evaluate_matched_one_shot(
        load_manifest(args.manifest),
        encoder,
        threshold=args.threshold,
    )
    if args.summary:
        report = {k: v for k, v in report.items() if k != "details"}

    print(json.dumps({
        "run": encoder.metadata(),
        "manifest": args.manifest,
        "metrics": report,
        "warning": (
            "OpenCLIP is pretrained. This experiment tests new label binding "
            "on a frozen representation, not visual concept learning from scratch."
        ),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
