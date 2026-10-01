"""Calibrate a one-shot rejection threshold on a validation manifest only.

Example:
    python -m examples.calibrate_vision_threshold \
      --manifest datasets/vision_v10_validation.csv --device auto

Freeze the returned threshold before running the final test manifest.
"""
import argparse
import json

from acm.benchmark import load_manifest
from acm.vision import OpenCLIPEncoder
from acm.vision_compare import calibrate_rejection_threshold


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit cosine rejection threshold on separate validation data"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()

    encoder = OpenCLIPEncoder(
        device=args.device,
        model_name=args.model,
        pretrained=args.pretrained,
    )
    report = calibrate_rejection_threshold(
        load_manifest(args.manifest), encoder
    )
    print(json.dumps({
        "run": encoder.metadata(),
        "validation_manifest": args.manifest,
        "calibration": report,
        "instruction": (
            "Record this threshold before inspecting final test predictions. "
            "Do not retune it on the test manifest."
        ),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
