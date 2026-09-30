"""Usage: python -m examples.evaluate_images --manifest datasets/episode.csv"""
import argparse
import json

from acm.benchmark import evaluate_one_shot, load_manifest
from acm.vision import OpenCLIPEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one-shot images with split checks")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--threshold", type=float, required=True,
                        help="Must be tuned ONLY on separate validation brands")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    from acm.benchmark import validate_manifest
    validate_manifest(rows)
    encoder = OpenCLIPEncoder(device=args.device, model_name=args.model, pretrained=args.pretrained)
    report = evaluate_one_shot(rows, encoder, threshold=args.threshold)
    print(json.dumps({"run": encoder.metadata(), "metrics": report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
