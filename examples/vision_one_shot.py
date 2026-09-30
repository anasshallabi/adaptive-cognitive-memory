"""Usage: python -m examples.vision_one_shot --support path --label ZENTRA --query path"""
import argparse
import json

from acm.vision import OpenCLIPEncoder, VisionMemory


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-encoder one-shot image demo")
    parser.add_argument("--support", required=True, help="One labeled image")
    parser.add_argument("--label", required=True)
    parser.add_argument("--query", required=True, nargs="+", help="Unseen query image(s)")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Uncalibrated example threshold; validate on separate data")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()

    encoder = OpenCLIPEncoder(device=args.device, model_name=args.model, pretrained=args.pretrained)
    memory = VisionMemory(encoder)
    memory.learn_image(args.support, args.label)
    print(json.dumps({
        "run": encoder.metadata(),
        "support": args.support,
        "label": args.label,
        "results": [
            {"image": p, **memory.recognize_image(p, threshold=args.threshold)}
            for p in args.query
        ],
        "warning": "A similarity score is not a calibrated probability or proof of concept learning.",
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
