"""Validate a v0.10 image manifest without loading OpenCLIP.

Usage:
    python -m examples.check_vision_manifest --manifest data/vision_v10_validation.csv
"""
import argparse
import json

from acm.benchmark import load_manifest, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ACM vision manifest structure and leakage guards"
    )
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    validate_manifest(rows)
    support = [r for r in rows if r.split == "support"]
    known = [r for r in rows if r.split == "known"]
    unknown = [r for r in rows if r.split == "unknown"]

    report = {
        "manifest": args.manifest,
        "status": "valid",
        "rows": len(rows),
        "support_count": len(support),
        "known_query_count": len(known),
        "unknown_query_count": len(unknown),
        "known_labels": sorted({r.label for r in support}),
        "unknown_labels": sorted({r.label for r in unknown}),
        "physical_instance_ids": len({r.vehicle_id for r in rows}),
        "capture_groups": len({r.capture_group for r in rows}),
        "note": (
            "This checks file existence and declared metadata leakage only. "
            "It cannot detect visually duplicated photos or dishonest IDs."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
