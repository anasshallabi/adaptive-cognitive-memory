"""Demonstrate ACM evidence persistence using local SQLite and no LLM.

Run: python -m examples.durable_review
Run with retained file: python -m examples.durable_review --db review.sqlite
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from acm.durable_evidence import SQLiteEvidenceLedger
from acm.text_extraction import RuleExtractor


def demonstrate(path: Path) -> dict:
    initial = SQLiteEvidenceLedger(path)
    first = initial.observe("Zentra est une marque.", RuleExtractor(),
                            source="fictional-source-A")
    steps = [{"phase": "observation", "status": initial.query(
        "Zentra", "is_a", "marque")["status"]}]

    reopened = SQLiteEvidenceLedger(path)
    restored_status = reopened.record(first)["status"]
    reopened.decide(first, decision="accept", actor="reviewer",
                    reason="Reviewed the first source")
    steps.append({"phase": "first restart and review",
                  "restored": restored_status,
                  "status": reopened.query("Zentra", "is_a", "marque")["status"]})

    again = SQLiteEvidenceLedger(path)
    second = again.observe("Zentra n'est pas une marque.", RuleExtractor(),
                           source="fictional-source-B")
    steps.append({"phase": "opposing candidate",
                  "status": again.query("Zentra", "is_a", "marque")["status"],
                  "approved_claims": len(again.approved_memory().claims)})

    corrected = SQLiteEvidenceLedger(path)
    corrected.decide(second, decision="reject", actor="reviewer",
                     reason="The second claim was not supported")
    corrected.decide(first, decision="retract", actor="reviewer",
                     reason="The first source issued a correction")
    final = SQLiteEvidenceLedger(path)
    steps.append({"phase": "second restart and correction",
                  "status": final.query("Zentra", "is_a", "marque")["status"],
                  "approved_claims": len(final.approved_memory().claims)})
    return {
        "steps": steps,
        "events": final.history(),
        "note": "This example records reviews, not verified external truth.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ACM durable evidence example")
    parser.add_argument("--db", type=Path, default=None,
                        help="SQLite filename to keep after process exits")
    args = parser.parse_args()
    if args.db is None:
        with TemporaryDirectory() as tmp:
            result = demonstrate(Path(tmp) / "review.sqlite")
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = demonstrate(args.db)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
