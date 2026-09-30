"""Run without dependencies: python -m examples.text_one_shot"""
import argparse
import json

from acm.text_memory import TextMemory


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled-text one-shot memory demonstration")
    parser.add_argument(
        "--fact", action="append", default=[],
        help="Fact to learn (repeatable); only documented grammar is accepted"
    )
    parser.add_argument(
        "--ask", action="append", default=[],
        help="Question to answer from stored claims (repeatable)"
    )
    parser.add_argument("--source", default="demo", help="Origin ID for the supplied claims")
    args = parser.parse_args()

    memory = TextMemory()
    facts = args.fact or [
        "Zentra est une marque automobile.",
        "Une marque automobile est une organisation.",
        "Zentra fabrique des voitures électriques.",
    ]
    questions = args.ask or [
        "Est-ce que Zentra est une organisation ?",
        "Que sais-tu de Zentra ?",
        "Est-ce que Zentra est une banque ?",
    ]
    learned = [memory.learn_text(fact, source=args.source) for fact in facts]
    answers = [memory.ask(question) for question in questions]
    print(json.dumps({"learned": learned, "answers": answers}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
