#!/usr/bin/env python3
import argparse
import json

from mtob_grammar.config import models, sources
from mtob_grammar.experiment import CONDITIONS, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one paper-aligned grammar experiment.")
    parser.add_argument("--model", required=True, choices=models())
    parser.add_argument("--source", required=True, choices=sources())
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N configured test examples.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.model,
                args.source,
                args.condition,
                smoke=args.smoke,
                limit=args.limit,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
