#!/usr/bin/env python3
from mtob_grammar.config import models, output_path, sources
from mtob_grammar.experiment import CONDITIONS, output_paths


def main() -> None:
    print("MODEL\tSOURCE\tCONDITION\tRESULT\tMETRICS")
    count = 0
    for model in models():
        for source in sources():
            for condition in CONDITIONS:
                result, metric = output_paths(model, source, condition)
                result_status = "complete" if result.is_file() else "pending"
                metric_status = "complete" if metric.is_file() else "pending"
                print(f"{model}\t{source}\t{condition}\t{result_status}\t{metric_status}")
                count += 1
    print(f"\nTotal: {count} experiments")


if __name__ == "__main__":
    main()
