#!/usr/bin/env python3
import argparse
import json
import sys

from lm_eval.utils import make_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_json", help="Path to aggregated results JSON")
    p.add_argument("--column", choices=["results", "groups"], default="results", help="Which section to render")
    p.add_argument("--sort", action="store_true", help="Sort rows by task/group name")
    args = p.parse_args()

    with open(args.results_json, "r", encoding="utf-8") as f:
        result = json.load(f)

    md = make_table(result, column=args.column, sort_results=args.sort)
    sys.stdout.write(md)


if __name__ == "__main__":
    main()
