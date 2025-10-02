#!/usr/bin/env python3
"""Command-line entry point for zero-shot evaluation tasks."""

import fire

from src.zero_shot_eval import ZeroShotEvalCLI


def main() -> None:
    fire.Fire(ZeroShotEvalCLI)


if __name__ == "__main__":
    main()
