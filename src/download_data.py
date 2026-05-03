"""Download the Car Damage Detection dataset from Roboflow.

Reads the API key from the ``ROBOFLOW_API_KEY`` environment variable (loaded
from ``.env`` if present). Writes the dataset into ``./datasets/`` by default.

Usage:
    python -m src.download_data
    python -m src.download_data --output-dir datasets/car-damage
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

# Defaults match the public Roboflow project this model was trained on.
DEFAULT_WORKSPACE = "rwandashis-workspace"
DEFAULT_PROJECT = "car-damage-detection-5ioys-f4qah"
DEFAULT_VERSION = 1
DEFAULT_FORMAT = "yolov8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument("--format", default=DEFAULT_FORMAT, dest="export_format")
    parser.add_argument(
        "--output-dir",
        default="datasets/car-damage",
        help="Directory the dataset will be written to (created if missing).",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print(
            "ERROR: ROBOFLOW_API_KEY is not set. Copy .env.example to .env and "
            "fill it in, or export the variable in your shell.",
            file=sys.stderr,
        )
        return 1

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to Roboflow workspace '{args.workspace}'...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)

    print(f"Downloading version {args.version} as '{args.export_format}'...")
    cwd = os.getcwd()
    os.chdir(output_dir.parent)
    try:
        version.download(args.export_format, location=output_dir.name)
    finally:
        os.chdir(cwd)

    print(f"Dataset ready at: {output_dir.resolve()}")
    print("Contents:", sorted(p.name for p in output_dir.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())