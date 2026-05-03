"""Fine-tune YOLOv8 on the Car Damage Detection dataset.

Reads hyperparameters from a YAML config (default ``configs/train.yaml``).
Any config field can be overridden on the command line, e.g.
``--epochs 100`` or ``--batch 8``.

Usage:
    python -m src.train
    python -m src.train --config configs/train.yaml --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    # Optional CLI overrides — any of these take precedence over the YAML.
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--name")
    parser.add_argument("--project")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config))

    # Apply CLI overrides
    for key in ("model", "data", "epochs", "imgsz", "batch", "patience",
                "name", "project", "device", "seed"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    print("Training with config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    model = YOLO(cfg["model"])
    model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        patience=cfg.get("patience", 10),
        name=cfg.get("name", "car_damage_v1"),
        project=cfg.get("project", "runs/detect"),
        plots=cfg.get("plots", True),
        seed=cfg.get("seed", 0),
        device=cfg.get("device", None),
    )

    weights_path = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {weights_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())