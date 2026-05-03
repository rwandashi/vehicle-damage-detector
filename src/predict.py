"""Run damage detection on a single image.

Loads the trained YOLOv8 weights — either from a local path passed via
``--weights``, or by downloading them from a Hugging Face Hub repo (default).
Saves an annotated copy of the input next to the original and prints the
detected damage classes with confidence scores.

Usage:
    python -m src.predict --image examples/input_1.jpg
    python -m src.predict --image my_car.jpg --weights checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

# Default Hugging Face Hub location for the trained weights.
# Override with --hf-repo if you publish under a different account.
DEFAULT_HF_REPO = "rwandashi/vehicle-damage-detector"
DEFAULT_HF_FILENAME = "best.pt"


def resolve_weights(local: str | None, hf_repo: str, hf_filename: str) -> str:
    """Return a local filesystem path to the model weights.

    If ``local`` is provided and exists, use it. Otherwise download
    ``hf_filename`` from ``hf_repo`` via huggingface_hub.
    """
    if local and Path(local).is_file():
        return local

    print(f"Local weights not provided; downloading {hf_filename} from {hf_repo}...")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=hf_repo,
        filename=hf_filename,
        token=os.getenv("HF_TOKEN"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--weights", default=None,
                        help="Local .pt file. If omitted, weights are pulled from HF Hub.")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-filename", default=DEFAULT_HF_FILENAME)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for predictions.")
    parser.add_argument("--out-dir", default="runs/detect/predict",
                        help="Where to save annotated outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"ERROR: image not found: {image_path}")
        return 1

    weights = resolve_weights(args.weights, args.hf_repo, args.hf_filename)
    model = YOLO(weights)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        save=True,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
    )

    detections = results[0].boxes
    print(f"\nFound {len(detections)} damage area(s) in {image_path.name}:")
    for box in detections:
        class_name = model.names[int(box.cls)]
        confidence = float(box.conf)
        print(f"  - {class_name}: {confidence:.0%} confidence")

    annotated = Image.fromarray(results[0].plot())
    annotated_path = out_dir / f"{image_path.stem}_annotated{image_path.suffix}"
    annotated.save(annotated_path)
    print(f"\nAnnotated image saved to {annotated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())