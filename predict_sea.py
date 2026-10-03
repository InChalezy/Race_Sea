#!/usr/bin/env python3
"""Run YOLO-OBB inference and export the race competition JSON format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

EXPECTED_CLASSES = {
    "集装箱船",
    "散装货船",
    "杂货船",
    "液体运输船",
    "渔船",
    "驱逐舰",
    "护卫舰",
    "航母",
    "保障船",
}


def parse_bool(value: str) -> bool:
    """Parse CLI booleans using the same True/False spelling as the project entrypoints."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trained YOLO-OBB checkpoint, e.g. runs/.../weights/best.pt")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/mnt/shared_data/wangzijian/datasets/race_dataset/test"),
        help="Directory containing competition test JPG images",
    )
    parser.add_argument("--output", type=Path, default=Path("result_team.json"), help="Submission JSON path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001, help="Low threshold is recommended for AP evaluation")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--end2end",
        type=parse_bool,
        default=False,
        metavar="BOOL",
        help="Use the YOLO26 end-to-end head; False applies rotated NMS with --iou",
    )
    parser.add_argument("--decimals", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(f"test image directory not found: {args.source}")
    image_paths = sorted(args.source.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"no JPG images found in {args.source}")

    model = YOLO(args.model, task="obb")
    model_names = {str(name) for name in model.names.values()}
    if model_names != EXPECTED_CLASSES:
        raise ValueError(
            "checkpoint class names do not match the competition classes: "
            f"expected={sorted(EXPECTED_CLASSES)}, actual={sorted(model_names)}"
        )

    predictions_by_image: dict[str, list[dict]] = {}
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        end2end=args.end2end,
        stream=True,
        verbose=False,
    )
    for result in results:
        image_name = Path(result.path).name
        if image_name in predictions_by_image:
            raise RuntimeError(f"duplicate prediction result for {image_name}")
        predictions = []
        obb = result.obb
        if obb is None:
            raise RuntimeError(f"model did not return OBB predictions for {image_name}")
        height, width = result.orig_shape
        for points, class_id, score in zip(obb.xyxyxyxy.cpu(), obb.cls.cpu(), obb.conf.cpu()):
            clipped_points = [
                [
                    round(min(max(float(point[0]), 0.0), float(width)), args.decimals),
                    round(min(max(float(point[1]), 0.0), float(height)), args.decimals),
                ]
                for point in points
            ]
            predictions.append(
                {
                    "category": model.names[int(class_id)],
                    "points": clipped_points,
                    "score": round(float(score), 6),
                }
            )
        predictions.sort(key=lambda item: item["score"], reverse=True)
        predictions_by_image[image_name] = predictions

    expected_names = {path.name for path in image_paths}
    missing = sorted(expected_names - predictions_by_image.keys())
    unexpected = sorted(predictions_by_image.keys() - expected_names)
    if missing or unexpected:
        raise RuntimeError(f"inference result mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")

    submission = [{"image_id": path.name, "predictions": predictions_by_image[path.name]} for path in image_paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    detection_count = sum(len(item["predictions"]) for item in submission)
    print(f"Wrote {len(submission)} images and {detection_count} detections to {args.output}")


if __name__ == "__main__":
    main()
