#!/usr/bin/env python3
"""Prepare the race X-AnyLabeling dataset for YOLO OBB training.

The source dataset is never modified. Images are exposed through directory
symlinks while X-AnyLabeling JSON annotations are converted to normalized
YOLO-OBB text files under the requested output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import struct
from collections import Counter
from pathlib import Path

DEFAULT_SOURCE = Path("/mnt/shared_data/wangzijian/datasets/race_dataset")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "train_data_sea"

# Keep this order stable: class IDs are embedded in generated labels and model weights.
CLASS_NAMES = (
    "集装箱船",
    "散装货船",
    "杂货船",
    "液体运输船",
    "渔船",
    "驱逐舰",
    "护卫舰",
    "航母",
    "保障船",
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
SUPPORTED_SHAPES = {"rotation", "rectangle"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Dataset root containing train/ and test/")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Generated YOLO dataset directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of source train images used for train")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    return parser.parse_args()


def jpeg_size(path: Path) -> tuple[int, int]:
    """Read JPEG width and height without third-party image packages."""
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    with path.open("rb") as file:
        if file.read(2) != b"\xff\xd8":
            raise ValueError(f"not a JPEG file: {path}")
        while True:
            byte = file.read(1)
            while byte == b"\xff":
                byte = file.read(1)
            if not byte:
                break
            marker = byte[0]
            if marker in {0x01, *range(0xD0, 0xDA)}:
                continue
            raw_length = file.read(2)
            if len(raw_length) != 2:
                break
            length = struct.unpack(">H", raw_length)[0]
            if length < 2:
                raise ValueError(f"invalid JPEG segment length in {path}")
            if marker in sof_markers:
                payload = file.read(5)
                if len(payload) != 5:
                    break
                height, width = struct.unpack(">HH", payload[1:])
                return width, height
            file.seek(length - 2, 1)
    raise ValueError(f"could not read JPEG dimensions: {path}")


def tiff_size(path: Path, byte_order: str) -> tuple[int, int]:
    """Read classic TIFF width and height from its first image file directory."""
    endian = "<" if byte_order == "little" else ">"
    with path.open("rb") as file:
        header = file.read(8)
        if len(header) != 8 or struct.unpack(f"{endian}H", header[2:4])[0] != 42:
            raise ValueError(f"unsupported TIFF header: {path}")
        file.seek(struct.unpack(f"{endian}I", header[4:8])[0])
        raw_count = file.read(2)
        if len(raw_count) != 2:
            raise ValueError(f"invalid TIFF directory: {path}")
        entry_count = struct.unpack(f"{endian}H", raw_count)[0]
        dimensions = {}
        for _ in range(entry_count):
            entry = file.read(12)
            if len(entry) != 12:
                raise ValueError(f"truncated TIFF directory: {path}")
            tag, value_type, count = struct.unpack(f"{endian}HHI", entry[:8])
            if tag not in {256, 257} or count != 1:
                continue
            if value_type == 3:  # SHORT
                value = struct.unpack(f"{endian}H", entry[8:10])[0]
            elif value_type == 4:  # LONG
                value = struct.unpack(f"{endian}I", entry[8:12])[0]
            else:
                continue
            dimensions[tag] = value
        if 256 in dimensions and 257 in dimensions:
            return dimensions[256], dimensions[257]
    raise ValueError(f"could not read TIFF dimensions: {path}")


def image_size(path: Path) -> tuple[int, int, str]:
    """Read actual image dimensions and encoding despite misleading .jpg suffixes."""
    with path.open("rb") as file:
        header = file.read(24)
    if header.startswith(b"\xff\xd8"):
        return (*jpeg_size(path), "jpeg")
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(header) < 24 or header[12:16] != b"IHDR":
            raise ValueError(f"invalid PNG header: {path}")
        width, height = struct.unpack(">II", header[16:24])
        return width, height, "png"
    if header.startswith(b"II\x2a\x00"):
        return (*tiff_size(path, "little"), "tiff")
    if header.startswith(b"MM\x00\x2a"):
        return (*tiff_size(path, "big"), "tiff")
    raise ValueError(f"unsupported image encoding in {path}: magic={header[:12].hex()}")


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1]
            for i in range(len(points))
        )
        / 2.0
    )


def ensure_link(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"refusing to replace non-symlink path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def convert_annotation(json_path: Path, image_path: Path) -> tuple[str, Counter]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {json_path}: {error}") from error

    if data.get("imagePath") != image_path.name:
        raise ValueError(f"{json_path}: imagePath={data.get('imagePath')!r}, expected {image_path.name!r}")
    width, height, image_format = image_size(image_path)
    if (data.get("imageWidth"), data.get("imageHeight")) != (width, height):
        raise ValueError(
            f"{json_path}: declared size {(data.get('imageWidth'), data.get('imageHeight'))} "
            f"does not match JPEG size {(width, height)}"
        )

    lines = []
    stats = Counter(images=1)
    stats[f"format:{image_format}"] += 1
    for index, shape in enumerate(data.get("shapes", [])):
        label = shape.get("label")
        if label not in CLASS_TO_ID:
            raise ValueError(f"{json_path}: shape {index} has unknown label {label!r}")
        shape_type = shape.get("shape_type")
        if shape_type not in SUPPORTED_SHAPES:
            raise ValueError(f"{json_path}: shape {index} has unsupported shape_type {shape_type!r}")
        raw_points = shape.get("points")
        if not isinstance(raw_points, list) or len(raw_points) != 4:
            raise ValueError(f"{json_path}: shape {index} must contain exactly four points")

        points = []
        clipped = False
        for point in raw_points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"{json_path}: shape {index} contains an invalid point {point!r}")
            x, y = map(float, point)
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"{json_path}: shape {index} contains a non-finite point")
            clipped_x, clipped_y = min(max(x, 0.0), width), min(max(y, 0.0), height)
            clipped |= clipped_x != x or clipped_y != y
            points.append((clipped_x, clipped_y))

        if polygon_area(points) <= 1e-6:
            raise ValueError(f"{json_path}: shape {index} is degenerate after clipping")
        coordinates = [value for x, y in points for value in (x / width, y / height)]
        lines.append(" ".join([str(CLASS_TO_ID[label]), *(f"{value:.8f}" for value in coordinates)]))
        stats["objects"] += 1
        stats[f"class:{label}"] += 1
        stats[f"shape:{shape_type}"] += 1
        if clipped:
            stats["clipped_objects"] += 1

    if not lines:
        stats["empty_images"] += 1
    return ("\n".join(lines) + ("\n" if lines else "")), stats


def write_list(path: Path, split: str, images: list[Path]) -> None:
    text = "".join(f"./images/{split}/{image.name}\n" for image in images)
    path.write_text(text, encoding="utf-8")


def yaml_text() -> str:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    return (
        "# Generated by tools/prepare_sea_dataset.py.\n"
        "# X-AnyLabeling JSON is converted to normalized YOLO OBB corner coordinates.\n"
        "train: sea_train80.txt\n"
        "val: sea_val20.txt\n"
        "test: sea_test.txt\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n"
        f"{names}\n"
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    source = args.source.resolve()
    output = args.output.resolve()
    source_train, source_test = source / "train", source / "test"
    if not source_train.is_dir() or not source_test.is_dir():
        raise FileNotFoundError(f"expected train/ and test/ under {source}")

    train_images = sorted(source_train.glob("*.jpg"))
    test_images = sorted(source_test.glob("*.jpg"))
    if not train_images or not test_images:
        raise ValueError(f"no JPG images found under {source}")
    json_files = {path.stem: path for path in source_train.glob("*.json")}
    image_stems = {path.stem for path in train_images}
    missing_json = sorted(image_stems - json_files.keys())
    extra_json = sorted(json_files.keys() - image_stems)
    if missing_json or extra_json:
        raise ValueError(f"image/JSON mismatch: missing_json={missing_json[:5]}, extra_json={extra_json[:5]}")

    shuffled = train_images.copy()
    random.Random(args.seed).shuffle(shuffled)
    train_count = int(len(shuffled) * args.train_ratio)
    train_split = sorted(shuffled[:train_count])
    val_split = sorted(shuffled[train_count:])
    split_by_stem = {image.stem: "train" for image in train_split}
    split_by_stem.update({image.stem: "val" for image in val_split})

    output.mkdir(parents=True, exist_ok=True)
    ensure_link(output / "images" / "train", source_train)
    ensure_link(output / "images" / "val", source_train)
    ensure_link(output / "images" / "test", source_test)
    labels_root = output / "labels"
    if labels_root.exists():
        shutil.rmtree(labels_root)
    (labels_root / "train").mkdir(parents=True)
    (labels_root / "val").mkdir(parents=True)

    totals = Counter()
    split_totals = {"train": Counter(), "val": Counter()}
    for image_path in train_images:
        annotation, stats = convert_annotation(json_files[image_path.stem], image_path)
        split = split_by_stem[image_path.stem]
        (labels_root / split / f"{image_path.stem}.txt").write_text(annotation, encoding="utf-8")
        totals.update(stats)
        split_totals[split].update(stats)

    write_list(output / "sea_train80.txt", "train", train_split)
    write_list(output / "sea_val20.txt", "val", val_split)
    write_list(output / "sea_test.txt", "test", test_images)
    (output / "race_dataset.yaml").write_text(yaml_text(), encoding="utf-8")

    manifest = {
        "source": str(source),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "train_images": len(train_split),
        "val_images": len(val_split),
        "test_images": len(test_images),
        "objects": totals["objects"],
        "clipped_objects": totals["clipped_objects"],
        "empty_images": totals["empty_images"],
        "shape_types": {name: totals[f"shape:{name}"] for name in sorted(SUPPORTED_SHAPES)},
        "image_formats": {name: totals[f"format:{name}"] for name in ("jpeg", "png", "tiff")},
        "class_counts": {name: totals[f"class:{name}"] for name in CLASS_NAMES},
        "split_class_counts": {
            split: {name: split_totals[split][f"class:{name}"] for name in CLASS_NAMES} for split in ("train", "val")
        },
        "class_names": list(CLASS_NAMES),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Prepared dataset YAML: {output / 'race_dataset.yaml'}")


if __name__ == "__main__":
    main()
