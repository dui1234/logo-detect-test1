#!/usr/bin/env python3
"""
Create a normalized YOLO detection test dataset by letterboxing every image
to a fixed square size and updating the YOLO bounding-box labels.

Expected input:
    images/test/
    labels/test/

Generated output:
    <output-root>/images/test/
    <output-root>/labels/test/
    <output-root>/normalization_manifest.csv

YOLO label format:
    class_id x_center y_center width height
where coordinates are normalized to [0, 1].
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Letterbox a YOLO test dataset to a fixed square size and "
            "transform its detection labels."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Input image directory, e.g. bank_logo_dataset/images/test",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Input YOLO label directory, e.g. bank_logo_dataset/labels/test",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output dataset root, e.g. bank_logo_dataset_normalized_960",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=960,
        help="Output width and height. Default: 960",
    )
    parser.add_argument(
        "--pad-value",
        type=int,
        default=114,
        help="Gray padding value from 0 to 255. Default: 114",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG output quality from 1 to 100. Default: 95",
    )
    parser.add_argument(
        "--no-scale-up",
        action="store_true",
        help=(
            "Do not enlarge images smaller than the target size. "
            "By default, small images are enlarged to match standard YOLO letterboxing."
        ),
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the output root before generating the normalized dataset.",
    )
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def letterbox(
    image: Image.Image,
    size: int,
    pad_value: int,
    scale_up: bool,
) -> tuple[Image.Image, float, int, int]:
    """Resize while preserving aspect ratio, then pad to size x size."""
    original_width, original_height = image.size

    scale = min(size / original_width, size / original_height)
    if not scale_up:
        scale = min(scale, 1.0)

    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))

    resized = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.LANCZOS,
    )

    pad_width = size - resized_width
    pad_height = size - resized_height

    pad_left = pad_width // 2
    pad_top = pad_height // 2

    canvas = Image.new(
        "RGB",
        (size, size),
        color=(pad_value, pad_value, pad_value),
    )
    canvas.paste(resized, (pad_left, pad_top))

    return canvas, scale, pad_left, pad_top


def transform_yolo_labels(
    label_path: Path,
    original_width: int,
    original_height: int,
    scale: float,
    pad_left: int,
    pad_top: int,
    output_size: int,
) -> tuple[list[str], int, int]:
    """
    Transform YOLO detection labels after resize + padding.

    Returns:
        output_lines, input_box_count, output_box_count
    """
    if not label_path.exists():
        return [], 0, 0

    raw_lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    output_lines: list[str] = []
    input_count = 0

    for line_number, line in enumerate(raw_lines, start=1):
        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_number}: expected 5 values for a YOLO "
                f"detection label, but found {len(parts)}. Line: {line!r}"
            )

        class_id = parts[0]

        try:
            x_center, y_center, box_width, box_height = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(
                f"{label_path}:{line_number}: invalid numeric label values."
            ) from exc

        input_count += 1

        # Convert normalized original-image coordinates to pixel corners.
        x1 = (x_center - box_width / 2.0) * original_width
        y1 = (y_center - box_height / 2.0) * original_height
        x2 = (x_center + box_width / 2.0) * original_width
        y2 = (y_center + box_height / 2.0) * original_height

        # Apply the same resize and padding as the image.
        x1 = x1 * scale + pad_left
        y1 = y1 * scale + pad_top
        x2 = x2 * scale + pad_left
        y2 = y2 * scale + pad_top

        # Clip boxes to the normalized image.
        x1 = min(max(x1, 0.0), float(output_size))
        y1 = min(max(y1, 0.0), float(output_size))
        x2 = min(max(x2, 0.0), float(output_size))
        y2 = min(max(y2, 0.0), float(output_size))

        new_width = x2 - x1
        new_height = y2 - y1

        # Ignore invalid or zero-area boxes.
        if new_width <= 0.0 or new_height <= 0.0:
            continue

        new_x_center = ((x1 + x2) / 2.0) / output_size
        new_y_center = ((y1 + y2) / 2.0) / output_size
        new_box_width = new_width / output_size
        new_box_height = new_height / output_size

        output_lines.append(
            f"{class_id} "
            f"{new_x_center:.8f} "
            f"{new_y_center:.8f} "
            f"{new_box_width:.8f} "
            f"{new_box_height:.8f}"
        )

    return output_lines, input_count, len(output_lines)


def save_jpeg(image: Image.Image, output_path: Path, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        output_path,
        format="JPEG",
        quality=quality,
        optimize=False,
        progressive=False,
        subsampling=0,
    )


def main() -> int:
    args = parse_args()

    if args.size <= 0:
        raise ValueError("--size must be greater than zero.")
    if not 0 <= args.pad_value <= 255:
        raise ValueError("--pad-value must be between 0 and 255.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.images_dir}")
    if not args.labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {args.labels_dir}")

    if args.clean_output and args.output_root.exists():
        shutil.rmtree(args.output_root)

    output_images_dir = args.output_root / "images" / "test"
    output_labels_dir = args.output_root / "labels" / "test"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.images_dir)
    if not images:
        raise RuntimeError(f"No supported images found in: {args.images_dir}")

    # Because every output is JPEG, duplicate stems would overwrite each other.
    stems: dict[str, Path] = {}
    for image_path in images:
        stem = image_path.stem
        if stem in stems:
            raise RuntimeError(
                "Two source images have the same stem and would collide after "
                f"conversion to JPEG:\n  {stems[stem]}\n  {image_path}"
            )
        stems[stem] = image_path

    manifest_path = args.output_root / "normalization_manifest.csv"

    total_input_boxes = 0
    total_output_boxes = 0
    missing_label_files = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "source_image",
                "output_image",
                "original_width",
                "original_height",
                "output_width",
                "output_height",
                "scale",
                "pad_left",
                "pad_top",
                "input_boxes",
                "output_boxes",
            ],
        )
        writer.writeheader()

        for index, image_path in enumerate(images, start=1):
            relative_image = image_path.relative_to(args.images_dir)
            source_label_path = args.labels_dir / relative_image.with_suffix(".txt")

            if not source_label_path.exists():
                missing_label_files += 1

            with Image.open(image_path) as opened:
                # Correct camera rotation stored in EXIF before reading dimensions.
                image = ImageOps.exif_transpose(opened).convert("RGB")
                original_width, original_height = image.size

                normalized, scale, pad_left, pad_top = letterbox(
                    image=image,
                    size=args.size,
                    pad_value=args.pad_value,
                    scale_up=not args.no_scale_up,
                )

            output_image_path = output_images_dir / f"{image_path.stem}.jpg"
            output_label_path = output_labels_dir / f"{image_path.stem}.txt"

            save_jpeg(normalized, output_image_path, args.jpeg_quality)

            transformed_lines, input_boxes, output_boxes = transform_yolo_labels(
                label_path=source_label_path,
                original_width=original_width,
                original_height=original_height,
                scale=scale,
                pad_left=pad_left,
                pad_top=pad_top,
                output_size=args.size,
            )

            output_label_path.write_text(
                "\n".join(transformed_lines) + ("\n" if transformed_lines else ""),
                encoding="utf-8",
            )

            total_input_boxes += input_boxes
            total_output_boxes += output_boxes

            writer.writerow(
                {
                    "source_image": str(image_path),
                    "output_image": str(output_image_path),
                    "original_width": original_width,
                    "original_height": original_height,
                    "output_width": args.size,
                    "output_height": args.size,
                    "scale": f"{scale:.10f}",
                    "pad_left": pad_left,
                    "pad_top": pad_top,
                    "input_boxes": input_boxes,
                    "output_boxes": output_boxes,
                }
            )

            print(
                f"[{index:>5}/{len(images)}] "
                f"{image_path.name} -> {output_image_path.name}"
            )

    print("\nNormalization complete")
    print(f"Images written       : {len(images)}")
    print(f"Input boxes          : {total_input_boxes}")
    print(f"Output boxes         : {total_output_boxes}")
    print(f"Missing label files  : {missing_label_files}")
    print(f"Normalized images    : {output_images_dir}")
    print(f"Normalized labels    : {output_labels_dir}")
    print(f"Manifest             : {manifest_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
