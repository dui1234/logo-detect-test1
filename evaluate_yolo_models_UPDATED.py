#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate YOLO11s 640 and 960 models on the labeled test split "
            "and measure inference, pipeline, and end-to-end FPS."
        )
    )

    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, default=None)
    parser.add_argument("--model-640", type=Path, required=True)
    parser.add_argument("--model-960", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("yolo_test_results"),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Examples: "cpu", "0", or "cuda:0".',
    )

    parser.add_argument(
        "--val-batch",
        "--batch",
        dest="val_batch",
        type=int,
        default=4,
        help="Batch size used for model.val().",
    )
    parser.add_argument(
        "--benchmark-batch",
        type=int,
        default=1,
        help="Batch size used for FPS benchmarking.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=5,
        help="Number of complete timed passes over all test images.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of untimed warm-up runs.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--predict-conf", type=float, default=0.50)
    parser.add_argument("--predict-iou", type=float, default=0.45)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")

    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    """Convert Torch/NumPy values into JSON-safe Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): to_builtin(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]

    if hasattr(value, "tolist"):
        return value.tolist()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def find_metric(
    results_dict: dict[str, Any],
    token: str,
) -> float | None:
    """Find a metric such as precision or recall by partial key."""
    token = token.lower()

    for key, value in results_dict.items():
        if token in key.lower():
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

    return None


def prepare_runtime_yaml(
    dataset_root: Path,
    source_yaml: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Create a temporary YAML whose path points to this local dataset."""
    config = yaml.safe_load(
        source_yaml.read_text(encoding="utf-8")
    )

    if not isinstance(config, dict):
        raise ValueError(f"Invalid dataset YAML: {source_yaml}")

    if "test" not in config:
        raise ValueError(
            "dataset.yaml must contain a test split, for example:\n"
            "test: images/test"
        )

    if "names" not in config:
        raise ValueError(
            "dataset.yaml must contain a names: section."
        )

    runtime_config = dict(config)
    runtime_config["path"] = str(dataset_root.resolve())

    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_yaml = output_dir / "dataset_test_runtime.yaml"
    runtime_yaml.write_text(
        yaml.safe_dump(
            runtime_config,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    return runtime_yaml, runtime_config


def resolve_path(
    dataset_root: Path,
    path_value: str,
) -> Path:
    path = Path(path_value)

    if not path.is_absolute():
        path = dataset_root / path

    return path.resolve()


def collect_test_images(
    dataset_root: Path,
    test_entry: Any,
) -> list[Path]:
    """
    Convert dataset.yaml test entry into a list of image paths.

    Supports:
    - test: images/test
    - test: /absolute/path/to/test
    - test: [images/test1, images/test2]
    - test: test_images.txt
    """
    entries = (
        test_entry
        if isinstance(test_entry, list)
        else [test_entry]
    )

    images: list[Path] = []

    for entry in entries:
        if not isinstance(entry, str):
            raise TypeError(
                "dataset.yaml 'test' must be a string "
                "or a list of strings."
            )

        path = resolve_path(dataset_root, entry)

        if path.is_dir():
            images.extend(
                sorted(
                    image_path
                    for image_path in path.rglob("*")
                    if (
                        image_path.is_file()
                        and image_path.suffix.lower()
                        in IMAGE_EXTENSIONS
                    )
                )
            )
            continue

        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(path)
            continue

        if path.is_file() and path.suffix.lower() == ".txt":
            for raw_line in path.read_text(
                encoding="utf-8"
            ).splitlines():
                line = raw_line.strip()

                if not line or line.startswith("#"):
                    continue

                image_path = Path(line)

                if not image_path.is_absolute():
                    image_path = (
                        path.parent / image_path
                    ).resolve()

                if (
                    image_path.is_file()
                    and image_path.suffix.lower()
                    in IMAGE_EXTENSIONS
                ):
                    images.append(image_path)

            continue

        raise FileNotFoundError(
            f"Could not resolve test source: {path}"
        )

    unique_images = sorted(
        {image.resolve() for image in images}
    )

    if not unique_images:
        raise RuntimeError(
            "No supported images found in the test split."
        )

    return unique_images


def cuda_sync_needed(device: str) -> bool:
    return (
        device.lower() != "cpu"
        and torch.cuda.is_available()
    )


def consume_results(results: Any) -> int:
    """Consume a streamed prediction generator and count images."""
    count = 0

    for _ in results:
        count += 1

    return count


def benchmark_end_to_end(
    *,
    model: YOLO,
    image_paths: list[Path],
    imgsz: int,
    device: str,
    batch: int,
    repeats: int,
    warmup_runs: int,
    conf: float,
    iou: float,
    agnostic_nms: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Benchmark complete prediction throughput.

    Includes:
    - image reading;
    - preprocessing;
    - inference;
    - NMS/postprocessing;
    - Python result iteration.

    Excludes:
    - model loading;
    - warm-up;
    - drawing boxes;
    - writing output images.
    """
    if batch < 1:
        raise ValueError(
            "--benchmark-batch must be at least 1."
        )

    if repeats < 1:
        raise ValueError(
            "--benchmark-repeats must be at least 1."
        )

    if warmup_runs < 0:
        raise ValueError(
            "--warmup-runs cannot be negative."
        )

    sources = [str(path) for path in image_paths]

    warmup_count = min(
        len(sources),
        max(1, batch),
    )
    warmup_sources = sources[:warmup_count]

    print("\nEnd-to-end FPS benchmark")
    print(f"  Images per pass:  {len(sources)}")
    print(f"  Benchmark batch:  {batch}")
    print(f"  Warm-up runs:     {warmup_runs}")
    print(f"  Timed repeats:    {repeats}")

    for warmup_index in range(
        1,
        warmup_runs + 1,
    ):
        warmup_results = model.predict(
            source=warmup_sources,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            batch=min(batch, warmup_count),
            agnostic_nms=agnostic_nms,
            stream=True,
            save=False,
            verbose=False,
        )

        consume_results(warmup_results)

        print(
            f"  Warm-up "
            f"{warmup_index}/{warmup_runs} complete"
        )

    if cuda_sync_needed(device):
        torch.cuda.synchronize()

    run_rows: list[dict[str, Any]] = []

    for repeat_index in range(
        1,
        repeats + 1,
    ):
        if cuda_sync_needed(device):
            torch.cuda.synchronize()

        started = time.perf_counter()

        prediction_results = model.predict(
            source=sources,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            batch=batch,
            agnostic_nms=agnostic_nms,
            stream=True,
            save=False,
            verbose=False,
        )

        processed = consume_results(
            prediction_results
        )

        if cuda_sync_needed(device):
            torch.cuda.synchronize()

        elapsed_seconds = (
            time.perf_counter() - started
        )

        ms_per_image = (
            elapsed_seconds * 1000.0 / processed
            if processed > 0
            else 0.0
        )

        fps = (
            processed / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        )

        row = {
            "repeat": repeat_index,
            "image_count": processed,
            "elapsed_seconds": elapsed_seconds,
            "ms_per_image": ms_per_image,
            "fps": fps,
        }
        run_rows.append(row)

        print(
            f"  Pass {repeat_index}/{repeats}: "
            f"{elapsed_seconds:.4f} s total, "
            f"{ms_per_image:.3f} ms/image, "
            f"{fps:.2f} FPS"
        )

    total_seconds_values = [
        float(row["elapsed_seconds"])
        for row in run_rows
    ]

    ms_values = [
        float(row["ms_per_image"])
        for row in run_rows
    ]

    fps_values = [
        float(row["fps"])
        for row in run_rows
    ]

    mean_total_seconds = statistics.mean(
        total_seconds_values
    )
    mean_ms_per_image = statistics.mean(
        ms_values
    )
    mean_fps = statistics.mean(
        fps_values
    )

    fps_std = (
        statistics.stdev(fps_values)
        if len(fps_values) > 1
        else 0.0
    )

    benchmark_summary = {
        "benchmark_image_count": len(sources),
        "benchmark_batch": batch,
        "benchmark_repeats": repeats,
        "benchmark_warmup_runs": warmup_runs,
        "benchmark_mean_total_seconds":
            mean_total_seconds,
        "benchmark_mean_ms_per_image":
            mean_ms_per_image,
        "benchmark_mean_fps":
            mean_fps,
        "benchmark_fps_std":
            fps_std,

        # Friendly aliases used in the final CSV.
        "end_to_end_mean_total_seconds":
            mean_total_seconds,
        "end_to_end_mean_ms_per_image":
            mean_ms_per_image,
        "end_to_end_mean_fps":
            mean_fps,
        "end_to_end_fps_std":
            fps_std,
    }

    return benchmark_summary, run_rows


def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_one(
    *,
    label: str,
    model_path: Path,
    imgsz: int,
    runtime_yaml: Path,
    runtime_config: dict[str, Any],
    dataset_root: Path,
    output_dir: Path,
    device: str,
    val_batch: int,
    benchmark_batch: int,
    benchmark_repeats: int,
    warmup_runs: int,
    workers: int,
    predict_conf: float,
    predict_iou: float,
    agnostic_nms: bool,
    save_predictions: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    print("\n" + "=" * 78)
    print(f"TESTING {label}")
    print("=" * 78)
    print(f"Model:             {model_path.resolve()}")
    print(f"Evaluation imgsz:  {imgsz}")
    print(f"Device:            {device}")
    print(f"Validation batch:  {val_batch}")
    print(f"Confidence:        {predict_conf}")
    print(f"NMS IoU:           {predict_iou}")
    print(f"Agnostic NMS:      {agnostic_nms}")

    model = YOLO(str(model_path))

    # --------------------------------------------------------
    # 1. Formal test metrics
    # --------------------------------------------------------
    metrics = model.val(
        data=str(runtime_yaml),
        split="test",
        imgsz=imgsz,
        batch=val_batch,
        device=device,
        workers=workers,
        plots=True,
        project=str(output_dir),
        name=f"{label}_metrics",
        exist_ok=True,
        verbose=True,
    )

    results_dict = dict(
        getattr(
            metrics,
            "results_dict",
            {},
        )
        or {}
    )

    speed = dict(
        getattr(
            metrics,
            "speed",
            {},
        )
        or {}
    )

    preprocess_ms = float(
        speed.get("preprocess", 0.0)
        or 0.0
    )
    inference_ms = float(
        speed.get("inference", 0.0)
        or 0.0
    )
    postprocess_ms = float(
        speed.get("postprocess", 0.0)
        or 0.0
    )

    pipeline_ms = (
        preprocess_ms
        + inference_ms
        + postprocess_ms
    )

    inference_fps = (
        1000.0 / inference_ms
        if inference_ms > 0
        else None
    )

    pipeline_fps = (
        1000.0 / pipeline_ms
        if pipeline_ms > 0
        else None
    )

    # --------------------------------------------------------
    # 2. Complete-folder FPS benchmark
    # --------------------------------------------------------
    test_images = collect_test_images(
        dataset_root,
        runtime_config["test"],
    )

    benchmark_summary, benchmark_rows = (
        benchmark_end_to_end(
            model=model,
            image_paths=test_images,
            imgsz=imgsz,
            device=device,
            batch=benchmark_batch,
            repeats=benchmark_repeats,
            warmup_runs=warmup_runs,
            conf=predict_conf,
            iou=predict_iou,
            agnostic_nms=agnostic_nms,
        )
    )

    summary: dict[str, Any] = {
        "experiment": label,
        "model_path": str(
            model_path.resolve()
        ),
        "imgsz": imgsz,
        "device": device,
        "val_batch": val_batch,
        "predict_conf": predict_conf,
        "predict_iou": predict_iou,
        "agnostic_nms": agnostic_nms,

        "precision": find_metric(
            results_dict,
            "precision",
        ),
        "recall": find_metric(
            results_dict,
            "recall",
        ),
        "mAP50": float(
            metrics.box.map50
        ),
        "mAP75": float(
            metrics.box.map75
        ),
        "mAP50-95": float(
            metrics.box.map
        ),

        "preprocess_ms_per_image":
            preprocess_ms,
        "inference_ms_per_image":
            inference_ms,
        "postprocess_ms_per_image":
            postprocess_ms,
        "pipeline_ms_per_image":
            pipeline_ms,

        "inference_fps":
            inference_fps,
        "pipeline_fps":
            pipeline_fps,

        **benchmark_summary,

        "ultralytics_speed":
            to_builtin(speed),
        "ultralytics_results":
            to_builtin(results_dict),
    }

    names = model.names

    if isinstance(names, list):
        names = {
            index: class_name
            for index, class_name
            in enumerate(names)
        }

    class_maps = list(
        metrics.box.maps
    )

    class_rows: list[
        dict[str, Any]
    ] = []

    for class_id, class_name in sorted(
        names.items(),
        key=lambda item: int(item[0]),
    ):
        numeric_id = int(class_id)

        class_rows.append(
            {
                "experiment": label,
                "class_id": numeric_id,
                "class_name":
                    str(class_name),
                "mAP50-95": (
                    float(
                        class_maps[
                            numeric_id
                        ]
                    )
                    if numeric_id
                    < len(class_maps)
                    else None
                ),
            }
        )

    metrics_dir = (
        output_dir
        / f"{label}_metrics"
    )
    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        metrics_dir
        / "test_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    save_csv(
        metrics_dir
        / "per_class_map.csv",
        class_rows,
        [
            "experiment",
            "class_id",
            "class_name",
            "mAP50-95",
        ],
    )

    save_csv(
        metrics_dir
        / "benchmark_runs.csv",
        benchmark_rows,
        [
            "repeat",
            "image_count",
            "elapsed_seconds",
            "ms_per_image",
            "fps",
        ],
    )

    # --------------------------------------------------------
    # 3. Optional annotated images, outside benchmark timing
    # --------------------------------------------------------
    if save_predictions:
        model.predict(
            source=[
                str(path)
                for path
                in test_images
            ],
            imgsz=imgsz,
            conf=predict_conf,
            iou=predict_iou,
            device=device,
            batch=benchmark_batch,
            agnostic_nms=agnostic_nms,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(output_dir),
            name=(
                f"{label}_predictions"
            ),
            exist_ok=True,
            verbose=True,
        )

    print("\nResult summary")
    print(
        f"  Precision:                  "
        f"{summary['precision']}"
    )
    print(
        f"  Recall:                     "
        f"{summary['recall']}"
    )
    print(
        f"  mAP50:                      "
        f"{summary['mAP50']:.6f}"
    )
    print(
        f"  mAP50-95:                   "
        f"{summary['mAP50-95']:.6f}"
    )

    if inference_fps is not None:
        print(
            f"  Inference-only FPS:         "
            f"{inference_fps:.2f}"
        )
    else:
        print(
            "  Inference-only FPS:         "
            "N/A"
        )

    if pipeline_fps is not None:
        print(
            f"  Pre+infer+post FPS:         "
            f"{pipeline_fps:.2f}"
        )
    else:
        print(
            "  Pre+infer+post FPS:         "
            "N/A"
        )

    print(
        f"  End-to-end folder FPS:      "
        f"{benchmark_summary['end_to_end_mean_fps']:.2f} "
        f"± "
        f"{benchmark_summary['end_to_end_fps_std']:.2f}"
    )

    print(
        f"  Mean complete-folder time:  "
        f"{benchmark_summary['end_to_end_mean_total_seconds']:.4f} s "
        f"for "
        f"{benchmark_summary['benchmark_image_count']} images"
    )

    print(
        f"  Saved summary:              "
        f"{metrics_dir / 'test_summary.json'}"
    )

    print(
        f"  Saved benchmark passes:     "
        f"{metrics_dir / 'benchmark_runs.csv'}"
    )

    return summary, class_rows


def main() -> None:
    args = parse_args()

    dataset_root = (
        args.dataset_root.resolve()
    )

    source_yaml = (
        args.dataset_yaml.resolve()
        if args.dataset_yaml is not None
        else (
            dataset_root
            / "dataset.yaml"
        )
    )

    output_dir = (
        args.output_dir.resolve()
    )

    if not dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset root not found: "
            f"{dataset_root}"
        )

    if not source_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: "
            f"{source_yaml}"
        )

    runtime_yaml, runtime_config = (
        prepare_runtime_yaml(
            dataset_root,
            source_yaml,
            output_dir,
        )
    )

    experiments = [
        (
            "yolo11s_640_exp01",
            args.model_640.resolve(),
            640,
        ),
        (
            "yolo11s_960_exp02",
            args.model_960.resolve(),
            960,
        ),
    ]

    summaries: list[
        dict[str, Any]
    ] = []

    all_class_rows: list[
        dict[str, Any]
    ] = []

    for (
        label,
        model_path,
        imgsz,
    ) in experiments:
        summary, class_rows = (
            evaluate_one(
                label=label,
                model_path=model_path,
                imgsz=imgsz,
                runtime_yaml=
                    runtime_yaml,
                runtime_config=
                    runtime_config,
                dataset_root=
                    dataset_root,
                output_dir=
                    output_dir,
                device=args.device,
                val_batch=
                    args.val_batch,
                benchmark_batch=
                    args.benchmark_batch,
                benchmark_repeats=
                    args.benchmark_repeats,
                warmup_runs=
                    args.warmup_runs,
                workers=
                    args.workers,
                predict_conf=
                    args.predict_conf,
                predict_iou=
                    args.predict_iou,
                agnostic_nms=
                    args.agnostic_nms,
                save_predictions=
                    not args.skip_predictions,
            )
        )

        summaries.append(summary)
        all_class_rows.extend(
            class_rows
        )

    comparison_fields = [
        "experiment",
        "model_path",
        "imgsz",
        "device",
        "val_batch",
        "predict_conf",
        "predict_iou",
        "agnostic_nms",
        "precision",
        "recall",
        "mAP50",
        "mAP75",
        "mAP50-95",
        "preprocess_ms_per_image",
        "inference_ms_per_image",
        "postprocess_ms_per_image",
        "pipeline_ms_per_image",
        "inference_fps",
        "pipeline_fps",
        "benchmark_image_count",
        "benchmark_batch",
        "benchmark_repeats",
        "benchmark_warmup_runs",
        "benchmark_mean_total_seconds",
        "benchmark_mean_ms_per_image",
        "benchmark_mean_fps",
        "benchmark_fps_std",
        "end_to_end_mean_total_seconds",
        "end_to_end_mean_ms_per_image",
        "end_to_end_mean_fps",
        "end_to_end_fps_std",
    ]

    comparison_path = (
        output_dir
        / "model_comparison.csv"
    )

    save_csv(
        comparison_path,
        [
            {
                key: row.get(key)
                for key
                in comparison_fields
            }
            for row
            in summaries
        ],
        comparison_fields,
    )

    save_csv(
        output_dir
        / "all_models_per_class_map.csv",
        all_class_rows,
        [
            "experiment",
            "class_id",
            "class_name",
            "mAP50-95",
        ],
    )

    best_accuracy = max(
        summaries,
        key=lambda row:
            float(
                row["mAP50-95"]
            ),
    )

    fastest = max(
        summaries,
        key=lambda row:
            float(
                row[
                    "end_to_end_mean_fps"
                ]
            ),
    )

    print("\n" + "=" * 78)
    print("FINAL COMPARISON")
    print("=" * 78)

    for row in summaries:
        inference_fps_text = (
            f"{row['inference_fps']:.2f}"
            if (
                row["inference_fps"]
                is not None
            )
            else "N/A"
        )

        pipeline_fps_text = (
            f"{row['pipeline_fps']:.2f}"
            if (
                row["pipeline_fps"]
                is not None
            )
            else "N/A"
        )

        print(
            f"{row['experiment']}: "
            f"mAP50-95="
            f"{row['mAP50-95']:.4f}, "
            f"inference FPS="
            f"{inference_fps_text}, "
            f"pipeline FPS="
            f"{pipeline_fps_text}, "
            f"end-to-end FPS="
            f"{row['end_to_end_mean_fps']:.2f} "
            f"± "
            f"{row['end_to_end_fps_std']:.2f}"
        )

    print(
        f"\nBest accuracy: "
        f"{best_accuracy['experiment']}"
    )

    print(
        f"Fastest end-to-end: "
        f"{fastest['experiment']}"
    )

    print(
        f"Comparison CSV: "
        f"{comparison_path}"
    )

    print(
        f"All outputs: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
