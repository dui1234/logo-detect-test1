#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate YOLO11s 640 and 960 checkpoints on a held-out test split."
    )
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--dataset-yaml", type=Path, default=None)
    p.add_argument("--model-640", type=Path, required=True)
    p.add_argument("--model-960", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("yolo_test_results"))
    p.add_argument("--device", default="cpu", help='Examples: "cpu", "0", "cuda:0"')
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--predict-conf", type=float, default=0.25)
    p.add_argument("--skip-predictions", action="store_true")
    return p.parse_args()


def make_builtin(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): make_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_builtin(v) for v in value]
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


def find_metric(results: dict[str, Any], token: str) -> float | None:
    token = token.lower()
    for key, value in results.items():
        if token in key.lower():
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def resolve_source(root: Path, entry: Any) -> str | list[str]:
    def one(item: str) -> str:
        path = Path(item)
        return str(path if path.is_absolute() else root / path)

    if isinstance(entry, str):
        return one(entry)
    if isinstance(entry, list):
        return [one(str(item)) for item in entry]
    raise TypeError("dataset.yaml 'test' must be a string or list of strings")


def prepare_runtime_yaml(root: Path, source_yaml: Path, output: Path) -> tuple[Path, dict]:
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML: {source_yaml}")
    if "test" not in config:
        raise ValueError("dataset.yaml needs: test: images/test")
    if "names" not in config:
        raise ValueError("dataset.yaml needs a names: section")

    runtime = dict(config)
    runtime["path"] = str(root.resolve())
    output.mkdir(parents=True, exist_ok=True)
    path = output / "dataset_test_runtime.yaml"
    path.write_text(
        yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path, runtime


def evaluate_one(
    label: str,
    model_path: Path,
    imgsz: int,
    runtime_yaml: Path,
    config: dict,
    dataset_root: Path,
    output: Path,
    device: str,
    batch: int,
    workers: int,
    predict_conf: float,
    save_predictions: bool,
) -> tuple[dict, list[dict]]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("\n" + "=" * 70)
    print(f"TESTING {label}: {model_path}")
    print("=" * 70)

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(runtime_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        plots=True,
        project=str(output),
        name=f"{label}_metrics",
        exist_ok=True,
    )

    results_dict = dict(getattr(metrics, "results_dict", {}) or {})
    summary = {
        "experiment": label,
        "model_path": str(model_path.resolve()),
        "imgsz": imgsz,
        "device": device,
        "precision": find_metric(results_dict, "precision"),
        "recall": find_metric(results_dict, "recall"),
        "mAP50": float(metrics.box.map50),
        "mAP75": float(metrics.box.map75),
        "mAP50-95": float(metrics.box.map),
        "speed_ms_per_image": make_builtin(getattr(metrics, "speed", {})),
        "ultralytics_results": make_builtin(results_dict),
    }

    names = model.names
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    maps = list(metrics.box.maps)
    class_rows = []
    for class_id, class_name in sorted(names.items(), key=lambda item: int(item[0])):
        class_id = int(class_id)
        class_rows.append({
            "experiment": label,
            "class_id": class_id,
            "class_name": str(class_name),
            "mAP50-95": float(maps[class_id]) if class_id < len(maps) else None,
        })

    metrics_dir = output / f"{label}_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (metrics_dir / "per_class_map.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "class_id", "class_name", "mAP50-95"])
        writer.writeheader()
        writer.writerows(class_rows)

    if save_predictions:
        model.predict(
            source=resolve_source(dataset_root, config["test"]),
            imgsz=imgsz,
            conf=predict_conf,
            device=device,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(output),
            name=f"{label}_predictions",
            exist_ok=True,
        )

    print(json.dumps(summary, indent=2))
    return summary, class_rows


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    source_yaml = args.dataset_yaml.resolve() if args.dataset_yaml else root / "dataset.yaml"
    output = args.output_dir.resolve()

    if not root.is_dir():
        raise NotADirectoryError(root)
    if not source_yaml.is_file():
        raise FileNotFoundError(source_yaml)

    runtime_yaml, config = prepare_runtime_yaml(root, source_yaml, output)
    experiments = [
        ("yolo11s_640_exp01", args.model_640.resolve(), 640),
        ("yolo11s_960_exp02", args.model_960.resolve(), 960),
    ]

    summaries = []
    all_class_rows = []
    for label, model_path, imgsz in experiments:
        summary, rows = evaluate_one(
            label=label,
            model_path=model_path,
            imgsz=imgsz,
            runtime_yaml=runtime_yaml,
            config=config,
            dataset_root=root,
            output=output,
            device=args.device,
            batch=args.batch,
            workers=args.workers,
            predict_conf=args.predict_conf,
            save_predictions=not args.skip_predictions,
        )
        summaries.append(summary)
        all_class_rows.extend(rows)

    with (output / "model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["experiment", "model_path", "imgsz", "device", "precision", "recall", "mAP50", "mAP75", "mAP50-95"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in fields})

    with (output / "all_models_per_class_map.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "class_id", "class_name", "mAP50-95"])
        writer.writeheader()
        writer.writerows(all_class_rows)

    best = max(summaries, key=lambda row: row["mAP50-95"])
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    for row in summaries:
        print(f"{row['experiment']}: mAP50={row['mAP50']:.4f}, mAP50-95={row['mAP50-95']:.4f}")
    print(f"Best by held-out test mAP50-95: {best['experiment']}")
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
