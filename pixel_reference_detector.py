"""離線、像素參考式影像異常偵測器。

此方法逐像素建立正常影像的平均值與標準差，推論時以標準化像素差異
產生異常分數。適用於相機、產品位置和影像對位固定的檢查情境。

使用方式：
    python pixel_reference_detector.py demo
    python pixel_reference_detector.py train --data-dir data/normal
    python pixel_reference_detector.py predict --model-path artifacts/pixel_reference.pt --image sample.png
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_IMAGE_SIZE = 128
STD_FLOOR = 0.02
SCORE_QUANTILE = 0.99


def _image_paths(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"找不到影像資料夾：{folder}")
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _validate_image_size(image_size: int) -> None:
    if image_size < 16 or image_size > 1024:
        raise ValueError("--image-size 必須介於 16 到 1024。")


def _load_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        fitted = ImageOps.fit(image.convert("RGB"), (image_size, image_size))
        return np.asarray(fitted, dtype=np.float32) / 255.0


def _score_map(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """計算 RGB 通道平均的標準化絕對差異圖。"""
    return np.mean(np.abs(image - mean) / np.maximum(std, STD_FLOOR), axis=2)


def _image_score(score_map: np.ndarray) -> float:
    count = max(1, math.ceil(score_map.size * (1.0 - SCORE_QUANTILE)))
    return float(np.partition(score_map.ravel(), -count)[-count:].mean())


def train_model(
    data_dir: Path,
    model_path: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> dict[str, object]:
    """以對位正常品建立逐像素參考模型並估算初始門檻。"""
    _validate_image_size(image_size)
    paths = _image_paths(data_dir)
    if len(paths) < 2:
        raise ValueError(f"正常品資料夾至少需要 2 張影像，目前找到 {len(paths)} 張：{data_dir}")

    images = np.stack([_load_image(path, image_size) for path in paths], axis=0)
    mean = images.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = images.std(axis=0, dtype=np.float64).astype(np.float32)
    calibration_scores = [_image_score(_score_map(image, mean, std)) for image in images]
    calibration_mean = float(np.mean(calibration_scores))
    calibration_std = float(np.std(calibration_scores))
    # 設最低分數門檻，避免完全相同的訓練圖像導致零門檻。
    threshold = max(calibration_mean + 3.0 * calibration_std, 4.0)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pixel_reference_v1",
            "mean": torch.from_numpy(mean),
            "std": torch.from_numpy(std),
            "image_size": image_size,
            "threshold": threshold,
            "calibration_mean": calibration_mean,
            "calibration_std": calibration_std,
            "training_images": len(paths),
        },
        model_path,
    )
    result = {
        "model_path": str(model_path),
        "training_images": len(paths),
        "image_size": image_size,
        "calibration_mean": calibration_mean,
        "calibration_std": calibration_std,
        "threshold": threshold,
        "method": "逐像素平均值與標準差參考",
    }
    print("\n參考模型建立完成：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("注意：影像需固定角度與位置；正式使用前請以獨立資料校準門檻。")
    return result


def _load_model(model_path: Path) -> dict[str, object]:
    if not model_path.is_file():
        raise ValueError(f"找不到模型檔：{model_path}。請先執行 train 或 demo。")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "pixel_reference_v1":
        raise ValueError(f"不支援的模型格式：{model_path}")
    return checkpoint


def predict_image(
    model_path: Path,
    image_path: Path,
    output_dir: Path,
    threshold: float | None = None,
) -> dict[str, object]:
    """比對單張影像並輸出熱區、疊圖和 JSON。"""
    if not image_path.is_file():
        raise ValueError(f"找不到輸入影像：{image_path}")
    checkpoint = _load_model(model_path)
    image_size = int(checkpoint["image_size"])
    mean = checkpoint["mean"].numpy()
    std = checkpoint["std"].numpy()
    original = Image.open(image_path).convert("RGB")
    image = _load_image(image_path, image_size)
    residual = _score_map(image, mean, std)
    score = _image_score(residual)
    applied_threshold = float(checkpoint["threshold"] if threshold is None else threshold)
    if not math.isfinite(applied_threshold) or applied_threshold <= 0:
        raise ValueError("threshold 必須是大於 0 的有限數值。")

    # 色階僅用於視覺化；分類使用完整標準化差異分數。
    scale = max(float(np.quantile(residual, 0.995)), 1e-6)
    heat = np.clip(residual / scale, 0, 1)
    heat_rgb = np.zeros((*heat.shape, 3), dtype=np.uint8)
    heat_rgb[..., 0] = np.clip(255 * heat, 0, 255).astype(np.uint8)
    heat_rgb[..., 1] = np.clip(255 * (1 - np.abs(heat - 0.5) * 2) * 0.35, 0, 255).astype(np.uint8)
    heat_image = Image.fromarray(heat_rgb, mode="RGB").resize(original.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(original, heat_image, alpha=0.45)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    heat_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    result_path = output_dir / f"{stem}_result.json"
    heat_image.save(heat_path)
    overlay.save(overlay_path)
    result = {
        "image": str(image_path),
        "status": "異常" if score > applied_threshold else "正常",
        "anomaly_score": score,
        "threshold": applied_threshold,
        "heatmap": str(heat_path),
        "overlay": str(overlay_path),
        "model": str(model_path),
        "method": "逐像素平均值與標準差參考",
        "note": "此方法要求拍攝位置與影像對位穩定；請用真實驗證資料校準門檻。",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _make_demo_images(root: Path, image_size: int) -> tuple[Path, Path, Path]:
    normal_dir = root / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(71)
    for index in range(12):
        shade = rng.randint(205, 215)
        image = Image.new("RGB", (image_size, image_size), (shade, shade, shade))
        draw = ImageDraw.Draw(image)
        margin = image_size // 5
        draw.rectangle(
            (margin, margin, image_size - margin, image_size - margin),
            fill=(150 + rng.randint(-2, 2), 155 + rng.randint(-2, 2), 160 + rng.randint(-2, 2)),
            outline=(90, 95, 100),
            width=max(1, image_size // 64),
        )
        image.save(normal_dir / f"normal_{index:02}.png")

    clean_path = root / "demo_clean.png"
    defect_path = root / "demo_defect.png"
    clean_path.write_bytes((normal_dir / "normal_00.png").read_bytes())
    with Image.open(clean_path) as clean:
        defect = clean.copy()
    draw = ImageDraw.Draw(defect)
    defect_size = max(4, image_size // 8)
    center = image_size // 2
    draw.rectangle(
        (center - defect_size // 2, center - defect_size // 2,
         center + defect_size // 2, center + defect_size // 2),
        fill=(10, 10, 10),
    )
    defect.save(defect_path)
    return normal_dir, clean_path, defect_path


def run_demo(output_dir: Path, image_size: int) -> None:
    _validate_image_size(image_size)
    normal_dir, clean_path, defect_path = _make_demo_images(output_dir / "demo_data", image_size)
    model_path = output_dir / "pixel_reference.pt"
    train_model(normal_dir, model_path, image_size=image_size)
    results_dir = output_dir / "demo_results"
    clean_result = predict_image(model_path, clean_path, results_dir)
    defect_result = predict_image(model_path, defect_path, results_dir)
    print(f"\nDemo 完成：{output_dir.resolve()}")
    print("合成影像僅展示像素參考流程；不代表真實產品檢測能力。")
    if clean_result["status"] != "正常" or defect_result["status"] != "異常":
        print("提醒：合成範例未同時正確分類；請勿直接使用此門檻。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="像素參考式工業影像異常偵測（固定視角）")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="用正常品影像建立逐像素參考模型")
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument("--model-path", type=Path, default=Path("artifacts/pixel_reference.pt"))
    train_parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)

    predict_parser = commands.add_parser("predict", help="比對單張影像並輸出分數和熱區")
    predict_parser.add_argument("--model-path", type=Path, required=True)
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.add_argument("--output-dir", type=Path, default=Path("outputs/pixel_reference"))
    predict_parser.add_argument("--threshold", type=float)

    demo_parser = commands.add_parser("demo", help="產生固定視角合成資料並跑完整流程")
    demo_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pixel_reference_demo"))
    demo_parser.add_argument("--image-size", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            train_model(args.data_dir, args.model_path, args.image_size)
        elif args.command == "predict":
            predict_image(args.model_path, args.image, args.output_dir, args.threshold)
        else:
            run_demo(args.output_dir, args.image_size)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"錯誤：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())