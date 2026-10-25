"""局部區塊最近鄰工業影像異常偵測原型。

只用正常品影像建立局部外觀特徵庫；推論時比較每個影像區塊與最近的
正常區塊，並輸出異常分數、熱區圖、疊圖及 JSON。適用於固定拍攝條件。

使用方式：
    python patch_knn_detector.py demo
    python patch_knn_detector.py train --data-dir data/normal
    python patch_knn_detector.py predict --model-path artifacts/patch_knn.pt --image sample.png
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
DEFAULT_PATCH_SIZE = 8
DEFAULT_STRIDE = 4
MAX_BANK_SIZE = 12000
DISTANCE_CHUNK_SIZE = 256


def _image_paths(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"找不到影像資料夾：{folder}")
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _validate_settings(image_size: int, patch_size: int, stride: int) -> None:
    if image_size < 32 or image_size > 512:
        raise ValueError("--image-size 必須介於 32 到 512。")
    if patch_size < 4 or patch_size > image_size:
        raise ValueError("--patch-size 必須介於 4 與 image-size 之間。")
    if stride < 1 or stride > patch_size:
        raise ValueError("--stride 必須介於 1 與 patch-size 之間。")


def _load_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        fitted = ImageOps.fit(image.convert("RGB"), (image_size, image_size))
        return np.asarray(fitted, dtype=np.float32) / 255.0


def _positions(length: int, patch_size: int, stride: int) -> list[int]:
    positions = list(range(0, length - patch_size + 1, stride))
    final = length - patch_size
    if positions[-1] != final:
        positions.append(final)
    return positions


def _extract_features(
    image: np.ndarray, patch_size: int, stride: int
) -> tuple[np.ndarray, list[int], list[int]]:
    """每個區塊以 RGB 均值/標準差及灰階方向梯度構成 8 維描述子。"""
    height, width = image.shape[:2]
    ys = _positions(height, patch_size, stride)
    xs = _positions(width, patch_size, stride)
    gray = image.mean(axis=2)
    features: list[list[float]] = []
    for y in ys:
        for x in xs:
            patch = image[y:y + patch_size, x:x + patch_size]
            gray_patch = gray[y:y + patch_size, x:x + patch_size]
            features.append([
                *patch.mean(axis=(0, 1)).tolist(),
                *patch.std(axis=(0, 1)).tolist(),
                float(np.abs(np.diff(gray_patch, axis=1)).mean()),
                float(np.abs(np.diff(gray_patch, axis=0)).mean()),
            ])
    return np.asarray(features, dtype=np.float32), ys, xs


def _nearest_distances(queries: np.ndarray, bank: np.ndarray) -> np.ndarray:
    bank_norm = np.sum(bank * bank, axis=1)
    distances = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), DISTANCE_CHUNK_SIZE):
        chunk = queries[start:start + DISTANCE_CHUNK_SIZE]
        squared = (
            np.sum(chunk * chunk, axis=1, keepdims=True)
            + bank_norm[None, :]
            - 2.0 * chunk @ bank.T
        )
        distances[start:start + len(chunk)] = np.sqrt(np.maximum(squared.min(axis=1), 0.0))
    return distances


def _image_score(distances: np.ndarray) -> float:
    count = max(1, math.ceil(len(distances) * 0.01))
    return float(np.partition(distances, -count)[-count:].mean())


def train_model(
    data_dir: Path,
    model_path: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> dict[str, object]:
    """用正常影像建立區塊特徵庫，並以留一張影像方式估算起始門檻。"""
    _validate_settings(image_size, patch_size, stride)
    paths = _image_paths(data_dir)
    if len(paths) < 2:
        raise ValueError(f"正常品資料夾至少需要 2 張影像，目前找到 {len(paths)} 張：{data_dir}")

    image_features = [
        _extract_features(_load_image(path, image_size), patch_size, stride)[0]
        for path in paths
    ]
    all_features = np.concatenate(image_features, axis=0)
    feature_mean = all_features.mean(axis=0)
    feature_scale = np.maximum(all_features.std(axis=0), 0.025)
    normalized_images = [((features - feature_mean) / feature_scale).astype(np.float32)
                         for features in image_features]

    # 限制模型大小；均勻抽樣保留不同影像位置與外觀。
    all_normalized = np.concatenate(normalized_images, axis=0)
    if len(all_normalized) > MAX_BANK_SIZE:
        selected = np.linspace(0, len(all_normalized) - 1, MAX_BANK_SIZE, dtype=np.int64)
        bank = all_normalized[selected]
    else:
        bank = all_normalized

    calibration_scores: list[float] = []
    if len(paths) > 2:
        for index, query in enumerate(normalized_images):
            reference = np.concatenate(normalized_images[:index] + normalized_images[index + 1:])
            calibration_scores.append(_image_score(_nearest_distances(query, reference)))
    else:
        # 僅有兩張時，各自以另一張作參考，避免對照自身而得到零距離。
        calibration_scores = [
            _image_score(_nearest_distances(normalized_images[index], normalized_images[1 - index]))
            for index in range(2)
        ]
    calibration_mean = float(np.mean(calibration_scores))
    calibration_std = float(np.std(calibration_scores))
    threshold = max(calibration_mean + 3.0 * calibration_std, 0.1)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "patch_knn_v1",
        "bank": torch.from_numpy(bank.copy()),
        "feature_mean": torch.from_numpy(feature_mean.astype(np.float32)),
        "feature_scale": torch.from_numpy(feature_scale.astype(np.float32)),
        "image_size": image_size,
        "patch_size": patch_size,
        "stride": stride,
        "threshold": threshold,
        "calibration_mean": calibration_mean,
        "calibration_std": calibration_std,
        "training_images": len(paths),
    }, model_path)
    result = {
        "model_path": str(model_path),
        "training_images": len(paths),
        "bank_patches": len(bank),
        "image_size": image_size,
        "patch_size": patch_size,
        "stride": stride,
        "calibration_mean": calibration_mean,
        "calibration_std": calibration_std,
        "threshold": threshold,
        "method": "局部區塊最近鄰特徵比對",
    }
    print("\n區塊特徵庫建立完成：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("注意：此初始門檻需以獨立正常/異常影像校準。")
    return result


def _load_model(model_path: Path) -> dict[str, object]:
    if not model_path.is_file():
        raise ValueError(f"找不到模型檔：{model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "patch_knn_v1":
        raise ValueError(f"不支援的模型格式：{model_path}")
    return checkpoint


def predict_image(
    model_path: Path,
    image_path: Path,
    output_dir: Path,
    threshold: float | None = None,
) -> dict[str, object]:
    """將影像局部特徵與正常區塊庫比對，輸出異常結果與可視化。"""
    if not image_path.is_file():
        raise ValueError(f"找不到輸入影像：{image_path}")
    checkpoint = _load_model(model_path)
    image_size = int(checkpoint["image_size"])
    patch_size = int(checkpoint["patch_size"])
    stride = int(checkpoint["stride"])
    original = Image.open(image_path).convert("RGB")
    image = _load_image(image_path, image_size)
    features, ys, xs = _extract_features(image, patch_size, stride)
    feature_mean = checkpoint["feature_mean"].numpy()
    feature_scale = checkpoint["feature_scale"].numpy()
    bank = checkpoint["bank"].numpy()
    distances = _nearest_distances(((features - feature_mean) / feature_scale).astype(np.float32), bank)
    score_map = distances.reshape(len(ys), len(xs))
    score = _image_score(distances)
    applied_threshold = float(checkpoint["threshold"] if threshold is None else threshold)
    if not math.isfinite(applied_threshold) or applied_threshold <= 0:
        raise ValueError("threshold 必須是有限的正數。")

    # 將區塊距離投影回影像位置，再放大到原始尺寸供檢視。
    pixel_map = np.zeros((image_size, image_size), dtype=np.float32)
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            pixel_map[y:y + patch_size, x:x + patch_size] = np.maximum(
                pixel_map[y:y + patch_size, x:x + patch_size], score_map[row, column]
            )
    scale = max(float(np.quantile(pixel_map, 0.995)), 1e-6)
    heat = np.clip(pixel_map / scale, 0, 1)
    heat_rgb = np.zeros((*heat.shape, 3), dtype=np.uint8)
    heat_rgb[..., 0] = (255 * heat).astype(np.uint8)
    heat_rgb[..., 1] = (255 * (1 - np.abs(heat - 0.5) * 2) * 0.35).astype(np.uint8)
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
        "method": "局部區塊最近鄰特徵比對",
        "note": "原型結果僅供概念驗證；請用實際產品資料校準門檻並評估誤報/漏報。",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _make_demo_images(root: Path, image_size: int) -> tuple[Path, Path, Path]:
    normal_dir = root / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(93)
    for index in range(8):
        shade = 215 + rng.randint(-2, 2)
        image = Image.new("RGB", (image_size, image_size), (shade, shade, shade))
        draw = ImageDraw.Draw(image)
        margin = image_size // 5
        draw.rectangle((margin, margin, image_size - margin, image_size - margin),
                       fill=(150, 155, 160), outline=(90, 95, 100), width=max(1, image_size // 64))
        image.save(normal_dir / f"normal_{index:02}.png")

    clean_path = root / "demo_clean.png"
    defect_path = root / "demo_defect.png"
    clean_path.write_bytes((normal_dir / "normal_00.png").read_bytes())
    with Image.open(clean_path) as clean:
        defect = clean.copy()
    defect_size = max(6, image_size // 8)
    center = image_size // 2
    ImageDraw.Draw(defect).rectangle(
        (center - defect_size // 2, center - defect_size // 2,
         center + defect_size // 2, center + defect_size // 2), fill=(5, 5, 5)
    )
    defect.save(defect_path)
    return normal_dir, clean_path, defect_path


def run_demo(output_dir: Path, image_size: int) -> None:
    _validate_settings(image_size, DEFAULT_PATCH_SIZE, DEFAULT_STRIDE)
    normal_dir, clean_path, defect_path = _make_demo_images(output_dir / "demo_data", image_size)
    model_path = output_dir / "patch_knn.pt"
    train_model(normal_dir, model_path, image_size=image_size)
    results_dir = output_dir / "demo_results"
    clean_result = predict_image(model_path, clean_path, results_dir)
    defect_result = predict_image(model_path, defect_path, results_dir)
    print(f"\nDemo 完成：{output_dir.resolve()}")
    print("合成影像僅示範局部最近鄰流程，不代表真實產品檢測能力。")
    if clean_result["status"] != "正常" or defect_result["status"] != "異常":
        print("提醒：合成範例未同時正確分類；請勿直接使用此門檻。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="局部區塊最近鄰工業影像異常偵測")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="以正常品建立局部區塊特徵庫")
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument("--model-path", type=Path, default=Path("artifacts/patch_knn.pt"))
    train_parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    train_parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    train_parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)

    predict_parser = commands.add_parser("predict", help="比對單張影像並輸出熱區與分數")
    predict_parser.add_argument("--model-path", type=Path, required=True)
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.add_argument("--output-dir", type=Path, default=Path("outputs/patch_knn"))
    predict_parser.add_argument("--threshold", type=float)

    demo_parser = commands.add_parser("demo", help="產生合成正常/缺損影像並執行完整流程")
    demo_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/patch_knn_demo"))
    demo_parser.add_argument("--image-size", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            train_model(args.data_dir, args.model_path, args.image_size, args.patch_size, args.stride)
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