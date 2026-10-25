"""離線工業影像異常偵測原型（只需正常品影像）。

使用方式：
    python anomaly_detector.py demo
    python anomaly_detector.py train --data-dir data/normal --model-path artifacts/model.pt
    python anomaly_detector.py predict --model-path artifacts/model.pt --image sample.png

此原型以 PyTorch 自編碼器學習正常影像，並以重建誤差產生異常熱區。
它不是已驗證的工業檢測系統；真實使用需依產品影像校準、測試與設定門檻。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_IMAGE_SIZE = 128


class ImageFolderDataset(Dataset[torch.Tensor]):
    """讀取資料夾中的影像，轉成 [0, 1] RGB 張量。"""

    def __init__(self, paths: Iterable[Path], image_size: int) -> None:
        self.paths = list(paths)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            image = ImageOps.fit(image.convert("RGB"), (self.image_size, self.image_size))
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class ConvAutoencoder(nn.Module):
    """小型卷積自編碼器，適合 CPU 上做原型訓練。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image))


def _image_paths(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"找不到影像資料夾：{folder}")
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _validate_image_size(image_size: int) -> None:
    if image_size < 32 or image_size > 512 or image_size % 8 != 0:
        raise ValueError("--image-size 必須是 32 到 512 之間的 8 倍數。")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("指定了 CUDA，但目前 PyTorch 無法使用 CUDA。")
    return requested


def _load_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.fit(image.convert("RGB"), (image_size, image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def _image_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for images in loader:
            images = images.to(device)
            residual = (model(images) - images).abs().mean(dim=1)
            # 使用最異常 1% 像素的平均值，減少大面積正常背景對分數的稀釋。
            count = max(1, math.ceil(residual.shape[-2] * residual.shape[-1] * 0.01))
            scores.extend(residual.flatten(1).topk(count, dim=1).values.mean(dim=1).cpu().tolist())
    return scores


def train_model(
    data_dir: Path,
    model_path: Path,
    epochs: int = 20,
    batch_size: int = 8,
    image_size: int = DEFAULT_IMAGE_SIZE,
    learning_rate: float = 1e-3,
    device_name: str = "auto",
    seed: int = 42,
) -> dict[str, object]:
    """只用正常品圖片訓練並保存模型及初始異常分數門檻。"""
    _validate_image_size(image_size)
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs、batch-size 必須大於 0，learning-rate 也必須大於 0。")
    paths = _image_paths(data_dir)
    if len(paths) < 2:
        raise ValueError(f"正常品資料夾至少需要 2 張影像，目前找到 {len(paths)} 張：{data_dir}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device(device_name)
    dataset = ImageFolderDataset(paths, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.shape[0]
        print(f"Epoch {epoch:>3}/{epochs} | loss={total_loss / len(dataset):.6f}")

    calibration_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    scores = _image_scores(model, calibration_loader, device)
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))
    threshold = max(score_mean + 3.0 * score_std, score_mean * 1.25, 1e-6)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "image_size": image_size,
            "threshold": threshold,
            "calibration_mean": score_mean,
            "calibration_std": score_std,
            "training_images": len(paths),
            "epochs": epochs,
            "seed": seed,
        },
        model_path,
    )
    result = {
        "model_path": str(model_path),
        "training_images": len(paths),
        "epochs": epochs,
        "image_size": image_size,
        "device": str(device),
        "calibration_mean": score_mean,
        "calibration_std": score_std,
        "threshold": threshold,
    }
    print("\n訓練完成：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("注意：此門檻是原型校準值，正式檢測需用獨立正常/異常驗證資料調整。")
    return result


def load_model(model_path: Path, device: torch.device) -> tuple[ConvAutoencoder, dict[str, object]]:
    if not model_path.is_file():
        raise ValueError(f"找不到模型檔：{model_path}。請先執行 train 或 demo。")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = ConvAutoencoder()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def predict_image(
    model_path: Path,
    image_path: Path,
    output_dir: Path,
    device_name: str = "auto",
    threshold: float | None = None,
) -> dict[str, object]:
    """預測單張影像，輸出熱區圖、疊圖與 JSON 結果。"""
    if not image_path.is_file():
        raise ValueError(f"找不到輸入影像：{image_path}")
    device = _device(device_name)
    model, checkpoint = load_model(model_path, device)
    image_size = int(checkpoint["image_size"])
    original = Image.open(image_path).convert("RGB")
    tensor = _load_tensor(image_path, image_size).to(device)

    with torch.inference_mode():
        reconstruction = model(tensor)
        residual = (reconstruction - tensor).abs().mean(dim=1)[0].cpu().numpy()
    count = max(1, math.ceil(residual.size * 0.01))
    score = float(np.partition(residual.ravel(), -count)[-count:].mean())
    applied_threshold = float(checkpoint["threshold"] if threshold is None else threshold)
    if applied_threshold <= 0:
        raise ValueError("threshold 必須大於 0。")

    # 依影像自身最高誤差縮放色階，以利觀察；判定仍使用原始重建誤差分數。
    scale = max(float(np.quantile(residual, 0.995)), 1e-6)
    heat = np.clip(residual / scale, 0, 1)
    heat_rgb = np.zeros((*heat.shape, 3), dtype=np.uint8)
    heat_rgb[..., 0] = np.clip(255 * heat, 0, 255).astype(np.uint8)
    heat_rgb[..., 1] = np.clip(255 * (1 - np.abs(heat - 0.5) * 2) * 0.35, 0, 255).astype(np.uint8)
    heat_image = Image.fromarray(heat_rgb, mode="RGB").resize(original.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(original, heat_image, alpha=0.45)
    status = "異常" if score > applied_threshold else "正常"

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    heat_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    result_path = output_dir / f"{stem}_result.json"
    heat_image.save(heat_path)
    overlay.save(overlay_path)
    result = {
        "image": str(image_path),
        "status": status,
        "anomaly_score": score,
        "threshold": applied_threshold,
        "heatmap": str(heat_path),
        "overlay": str(overlay_path),
        "model": str(model_path),
        "note": "原型結果僅供概念驗證；請使用目標產品資料校準門檻並評估誤報/漏報。",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _make_demo_images(root: Path, image_size: int = DEFAULT_IMAGE_SIZE) -> tuple[Path, Path, Path]:
    normal_dir = root / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(17)
    for index in range(16):
        background = rng.randint(215, 229)
        image = Image.new("RGB", (image_size, image_size), (background, background, background))
        draw = ImageDraw.Draw(image)
        margin = image_size // 5
        draw.rounded_rectangle(
            (margin, margin, image_size - margin, image_size - margin),
            radius=max(3, image_size // 32),
            fill=(rng.randint(145, 160), rng.randint(150, 165), rng.randint(155, 170)),
            outline=(95, 100, 105),
            width=max(1, image_size // 64),
        )
        hole_radius = max(2, image_size // 32)
        for x in (margin + image_size // 12, image_size - margin - image_size // 12):
            for y in (margin + image_size // 12, image_size - margin - image_size // 12):
                draw.ellipse((x - hole_radius, y - hole_radius, x + hole_radius, y + hole_radius), fill=(65, 70, 75))
        # 輕微感光差異，模擬正常拍攝變動。
        image = image.filter(ImageFilter.GaussianBlur(radius=0.15))
        image.save(normal_dir / f"normal_{index:02}.png")

    clean_path = root / "demo_clean.png"
    defect_path = root / "demo_defect.png"
    clean_path.write_bytes((normal_dir / "normal_00.png").read_bytes())
    with Image.open(clean_path) as clean:
        defect = clean.copy()
    draw = ImageDraw.Draw(defect)
    # 使用明顯的局部缺損，讓合成 demo 示範異常定位；不代表可檢出細微真實瑕疵。
    defect_size = max(4, image_size // 8)
    center = image_size // 2
    draw.rectangle(
        (center - defect_size // 2, center - defect_size // 2,
         center + defect_size // 2, center + defect_size // 2),
        fill=(25, 20, 20),
    )
    defect.save(defect_path)
    return normal_dir, clean_path, defect_path


def run_demo(output_dir: Path, epochs: int, image_size: int, device_name: str) -> None:
    _validate_image_size(image_size)
    demo_root = output_dir / "demo_data"
    normal_dir, clean_image, defect_image = _make_demo_images(demo_root, image_size)
    model_path = output_dir / "demo_model.pt"
    print(f"建立 {len(_image_paths(normal_dir))} 張合成正常品影像，開始訓練原型模型。")
    train_model(normal_dir, model_path, epochs=epochs, batch_size=4, image_size=image_size, device_name=device_name)
    clean_result = predict_image(model_path, clean_image, output_dir / "demo_results", device_name)
    defect_result = predict_image(model_path, defect_image, output_dir / "demo_results", device_name)
    print(f"\nDemo 完成。可查看輸出資料夾：{output_dir.resolve()}")
    print("合成影像僅用來確認程式流程，不代表真實產品上的檢測準確率。")
    if clean_result["status"] != "正常" or defect_result["status"] != "異常":
        print(
            "提醒：這組極小合成資料未能同時正確分類正常/缺損範例；"
            "請勿以此門檻判斷真實產品，需用代表性影像校準與評估。"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch 工業影像異常偵測第一版（正常品自編碼器）")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="使用正常品影像訓練模型")
    train_parser.add_argument("--data-dir", type=Path, required=True, help="正常品影像資料夾（會遞迴讀取）")
    train_parser.add_argument("--model-path", type=Path, default=Path("artifacts/anomaly_model.pt"))
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train_parser.add_argument("--seed", type=int, default=42)

    predict_parser = commands.add_parser("predict", help="對單張影像產生分數、熱區圖與疊圖")
    predict_parser.add_argument("--model-path", type=Path, required=True)
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    predict_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    predict_parser.add_argument("--threshold", type=float, help="覆寫模型內的異常門檻")

    demo_parser = commands.add_parser("demo", help="產生合成資料、訓練並執行正常/瑕疵範例")
    demo_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    demo_parser.add_argument("--epochs", type=int, default=12)
    demo_parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    demo_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            train_model(
                args.data_dir,
                args.model_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                image_size=args.image_size,
                learning_rate=args.learning_rate,
                device_name=args.device,
                seed=args.seed,
            )
        elif args.command == "predict":
            predict_image(args.model_path, args.image, args.output_dir, args.device, args.threshold)
        else:
            run_demo(args.output_dir, args.epochs, args.image_size, args.device)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"錯誤：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())