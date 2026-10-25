"""Gradio web demo for industrial image anomaly detection.

Supports three detector methods:
1. Convolutional Autoencoder (重建誤差)
2. Pixel Reference (逐像素參考)
3. Patch KNN (局部區塊最近鄰)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from anomaly_detector import train_model as train_autoencoder, predict_image as predict_autoencoder
from pixel_reference_detector import train_model as train_pixel_ref, predict_image as predict_pixel_ref
from patch_knn_detector import train_model as train_patch_knn, predict_image as predict_patch_knn


ROOT = Path(__file__).resolve().parent
DEMO_MODEL_PATH = ROOT / "demo_assets" / "patch_knn.pt"
MAX_IMAGE_PIXELS = 20_000_000


def _validate_image(image) -> str:
    if image is None:
        raise gr.Error("請先上傳圖片，或點選下方的範例圖片。")
    if hasattr(image, "width") and hasattr(image, "height"):
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise gr.Error("圖片尺寸過大，請使用總像素不超過 2,000 萬的圖片。")
    return ""


def _run_autoencoder(model_path: Path | None, image, image_size: int, threshold: float | None):
    if image is None:
        raise gr.Error("請先上傳圖片。")
    
    with tempfile.TemporaryDirectory(prefix="ae-demo-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "uploaded.png"
        output_dir = temp_path / "results"
        
        img = image.convert("RGB") if hasattr(image, "convert") else Image.fromarray(image).convert("RGB")
        img.save(input_path, format="PNG")
        
        if model_path is None or not model_path.exists():
            raise gr.Error("請先上傳或訓練模型。")
        
        result = predict_autoencoder(model_path, input_path, output_dir, threshold=threshold)
        overlay = Image.open(result["overlay"]).copy()
        heatmap = Image.open(result["heatmap"]).copy()
        
    status = "🔴 判定：異常" if result["status"] == "異常" else "🟢 判定：正常"
    details = {
        "判定": result["status"],
        "異常分數": round(float(result["anomaly_score"]), 6),
        "判定門檻": round(float(result.get("threshold", 0)), 6),
        "方法": "卷積自編碼器重建誤差",
    }
    return overlay, heatmap, f"## {status}\n\n```json\n{json.dumps(details, ensure_ascii=False, indent=2)}\n```"


def _run_pixel_reference(model_path: Path | None, image, threshold: float | None):
    if image is None:
        raise gr.Error("請先上傳圖片。")
    
    with tempfile.TemporaryDirectory(prefix="pixel-demo-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "uploaded.png"
        output_dir = temp_path / "results"
        
        img = image.convert("RGB") if hasattr(image, "convert") else Image.fromarray(image).convert("RGB")
        img.save(input_path, format="PNG")
        
        if model_path is None or not model_path.exists():
            raise gr.Error("請先上傳或訓練模型。")
        
        result = predict_pixel_ref(model_path, input_path, output_dir, threshold=threshold)
        overlay = Image.open(result["overlay"]).copy()
        heatmap = Image.open(result["heatmap"]).copy()
        
    status = "🔴 判定：異常" if result["status"] == "異常" else "🟢 判定：正常"
    details = {
        "判定": result["status"],
        "異常分數": round(float(result["anomaly_score"]), 6),
        "判定門檻": round(float(result.get("threshold", 0)), 6),
        "方法": "逐像素平均值與標準差參考",
    }
    return overlay, heatmap, f"## {status}\n\n```json\n{json.dumps(details, ensure_ascii=False, indent=2)}\n```"


def _run_patch_knn(model_path: Path | None, image, threshold: float | None):
    if image is None:
        raise gr.Error("請先上傳圖片。")
    
    with tempfile.TemporaryDirectory(prefix="knn-demo-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "uploaded.png"
        output_dir = temp_path / "results"
        
        img = image.convert("RGB") if hasattr(image, "convert") else Image.fromarray(image).convert("RGB")
        img.save(input_path, format="PNG")
        
        if model_path is None or not model_path.exists():
            raise gr.Error("請先上傳或訓練模型。")
        
        result = predict_patch_knn(model_path, input_path, output_dir, threshold=threshold)
        overlay = Image.open(result["overlay"]).copy()
        heatmap = Image.open(result["heatmap"]).copy()
        
    status = "🔴 判定：異常" if result["status"] == "異常" else "🟢 判定：正常"
    details = {
        "判定": result["status"],
        "異常分數": round(float(result["anomaly_score"]), 6),
        "判定門檻": round(float(result.get("threshold", 0)), 6),
        "方法": "局部區塊最近鄰特徵比對",
    }
    return overlay, heatmap, f"## {status}\n\n```json\n{json.dumps(details, ensure_ascii=False, indent=2)}\n```"


def _train_autoencoder(data_dir, image_size, epochs, batch_size, model_path):
    if data_dir is None or not data_dir.exists():
        raise gr.Error("請選擇有效的正常品影像資料夾。")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    result = train_autoencoder(data_dir, model_path, epochs=epochs, batch_size=batch_size, image_size=image_size)
    return str(model_path)


def _train_pixel_reference(data_dir, image_size, model_path):
    if data_dir is None or not data_dir.exists():
        raise gr.Error("請選擇有效的正常品影像資料夾。")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    result = train_pixel_ref(data_dir, model_path, image_size=image_size)
    return str(model_path)


def _train_patch_knn(data_dir, image_size, patch_size, stride, model_path):
    if data_dir is None or not data_dir.exists():
        raise gr.Error("請選擇有效的正常品影像資料夾。")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    result = train_patch_knn(data_dir, model_path, image_size=image_size, patch_size=patch_size, stride=stride)
    return str(model_path)


with gr.Blocks(title="工業影像異常偵測 Demo") as demo:
    gr.Markdown("""# 🔍 工業影像异常偵測 Demo

支援三種偵測方法，可上傳正常品影像訓練模型，再對新圖片進行異常檢測。

> **展示用途：** 此為概念原型，結果僅供操作示範，不能代表真實產品檢測效能。
""")
    with gr.Tabs():
        with gr.Tab("① 自編碼器"):
            gr.Markdown("### 卷積自編碼器重建誤差偵測")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 訓練模型")
                    ae_data_dir = gr.File(label="上傳正常品資料夾", file_count="directory")
                    ae_image_size = gr.Slider(minimum=32, maximum=512, value=128, step=8, label="影像尺寸")
                    ae_epochs = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="訓練 epoch")
                    ae_batch_size = gr.Slider(minimum=1, maximum=32, value=8, step=1, label="批次大小")
                    ae_model_path = gr.Textbox(label="模型儲存路徑", value="artifacts/ae_model.pt")
                    ae_train_btn = gr.Button("開始訓練", variant="primary")
                    ae_train_output = gr.Markdown()
                with gr.Column():
                    gr.Markdown("#### 上傳模型或預測")
                    ae_model_upload = gr.File(label="或上傳已有模型", file_count="single")
                    ae_image_input = gr.Image(label="上傳要檢查的圖片", type="pil")
                    ae_threshold = gr.Slider(minimum=0, maximum=5, value=0.0, step=0.01, label="門檻值 (0=使用模型預設)")
                    ae_run_btn = gr.Button("開始檢查", variant="primary")
            with gr.Row():
                ae_overlay = gr.Image(label="異常區域疊圖", type="pil")
                ae_heatmap = gr.Image(label="異常熱區圖", type="pil")
            ae_result = gr.Markdown()
        with gr.Tab("② 像素參考法"):
            gr.Markdown("### 逐像素平均值與標準差參考偵測")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 訓練模型")
                    pr_data_dir = gr.File(label="上傳正常品資料夾", file_count="directory")
                    pr_image_size = gr.Slider(minimum=16, maximum=1024, value=128, step=1, label="影像尺寸")
                    pr_model_path = gr.Textbox(label="模型儲存路徑", value="artifacts/pr_model.pt")
                    pr_train_btn = gr.Button("開始訓練", variant="primary")
                    pr_train_output = gr.Markdown()
                with gr.Column():
                    gr.Markdown("#### 上傳模型或預測")
                    pr_model_upload = gr.File(label="或上傳已有模型", file_count="single")
                    pr_image_input = gr.Image(label="上傳要檢查的圖片", type="pil")
                    pr_threshold = gr.Slider(minimum=0, maximum=20, value=0.0, step=0.1, label="門檻值 (0=使用模型預設)")
                    pr_run_btn = gr.Button("開始檢查", variant="primary")
            with gr.Row():
                pr_overlay = gr.Image(label="異常區域疊圖", type="pil")
                pr_heatmap = gr.Image(label="異常熱區圖", type="pil")
            pr_result = gr.Markdown()
        with gr.Tab("③ 局部區塊 KNN"):
            gr.Markdown("### 局部區塊最近鄰特徵比對偵測")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 訓練模型")
                    knn_data_dir = gr.File(label="上傳正常品資料夾", file_count="directory")
                    knn_image_size = gr.Slider(minimum=32, maximum=512, value=128, step=8, label="影像尺寸")
                    knn_patch_size = gr.Slider(minimum=4, maximum=64, value=8, step=1, label="區塊大小")
                    knn_stride = gr.Slider(minimum=1, maximum=32, value=4, step=1, label="步幅")
                    knn_model_path = gr.Textbox(label="模型儲存路徑", value="artifacts/knn_model.pt")
                    knn_train_btn = gr.Button("開始訓練", variant="primary")
                    knn_train_output = gr.Markdown()
                with gr.Column():
                    gr.Markdown("#### 上傳模型或預測")
                    knn_model_upload = gr.File(label="或上傳已有模型", file_count="single")
                    knn_image_input = gr.Image(label="上傳要檢查的圖片", type="pil")
                    knn_threshold = gr.Slider(minimum=0, maximum=50, value=0.0, step=0.1, label="門檻值 (0=使用模型預設)")
                    knn_run_btn = gr.Button("開始檢查", variant="primary")
            with gr.Row():
                knn_overlay = gr.Image(label="異常區域疊圖", type="pil")
                knn_heatmap = gr.Image(label="異常熱區圖", type="pil")
            knn_result = gr.Markdown()
        with gr.Tab("🎮 Demo 快速體驗"):
            gr.Markdown("使用內建 demo 模型快速體驗（僅限第三套 KNN 偵測器）")
            demo_image_input = gr.Image(label="上傳或選擇範例圖片", type="pil")
            demo_run_btn = gr.Button("開始檢查", variant="primary")
            demo_overlay = gr.Image(label="異常區域疊圖", type="pil")
            demo_heatmap = gr.Image(label="異常熱區圖", type="pil")
            demo_result = gr.Markdown()
            gr.Examples(examples=[[str(ROOT / "demo_assets" / "demo_clean.png")], [str(ROOT / "demo_assets" / "demo_defect.png")]], inputs=demo_image_input, label="點選範例圖片快速試用")
    ae_train_btn.click(fn=_train_autoencoder, inputs=[ae_data_dir, ae_image_size, ae_epochs, ae_batch_size, ae_model_path], outputs=ae_train_output).then(fn=lambda x: x, inputs=ae_model_path, outputs=ae_model_upload)
    ae_run_btn.click(fn=_validate_image, inputs=ae_image_input, outputs=None).then(fn=_run_autoencoder, inputs=[ae_model_upload, ae_image_input, ae_image_size, ae_threshold], outputs=[ae_overlay, ae_heatmap, ae_result])
    pr_train_btn.click(fn=_train_pixel_reference, inputs=[pr_data_dir, pr_image_size, pr_model_path], outputs=pr_train_output).then(fn=lambda x: x, inputs=pr_model_path, outputs=pr_model_upload)
    pr_run_btn.click(fn=_validate_image, inputs=pr_image_input, outputs=None).then(fn=_run_pixel_reference, inputs=[pr_model_upload, pr_image_input, pr_threshold], outputs=[pr_overlay, pr_heatmap, pr_result])
    knn_train_btn.click(fn=_train_patch_knn, inputs=[knn_data_dir, knn_image_size, knn_patch_size, knn_stride, knn_model_path], outputs=knn_train_output).then(fn=lambda x: x, inputs=knn_model_path, outputs=knn_model_upload)
    knn_run_btn.click(fn=_validate_image, inputs=knn_image_input, outputs=None).then(fn=_run_patch_knn, inputs=[knn_model_upload, knn_image_input, knn_threshold], outputs=[knn_overlay, knn_heatmap, knn_result])
    demo_run_btn.click(fn=_validate_image, inputs=demo_image_input, outputs=None).then(fn=_run_patch_knn, inputs=[gr.State(DEMO_MODEL_PATH), demo_image_input, gr.State(0.0)], outputs=[demo_overlay, demo_heatmap, demo_result])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7861, share=False, theme=gr.themes.Soft())