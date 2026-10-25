from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Callable

import streamlit as st
from PIL import Image

from anomaly_detector import predict_image as predict_autoencoder
from patch_knn_detector import predict_image as predict_patch_knn
from pixel_reference_detector import predict_image as predict_pixel_reference


ROOT = Path(__file__).resolve().parent
DEMO_ASSETS = ROOT / "demo_assets"
MAX_IMAGE_PIXELS = 20_000_000
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

DETECTORS: dict[str, tuple[str, str, Callable]] = {
    "卷積自編碼器": (
        "autoencoder.pt",
        "以重建誤差標示與正常影像不同的區域。",
        predict_autoencoder,
    ),
    "逐像素參考法": (
        "比較輸入影像與正常品像素平均值、標準差；需要固定拍攝位置與光線。",
        predict_pixel_reference,
    ),
    "局部區塊 KNN": (
        "以局部區塊特徵最近鄰距離標示不熟悉的區域。",
        predict_patch_knn,
    ),
}


def _read_image(image_bytes: bytes) -> Image.Image:
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("圖片檔案大小不可超過 20 MB。")
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("圖片尺寸過大，請使用總像素不超過 2,000 萬的圖片。")
        return image.convert("RGB")


st.set_page_config(page_title="PyTorch 工業影像異常偵測", page_icon="🔍", layout="wide")
st.title("🔍 工業影像異常偵測 Demo")
st.markdown(
    "以三種不同方法檢查影像，並比較異常分數、熱區圖與疊圖。"
    "\n\n> 這是概念原型；內建模型使用合成資料，展示結果不代表真實產品檢測效能。"
)

method = st.selectbox("選擇偵測方法", list(DETECTORS))
model_name, description, predict = DETECTORS[method]
st.caption(description)

image_source = st.radio("影像來源", ["上傳圖片", "合成正常範例", "合成瑕疵範例"], horizontal=True)
uploaded_file = None
sample_image_path = None

if image_source == "上傳圖片":
    uploaded_file = st.file_uploader(
        "選擇要檢查的圖片",
        type=["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"],
    )
else:
    sample_image_path = DEMO_ASSETS / (
        "demo_clean.png" if image_source == "合成正常範例" else "demo_defect.png"
    )
    if sample_image_path.is_file():
        st.image(str(sample_image_path), caption=image_source, width=320)
    else:
        st.error(f"找不到展示圖片：{sample_image_path.name}")

override_threshold = st.checkbox("自訂判定門檻（不勾選時使用模型門檻）")
threshold = (
    st.number_input("異常門檻", min_value=0.000001, value=0.1, format="%.6f")
    if override_threshold
    else None
)

if st.button("開始檢查", type="primary"):
    model_path = DEMO_ASSETS / model_name
    if not model_path.is_file():
        st.error(f"找不到展示模型：{model_path.name}")
    elif uploaded_file is None and sample_image_path is None:
        st.warning("請先上傳圖片或選擇範例圖片。")
    else:
        try:
            if uploaded_file is not None:
                image = _read_image(uploaded_file.getvalue())
            else:
                image = _read_image(sample_image_path.read_bytes())

            with tempfile.TemporaryDirectory(prefix="pytorch-demo-") as temp_dir:
                temp_path = Path(temp_dir)
                input_path = temp_path / "input.png"
                output_dir = temp_path / "results"
                image.save(input_path, format="PNG")

                if method == "卷積自編碼器":
                    result = predict(model_path, input_path, output_dir, "cpu", threshold)
                else:
                    result = predict(model_path, input_path, output_dir, threshold)

                overlay = Image.open(result["overlay"]).copy()
                heatmap = Image.open(result["heatmap"]).copy()

            st.subheader(f"判定：{result['status']}")
            st.metric("異常分數", f"{float(result['anomaly_score']):.6f}")
            st.caption(f"套用門檻：{float(result['threshold']):.6f}")
            image_column, heatmap_column, overlay_column = st.columns(3)
            with image_column:
                st.image(image, caption="輸入影像", use_container_width=True)
            with heatmap_column:
                st.image(heatmap, caption="異常熱區", use_container_width=True)
            with overlay_column:
                st.image(overlay, caption="熱區疊圖", use_container_width=True)
            st.json(result)
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"無法完成影像檢查：{error}")
