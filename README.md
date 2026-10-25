---
title: 工業影像異常偵測 Demo
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
python_version: 3.11
app_file: app.py
pinned: false
---

# PyTorch 學習環境

## Google Colab Notebook

專案包含 [`Colab_Demo.ipynb`](./Colab_Demo.ipynb)，可在 Colab 執行三套偵測器的合成資料 Demo。

1. 在 PowerShell 專案目錄執行以下指令，建立只含三個偵測器的 ZIP：

   ```powershell
   Compress-Archive -Path anomaly_detector.py,pixel_reference_detector.py,patch_knn_detector.py -DestinationPath colab_project.zip -Force
   ```

2. 將 Notebook 開啟於 Google Colab，依序執行儲存格，並在上傳步驟選取 `colab_project.zip`。
3. Notebook 會在 Colab 產生模型、熱區圖、疊圖與 JSON 結果。

## 免費互動式網站（Streamlit Community Cloud）

[`streamlit_app.py`](./streamlit_app.py) 提供三種偵測方法的互動展示，使用 `demo_assets/` 中的合成模型，可上傳圖片或選擇範例圖片。這些模型只用於展示，不代表真實產品檢測效能。

部署前需將程式放在 GitHub **公開**儲存庫（此專案的程式與合成展示資料會公開）：

1. 在 GitHub 建立公開儲存庫，加入專案檔案；不要上傳 `.venv/`、`artifacts/` 或任何私人資料。
2. 在 [Streamlit Community Cloud](https://share.streamlit.io/) 連結 GitHub 儲存庫。
3. 主程式路徑選 `streamlit_app.py`，Python 版本選 3.11，然後部署。

部署網站需連結你自己的 GitHub 帳號；本機工作區目前尚未連到 GitHub 儲存庫。Streamlit Cloud 的免費方案適合此 CPU 展示版；部署服務的免費額度與政策仍以平台當下規則為準。

## Hugging Face Spaces 網頁 Demo

本專案根目錄已包含 Gradio Space 設定。Demo 使用 `demo_assets/patch_knn.pt`（約 59 KB）與合成範例，可上傳圖片並查看正常/異常判定、異常熱區及疊圖。此模型只供介面展示，不代表真實產品檢測效能。

部署步驟：
1. 在 Hugging Face 建立新的 **Gradio** Space，Python 選 3.11。一般 Gradio/Docker 運算 Space 的免費可用性依帳號方案而異。
2. 將本專案檔案上傳至 Space；至少需要 `app.py`、`patch_knn_detector.py`、`requirements.txt`、`README.md` 和整個 `demo_assets/` 資料夾。
3. 等待 Space 安裝依賴並啟動，開啟 Space 網址即可上傳圖片測試。

本機啟動方式：`pip install -r requirements.txt`，再執行 `python app.py`。

## 結構

```
D:\pytorch 專案\
├── .venv\            # 虛擬環境（不要刪除！）
├── activate.bat      # 啟動環境用
├── 01_tensor_basics.py  # 第一堂課：張量基礎
├── app.py            # Gradio 展示頁
├── streamlit_app.py  # Streamlit 免費雲端展示頁
├── Colab_Demo.ipynb  # Google Colab 三套 Demo
├── demo_assets\      # 合成範例圖片與展示模型
├── requirements.txt  # Python 套件
├── anomaly_detector.py  # 第二個專案：工業影像異常偵測原型
├── test_anomaly_detector.py  # 異常偵測原型測試
├── pixel_reference_detector.py  # 第二套：逐像素參考式偵測器
├── test_pixel_reference_detector.py  # 第二套偵測器測試
├── patch_knn_detector.py  # 第三套：局部區塊最近鄰偵測器
├── test_patch_knn_detector.py  # 第三套偵測器測試
└── README.md         # 這個檔案
```

## 快速開始

在 PowerShell 執行：

```powershell
cd "D:\pytorch 專案"
.\activate.bat
python 01_tensor_basics.py
```

## 常用指令

| 指令 | 用途 |
|------|------|
| `.\activate.bat` | 啟動環境 |
| `deactivate` | 離開環境 |
| `python --version` | 確認 Python 版本 |
| `pip list` | 檢視已安裝套件 |

## 已安裝套件

- Python 3.14.7
- torch 2.14.1
- torchvision 0.29.1
- numpy 2.5.4
- pillow 13.0.0

## 硬體

目前使用 CPU（未偵測到 NVIDIA GPU）。學習階段完全沒問題！

## 工業影像異常偵測原型（第一版）

使用 PyTorch 卷積自編碼器，只用「正常品」圖片訓練，推論時以重建誤差產生異常分數與熱區圖。不需下載模型或額外安裝套件，可在 CPU 執行。

### 先跑合成資料 Demo

```powershell
python anomaly_detector.py demo
```

Demo 會在 `artifacts/demo/` 建立合成的正常品與瑕疵影像，訓練模型並輸出熱區、疊圖和 JSON 結果。極小合成資料不保證分類結果正確；程式會在未能同時判對正常/瑕疵範例時顯示提醒。此 Demo 只驗證訓練及輸出流程，不代表真實產品的檢測準確度。

### 使用自己的正常品照片訓練

將正常品照片放入同一資料夾（可含子資料夾），再執行：

```powershell
python anomaly_detector.py train --data-dir data/normal --model-path artifacts/anomaly_model.pt --epochs 20
```

至少需要兩張圖片；建議固定拍攝角度、背景、光源與產品位置，並為正式評估保留未參與訓練的驗證照片。

### 檢查單張影像

```powershell
python anomaly_detector.py predict --model-path artifacts/anomaly_model.pt --image data/check/sample.jpg --output-dir outputs
```

輸出包括 `*_heatmap.png`（異常熱區）、`*_overlay.png`（疊圖）及 `*_result.json`（分數、門檻與判定）。可用 `--threshold` 暫時覆寫門檻；生產使用前應以獨立正常/異常樣本調整，並量測誤報與漏報。

### 執行測試

```powershell
python -m unittest -v
```

### 原型限制

- 這是可執行的概念原型，非已驗證的工業檢測系統；分數門檻由訓練影像估算，尚未以真實產品驗證。
- 重建式模型對光線、背景、拍攝角度及細微瑕疵可能敏感；請確保資料涵蓋實際拍攝變化。
- 圖片會縮放至正方形輸入；若產品細節很小，可提高 `--image-size`（32–512 的 8 倍數），但 CPU 執行會較慢。
- 輸入資料只讀取常見圖片格式（PNG/JPG/BMP/TIFF/WEBP），不執行或修改原始檔。

## 工業影像異常偵測原型（第二套：像素參考法）

此版本與第一套自編碼器不同，不訓練神經網路：只用正常品影像逐像素建立平均值與標準差，推論時計算標準化差異，產生分數、熱區、疊圖與 JSON。適合**固定相機、固定光源、產品位置與影像對位穩定**的簡單檢查；若產品有位移、旋轉或光線變化，可能大量誤報。

### 執行 Demo

```powershell
python pixel_reference_detector.py demo
```

### 訓練與推論

```powershell
python pixel_reference_detector.py train --data-dir data/normal --model-path artifacts/pixel_reference.pt --image-size 128
python pixel_reference_detector.py predict --model-path artifacts/pixel_reference.pt --image data/check/sample.jpg --output-dir outputs/pixel_reference
```

可用 `--threshold` 覆寫推論門檻。預設門檻僅供原型起步，正式使用前須以獨立的正常及異常樣本校準，量測誤報與漏報。此法沒有影像對位功能，也不是已驗證的工業檢測系統。

執行兩套測試：

```powershell
python -m unittest -v
```

## 工業影像異常偵測原型（第三套：局部區塊最近鄰）

第三套不訓練自編碼器，也不逐像素建立固定平均影像；它從正常品影像擷取局部區塊描述子（顏色、局部變化與梯度），建立正常特徵庫，推論時以最近鄰距離標出不熟悉的局部區域。較容易定位局部外觀差異，但仍要求影像尺寸、視角、位置及照明大致一致，且簡化的區塊描述子不是通用視覺模型。

### 執行 Demo

```powershell
python patch_knn_detector.py demo
```

### 訓練與推論

```powershell
python patch_knn_detector.py train --data-dir data/normal --model-path artifacts/patch_knn.pt --image-size 128 --patch-size 8 --stride 4
python patch_knn_detector.py predict --model-path artifacts/patch_knn.pt --image data/check/sample.jpg --output-dir outputs/patch_knn
```

推論會輸出 `*_heatmap.png`、`*_overlay.png` 及 `*_result.json`；可用 `--threshold` 暫時覆寫門檻。區塊大小與步幅可在訓練時調整，較小步幅會增加特徵數量與運算量。預設門檻只是原型起點，正式使用前須以獨立正常／異常影像校準並量測誤報、漏報；合成 Demo 不代表實際檢測效能。

三套偵測器的測試：

```powershell
python -m unittest -v
```
