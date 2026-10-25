"""像素參考式異常偵測器測試。執行：python -m unittest -v"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from pixel_reference_detector import predict_image, train_model


class PixelReferenceDetectorTests(unittest.TestCase):
    def test_train_predict_and_detect_obvious_defect(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            normal_dir = root / "normal"
            normal_dir.mkdir()
            for index, shade in enumerate((210, 211, 209, 212)):
                image = Image.new("RGB", (64, 64), (shade, shade, shade))
                draw = ImageDraw.Draw(image)
                draw.rectangle((16, 16, 48, 48), fill=(150, 155, 160), outline=(90, 95, 100))
                image.save(normal_dir / f"normal_{index}.png")

            model_path = root / "pixel_reference.pt"
            train_model(normal_dir, model_path, image_size=64)
            clean_path = normal_dir / "normal_0.png"
            defect_path = root / "defect.png"
            with Image.open(clean_path) as original:
                defect = original.copy()
            ImageDraw.Draw(defect).rectangle((29, 29, 35, 35), fill=(0, 0, 0))
            defect.save(defect_path)

            clean_result = predict_image(model_path, clean_path, root / "results")
            defect_result = predict_image(model_path, defect_path, root / "results")

            self.assertEqual(clean_result["status"], "正常")
            self.assertEqual(defect_result["status"], "異常")
            self.assertTrue(Path(defect_result["heatmap"]).is_file())
            self.assertTrue(Path(defect_result["overlay"]).is_file())
            result_path = root / "results" / "defect_result.json"
            self.assertEqual(json.loads(result_path.read_text(encoding="utf-8"))["status"], "異常")

    def test_requires_two_normal_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            normal_dir = root / "normal"
            normal_dir.mkdir()
            Image.new("RGB", (32, 32), "white").save(normal_dir / "one.png")
            with self.assertRaisesRegex(ValueError, "至少需要 2 張"):
                train_model(normal_dir, root / "model.pt", image_size=32)


if __name__ == "__main__":
    unittest.main()