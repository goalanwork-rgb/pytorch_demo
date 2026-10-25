"""第一版異常偵測原型的基本流程測試。執行：python -m unittest -v"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from anomaly_detector import ConvAutoencoder, predict_image, train_model


class AnomalyDetectorTests(unittest.TestCase):
    def test_autoencoder_preserves_image_shape(self) -> None:
        import torch

        model = ConvAutoencoder()
        image = torch.rand(2, 3, 64, 64)
        self.assertEqual(model(image).shape, image.shape)

    def test_train_and_predict_writes_heatmap_overlay_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            normal_dir = root / "normal"
            normal_dir.mkdir()

            for index in range(4):
                image = Image.new("RGB", (64, 64), (220, 220, 220))
                draw = ImageDraw.Draw(image)
                draw.rectangle((16, 16, 48, 48), fill=(150, 155, 160), outline=(90, 95, 100))
                image.save(normal_dir / f"normal_{index}.png")

            model_path = root / "model.pt"
            train_model(
                normal_dir,
                model_path,
                epochs=1,
                batch_size=2,
                image_size=64,
                device_name="cpu",
                seed=7,
            )

            result = predict_image(
                model_path,
                normal_dir / "normal_0.png",
                root / "results",
                device_name="cpu",
            )

            self.assertIn(result["status"], {"正常", "異常"})
            self.assertGreaterEqual(result["anomaly_score"], 0)
            self.assertTrue(Path(result["heatmap"]).is_file())
            self.assertTrue(Path(result["overlay"]).is_file())
            self.assertTrue((root / "results" / "normal_0_result.json").is_file())

    def test_invalid_image_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            normal_dir = root / "normal"
            normal_dir.mkdir()
            for index in range(2):
                Image.new("RGB", (32, 32), (200, 200, 200)).save(normal_dir / f"{index}.png")
            with self.assertRaisesRegex(ValueError, "8 倍數"):
                train_model(normal_dir, root / "model.pt", epochs=1, image_size=63)


if __name__ == "__main__":
    unittest.main()