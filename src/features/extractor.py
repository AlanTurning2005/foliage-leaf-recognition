"""
Module trích xuất toàn bộ đặc trưng lá cây.
"""
import os
import cv2
from typing import Optional, Dict

from configs.config import FEATURE_COLS
from src.preprocessing.segmentation import segment_leaf
from src.features.pft import extract_pft_features
from src.features.geometric import extract_geometric_features
from src.features.color import extract_color_features
from src.features.vein import extract_vein_features
from src.features.texture import extract_texture_features


class LeafFeatureExtractor:
    def __init__(self, max_dim: int = 512):
        self.max_dim = max_dim

    def extract_single_image(self, image_path: str) -> Optional[Dict[str, float]]:
        processed = segment_leaf(str(image_path), max_dim=self.max_dim)
        if processed is None:
            return None

        img_masked, img_orig, gray, mask, cnt = processed

        # Ngưỡng diện tích nhỏ an toàn (50 pixel) theo đúng code gốc
        if len(cnt) < 5 or cv2.contourArea(cnt) < 50:
            return None

        try:
            pft = extract_pft_features(mask)
            geo = extract_geometric_features(cnt)
            col = extract_color_features(img_orig, mask)
            vein = extract_vein_features(gray, mask)
            tex = extract_texture_features(img_orig, gray, mask)
        except Exception as e:
            print(f"[WARN] Lỗi khi extract {os.path.basename(str(image_path))}: {e}")
            return None

        all_vals = pft + geo + col + vein + tex
        if len(all_vals) != len(FEATURE_COLS):
            return None

        return dict(zip(FEATURE_COLS, all_vals))