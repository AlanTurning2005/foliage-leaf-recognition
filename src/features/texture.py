"""
Module trích xuất đặc trưng kết cấu bề mặt (Texture Features)
Dựa trên Kadir et al. (2011) - Dùng biến đổi Lacunarity bậc p = 2, 4, 6.
"""
from typing import List
import cv2
import numpy as np


def _lacunarity_lp(channel_crop: np.ndarray, p: int) -> float:
    """
    Tính chỉ số Lacunarity bậc p trên một kênh ảnh (crop).
    Công thức: Lp = ( (1/MN) * sum(|P_ij / mu - 1|^p) )^(1/p)
    """
    channel = channel_crop.astype(np.float64)
    pixel_count = channel.size
    if pixel_count == 0:
        return 0.0

    mean_val = np.mean(channel)
    if mean_val == 0.0:
        return 0.0

    ratio = (channel / mean_val) - 1.0
    lp = float(np.mean(np.abs(ratio) ** p) ** (1.0 / p))
    return lp


def extract_texture_features(img: np.ndarray, gray: np.ndarray, mask: np.ndarray) -> List[float]:
    """
    Trích xuất 12 đặc trưng kết cấu từ bounding box của lá.

    Args:
        img (np.ndarray): Ảnh BGR gốc.
        gray (np.ndarray): Ảnh grayscale.
        mask (np.ndarray): Binary mask xác định tọa độ biên lá.

    Returns:
        List[float]: 12 giá trị kết cấu [R_p2, R_p4, R_p6, G_p2, ..., gray_p6].
    """
    if img is None or gray is None or mask is None:
        return [0.0] * 12

    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return [0.0] * 12

    # Crop vùng bounding box chứa lá
    b_crop = img[y:y+h, x:x+w, 0]
    g_crop = img[y:y+h, x:x+w, 1]
    r_crop = img[y:y+h, x:x+w, 2]
    gray_crop = gray[y:y+h, x:x+w]

    features: List[float] = []
    # Quét qua thứ tự R, G, B, Gray với mỗi bậc p = 2, 4, 6
    for channel in [r_crop, g_crop, b_crop, gray_crop]:
        for p in [2, 4, 6]:
            features.append(_lacunarity_lp(channel, p))

    return features