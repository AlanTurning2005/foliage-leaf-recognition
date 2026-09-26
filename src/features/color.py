"""
Module trích xuất các đặc trưng mô-men màu sắc (Color Moments)
Dựa trên Kadir et al. (2011) - Gồm Mean, Std, Skewness trên 3 kênh R, G, B.
"""
from typing import List
import cv2
import numpy as np
from scipy.stats import skew


def extract_color_features(img: np.ndarray, mask: np.ndarray) -> List[float]:
    """
    Trích xuất 9 đặc trưng màu sắc từ các pixel nằm trong vùng lá (mask > 0).

    Args:
        img (np.ndarray): Ảnh BGR gốc (chưa nhân mask hoặc đã qua xử lý).
        mask (np.ndarray): Binary mask của lá (255 cho vùng lá, 0 cho nền).

    Returns:
        List[float]: 9 giá trị [R_mean, R_std, R_skew, G_mean, G_std, G_skew, B_mean, B_std, B_skew].
    """
    if img is None or mask is None:
        return [0.0] * 9

    b, g, r = cv2.split(img)
    features: List[float] = []

    # Duyệt tuần tự qua 3 kênh theo đúng thứ tự R, G, B
    for channel in [r, g, b]:
        pixels = channel[mask > 0].astype(np.float64)
        
        # Phòng thủ: Nếu không có pixel lá nào trong mask
        if len(pixels) == 0:
            features.extend([0.0, 0.0, 0.0])
            continue

        mu = float(np.mean(pixels))
        sigma = float(np.std(pixels))
        sk = float(skew(pixels)) if sigma > 0.0 else 0.0
        
        features.extend([mu, sigma, sk])

    return features