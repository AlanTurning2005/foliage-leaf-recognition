"""
Module trích xuất đặc trưng gân lá (Vein Features)
Dựa trên Kadir et al. (2011) - Dùng phép mở hình thái học (Morphological Opening).
"""
from typing import List
import cv2
import numpy as np


def extract_vein_features(gray: np.ndarray, mask: np.ndarray) -> List[float]:
    """
    Trích xuất 3 đặc trưng tỷ lệ diện tích gân lá V1, V2, V3.

    Args:
        gray (np.ndarray): Ảnh mức xám (Grayscale).
        mask (np.ndarray): Binary mask của lá.

    Returns:
        List[float]: 3 giá trị tỷ lệ gân [V1, V2, V3].
    """
    if gray is None or mask is None:
        return [0.0, 0.0, 0.0]

    # Tổng số pixel thuộc thân lá (A)
    leaf_area = float(np.sum(mask > 0))
    if leaf_area == 0.0:
        return [0.0, 0.0, 0.0]

    features: List[float] = []

    # Quét qua các bán kính 1, 2, 3 tương ứng với kernel kích thước (2*r + 1)
    for radius in [1, 2, 3]:
        kernel_size = 2 * radius + 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Phép mở hình thái học để xóa chi tiết gân nhỏ
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k)
        
        # Trừ ảnh gốc cho ảnh mở để thu lại cấu trúc gân bị mất
        diff = cv2.subtract(gray, opened)
        
        # Đếm số pixel gân xuất hiện bên trong phạm vi mặt nạ lá (A_k)
        vein_pixels = float(np.sum(diff[mask > 0] > 0))
        features.append(vein_pixels / leaf_area)

    return features