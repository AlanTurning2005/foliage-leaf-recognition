"""
Module tiền xử lý ảnh và tách mask lá cây theo đúng code gốc 
"""
from typing import Optional, Tuple
import cv2
import numpy as np


def resize_maintain_aspect_ratio(image: np.ndarray, max_dim: int = 512) -> np.ndarray:
    """Co giãn ảnh sao cho cạnh lớn nhất không vượt quá max_dim, giữ nguyên tỷ lệ."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image

    scale = max_dim / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def segment_leaf(image_path: str, max_dim: int = 512) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Tách nền ảnh lá cây theo phương pháp Otsu + nhận diện 4 góc từ bản code gốc.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # 1. Resize an toàn giữ tỷ lệ
    img = resize_maintain_aspect_ratio(img, max_dim=max_dim)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Khử nhiễu nhẹ
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Phân ngưỡng Otsu chuẩn
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Tự phát hiện màu nền (dựa vào 4 góc ảnh)
    h, w = binary.shape
    corners = [binary[0, 0], binary[0, w - 1], binary[h - 1, 0], binary[h - 1, w - 1]]
    if np.mean(corners) > 127:  # Nền trắng -> đảo bit để thân lá thành màu trắng (255)
        binary = cv2.bitwise_not(binary)

    # 5. Lấp khoảng trống ruột lá
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 6. Tìm contour lá ngoài cùng
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    # 7. Tạo mask sạch và cắt ảnh masked
    mask_clean = np.zeros_like(binary)
    cv2.drawContours(mask_clean, [cnt], -1, 255, thickness=cv2.FILLED)
    img_masked = cv2.bitwise_and(img, img, mask=mask_clean)

    return img_masked, img, gray, mask_clean, cnt