import cv2
import numpy as np

def extract_geometric_features(cnt: np.ndarray) -> list:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    _, _, w, h = cv2.boundingRect(cnt)

    slimness = w / h if h > 0 else 0.0
    roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return [slimness, roundness, 0.0]
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    pts = cnt.reshape(-1, 2).astype(float)
    dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    dists = dists[dists > 0]
    # Xử lý bảo vệ tránh outlier vọt lên 250
    dispersion = np.max(dists) / np.min(dists) if len(dists) > 0 and np.min(dists) > 0 else 0.0

    return [slimness, roundness, dispersion]