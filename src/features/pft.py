import numpy as np
from configs.config import PFT_COLS

def extract_pft_features(mask: np.ndarray, m: int = 4, n: int = 6) -> list:
    y_locs, x_locs = np.nonzero(mask)
    if len(y_locs) == 0:
        return [0.0] * len(PFT_COLS)

    cx, cy = np.mean(x_locs), np.mean(y_locs)
    dx, dy = x_locs - cx, y_locs - cy
    rho = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx)

    R = np.max(rho) if np.max(rho) > 0 else 1.0
    T = 2 * np.pi
    PF_00_val = float(len(rho))
    area_norm = 2 * np.pi * (R ** 2)
    fd_00 = PF_00_val / area_norm if area_norm > 0 else 0.0

    fds = [fd_00]
    for r_freq in range(m + 1):
        for i_freq in range(n + 1):
            if r_freq == 0 and i_freq == 0:
                continue
            phase = 2 * np.pi * (r_freq / R * rho + i_freq / T * phi)
            PF_ri = np.sum(np.exp(1j * phase))
            fd = np.abs(PF_ri) / PF_00_val if PF_00_val > 0 else 0.0
            fds.append(fd)

    fds = fds[:len(PFT_COLS)]
    while len(fds) < len(PFT_COLS):
        fds.append(0.0)
    return fds