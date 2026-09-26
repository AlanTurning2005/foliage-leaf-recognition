# src/data/dataset_builder.py
import re
from pathlib import Path
import pandas as pd
from configs.config import VALID_EXTENSIONS

def extract_label_from_filename(stem: str) -> str:
    match = re.match(r"^([a-zA-Z]+)", stem)
    return match.group(1) if match else stem.split("_")[0].split("-")[0]

def build_initial_df(path: Path) -> pd.DataFrame:
    """Quét thư mục ảnh và tạo DataFrame ['path', 'label']."""
    if not path.exists():
        print(f"Cảnh báo: Không tìm thấy {path}")
        return pd.DataFrame(columns=["path", "label"])
    
    records = []
    for fp in path.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in VALID_EXTENSIONS and not any(p.startswith(".") for p in fp.parts):
            label = fp.parent.name if fp.parent != path else extract_label_from_filename(fp.stem)
            records.append({"path": str(fp), "label": label})
            
    return pd.DataFrame(records, columns=["path", "label"])