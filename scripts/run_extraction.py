# scripts/run_extraction.py
import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc project được thêm vào hệ thống tìm kiếm
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from tqdm import tqdm
from configs.config import ALL_COLS
from src.data.dataset_builder import build_initial_df
from src.features.extractor import LeafFeatureExtractor

def process_and_save(input_df: pd.DataFrame, save_path: str):
    extractor = LeafFeatureExtractor()
    rows = []
    
    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc=f"Saving to {Path(save_path).name}"):
        feat_dict = extractor.extract_single_image(row['path'])
        if feat_dict:
            record = {"path": row['path'], "label": row.get('label', None)}
            record.update(feat_dict)
            rows.append(record)
            
    res_df = pd.DataFrame(rows).reindex(columns=ALL_COLS)
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(save_file, index=False)
    print(f"Hoàn tất trích xuất: {save_file} ({len(res_df)} mẫu)")

if __name__ == "__main__":
    # 1. Khai báo chính xác 3 đường dẫn thư mục ảnh của bạn
    DATASET_PATH = Path(r"C:\Users\DANH\Desktop\DS_full\Leaf_recognition\Dataset")
    TEST_1_PATH  = Path(r"C:\Users\DANH\Desktop\DS_full\Leaf_recognition\Testing-1")
    TEST_2_PATH  = Path(r"C:\Users\DANH\Desktop\DS_full\Leaf_recognition\Testing-2")

    tasks = [
        ("Dataset",   DATASET_PATH, ROOT_DIR / "data" / "processed" / "df_dataset.csv"),
        ("Testing-1", TEST_1_PATH,  ROOT_DIR / "data" / "processed" / "df_testing_1.csv"),
        ("Testing-2", TEST_2_PATH,  ROOT_DIR / "data" / "processed" / "df_testing_2.csv"),
    ]

    for name, folder_path, out_csv in tasks:
        print(f"\n==================================================")
        print(f"Xử lý tập: {name}")
        print(f"Đường dẫn folder: {folder_path}")
        print(f"==================================================")

        if not folder_path.exists():
            print(f"Không tìm thấy thư mục: {folder_path}")
            continue

        # BƯỚC QUAN TRỌNG: Quét thư mục ảnh để sinh ra DataFrame
        df_initial = build_initial_df(folder_path)
        print(f"-> Đã quét được {len(df_initial)} ảnh hợp lệ.")

        if len(df_initial) > 0:
            # BƯỚC TIẾP THEO: Trích xuất 57 features và lưu file CSV
            process_and_save(df_initial, out_csv)
        else:
            print("Thư mục tồn tại nhưng không tìm thấy file ảnh hợp lệ nào.")