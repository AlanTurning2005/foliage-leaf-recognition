import sys
import numpy as np
import pandas as pd
import time
from typing import Any, Dict, Tuple
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Dimensionality Reduction
from sklearn.decomposition import PCA

# Models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    make_scorer
)

from sklearn.base import BaseEstimator, ClassifierMixin

from configs.config import FEATURE_COLS
from src.models.classical import PNNClassifier

# build modules
def get_default_scorers() -> Dict[str, Any]:
    """
    Tạo bộ scorer dùng chung cho các models.
    """
    return {
        "accuracy": make_scorer(accuracy_score),
        "macro_f1": make_scorer(f1_score, average="macro"),
        "macro_precision": make_scorer(precision_score, average="macro", zero_division=0),
        "macro_recall": make_scorer(recall_score, average="macro", zero_division=0)
    }


def train_evaluate_model(model_name: str,
        pipeline : Any,
        param_grid : Dict[str, list],
        X_train : Any,
        y_train: Any, 
        X_test: Any,
        y_test: Any, 
        cv_splits:int = 5,
        random_state: int = 2026) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle= True, random_state= random_state)
    scorers  = get_default_scorers()
    grid = GridSearchCV(
        estimator=pipeline, 
        param_grid= param_grid,
        cv= cv,
        scoring= scorers,
        refit= "accuracy",
        n_jobs= 2,
        verbose = 2,
        return_train_score= True
    )
    # do thoi gian huan luyen
    start_time = time.perf_counter()
    grid.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    # do thoi gian du doan
    best_estimator = grid.best_estimator_
    start_time = time.perf_counter()
    y_pred = best_estimator.predict(X_test)
    test_time = time.perf_counter() - start_time


    acc = accuracy_score(y_test, y_pred)
    macro_p = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    ms_per_img = (test_time / len(X_test)) * 1000

    print(f"-> Tham số tối ưu: {grid.best_params_}")
    print(f"-> Accuracy (Test): {acc * 100:.2f}% | Macro F1: {macro_f1:.4f}")
    print(f"-> Train time: {train_time:.2f}s | Test latency: {ms_per_img:.2f} ms/ảnh")

    metrics_summary = {
        "Model": model_name,
        "Best_Params": str(grid.best_params_),
        "CV_Accuracy": round(grid.best_score_ * 100, 2),
        "Test_Accuracy": round(acc * 100, 2),
        "Macro_Precision": round(macro_p, 4),
        "Macro_Recall": round(macro_r, 4),
        "Macro_F1": round(macro_f1, 4),
        "Train_Time_Sec": round(train_time, 2),
        "Test_Time_Sec": round(test_time, 4),
        "Latency_MS_Per_Img": round(ms_per_img, 2)
    }

    # Báo cáo theo từng lớp lá
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df_class_report = pd.DataFrame(report_dict).transpose()
    df_class_report["Model"] = model_name

    # Lịch sử GridSearch
    df_cv_results = pd.DataFrame(grid.cv_results_)
    df_cv_results["Model"] = model_name    

    return metrics_summary, df_class_report, df_cv_results



# nap du lieu
def run_experiment():
    data_dir = ROOT_DIR / "data" / "processed" # C:\Users\DANH\Desktop\DS_full\Project_leaves_recognition\data\processed\df_dataset.csv
    train_file = data_dir / "df_dataset.csv"
    test1_file = data_dir / "df_testing_1.csv"
    test2_file = data_dir / "df_testing_2.csv"


    df_train = pd.read_csv(train_file)
    df_test1 = pd.read_csv(test1_file)
    df_test2 = pd.read_csv(test2_file)
    df_test = pd.concat([df_test1, df_test2], ignore_index = True)
    # sua loi ten label
    df_train["label"] = df_train["label"].replace({"jd": "jg"})
    df_train["label"] = df_train["label"].str.strip()
    df_test["label"] = df_test["label"].str.strip()
    # label mapping
    labels  = sorted(df_train['label'].unique())
    label_to_idx = {l : i for i, l in enumerate(labels)}
    # xay dung X,y train/ test

    y_train = df_train['label'].map(label_to_idx).values
    y_test = df_test['label'].map(label_to_idx).values

    # scaling standard scaling 
    X_train  = df_train[FEATURE_COLS]
    X_test = df_test[FEATURE_COLS]
    scaler = MinMaxScaler(feature_range = (0,1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"-> Tập Train: {X_train_scaled.shape[0]} mẫu, {X_train_scaled.shape[1]} đặc trưng")
    print(f"-> Tập Test : {X_test_scaled.shape[0]} mẫu ({len(df_test1)} test1 + {len(df_test2)} test2)")
    print(f"-> Số lượng loài (classes): {len(labels)}")

    """"
     BAT DAU HUAN LUYEN MO HINH
    
    1. SVM
    """

    all_summaries = []
    all_class_reports = []
    all_cv_results = []
    svm_pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("pca", PCA(random_state = 2026)),
        ("svc", SVC(random_state= 2026))
    ])

    svm_param_grid = {
        "pca__n_components": [30,40, 50, 57],
        "pca__whiten": [False, True],
        "svc__C": [1, 10, 100],
        "svc__gamma": ["scale", 0.01],
        "svc__kernel": ["rbf"]
    }
    summary_svm, report_svm, svm_cv_result = train_evaluate_model("SVM (RBF)",
                                                   svm_pipeline, svm_param_grid,
                                                   X_train = X_train, y_train= y_train, X_test= X_test, y_test = y_test)
    all_summaries.append(summary_svm)
    all_class_reports.append(report_svm)
    all_cv_results.append(svm_cv_result)

# 2. Huấn luyện KNN 
    knn_pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("pca", PCA(random_state=42)),
        ("knn", KNeighborsClassifier(metric="euclidean"))
    ])
    knn_grid = {
        "pca__n_components": [15, 30, 57],
        "knn__n_neighbors": [1, 3, 5, 7],
        "knn__weights": ["uniform", "distance"]
    }
    summary_knn, report_knn, knn_cv_result = train_evaluate_model(
        "KNN", knn_pipeline, knn_grid, X_train = X_train, y_train= y_train, X_test= X_test, y_test = y_test
    )
    all_summaries.append(summary_knn)
    all_class_reports.append(report_knn)
    all_cv_results.append(knn_cv_result)

    # 3. PNNs + PCA. 
    pnn_pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("pca", PCA(random_state= 2026)),
        ("PNN", PNNClassifier())
    ])
    pnn_param_grid = {
        "pca__n_components": [10, 20, 30, 40, 50, 57],
        "PNN__sigma" : [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]
    }
    summary_pnn, report_pnn, pnn_cv_result = train_evaluate_model("PNN + PCA",
                                                pnn_pipeline, pnn_param_grid,
                                                X_train = X_train, y_train= y_train, X_test= X_test, y_test = y_test)

    all_summaries.append(summary_pnn)
    all_class_reports.append(report_pnn)
    all_cv_results.append(pnn_cv_result)

    df_final_metrics = pd.DataFrame(all_summaries)
    df_all_classes = pd.concat(all_class_reports)
    df_all_cv_results = pd.concat(all_cv_results,ignore_index=True)
    return df_final_metrics, df_all_classes, df_all_cv_results

if __name__ == "__main__": 
    # nạp file csv, mapping label, scale, in ra shape.
    df_final_metrics, df_all_classes, df_all_cv_results= run_experiment()  
    # save history
    ROOT_DIR = Path(__file__).resolve().parents[1] if "__file__" in locals() else Path.cwd()
    REPORTS_DIR = ROOT_DIR / "reports"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Định nghĩa đường dẫn các file CSV đầu ra
    metrics_summary_path = REPORTS_DIR / "metrics_summary.csv"
    per_class_metrics_path = REPORTS_DIR / "per_class_metrics.csv"
    cv_results_path = REPORTS_DIR / "cv_result.csv"
# 3. Xuất hai DataFrame ra file CSV
    df_final_metrics.to_csv(metrics_summary_path, index=False, encoding="utf-8")
    df_all_classes.to_csv(per_class_metrics_path, index=False, encoding="utf-8")
    df_all_cv_results.to_csv(cv_results_path, index= False, encoding= "utf-8")

    print(f"\n{'='*60}")
    print(f"  1. Bảng tổng hợp các model : {metrics_summary_path}")
    print(f"  2. Bảng chi tiết 60 classes: {per_class_metrics_path}")
    print(f"{'='*60}\n")
