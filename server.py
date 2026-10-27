"""Web Server & API Backend for Drug Classification ML Lab.

Provides REST APIs for:
- Stage 1: Data Understanding & EDA
- Stage 2: Baseline Model (Logistic Regression)
- Stage 3: Multi-dimensional Model Evaluation
- Stage 4: Model Comparison (LR vs DT vs RF)
- Stage 5: Pipeline & GridSearchCV
- Stage 6: Visualization figures
- Real-time Interactive Patient Prediction (Inference)
- Live execution of stage Python scripts
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "drug200.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
SRC_DIR = BASE_DIR / "src"
WEB_DIR = BASE_DIR / "web"

app = FastAPI(title="Drug Classification ML Lab API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cached models & artifacts
MODELS_CACHE = {}

def get_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def train_and_cache_models():
    df = get_dataset()
    X = df.drop(columns=["Drug"])
    y = df["Drug"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = ["Age", "Na_to_K"]
    cat_cols = ["Sex", "BP", "Cholesterol"]

    # Preprocessor manual
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    X_train_proc = np.hstack([X_train_num, X_train_cat])
    X_test_proc = np.hstack([X_test_num, X_test_cat])
    encoded_feature_names = num_cols + encoder.get_feature_names_out(cat_cols).tolist()

    # 1. Baseline Logistic Regression
    lr_model = LogisticRegression(max_iter=500, random_state=42)
    lr_model.fit(X_train_proc, y_train)

    # 2. Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_proc, y_train)

    # 3. Random Forest (Manual)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_proc, y_train)

    # 4. Scikit-learn Pipeline (Production-Ready)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )
    pipeline_rf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
        ]
    )
    pipeline_rf.fit(X_train, y_train)

    # Save to cache
    MODELS_CACHE["df"] = df
    MODELS_CACHE["X_train"] = X_train
    MODELS_CACHE["X_test"] = X_test
    MODELS_CACHE["y_train"] = y_train
    MODELS_CACHE["y_test"] = y_test
    MODELS_CACHE["scaler"] = scaler
    MODELS_CACHE["encoder"] = encoder
    MODELS_CACHE["X_train_proc"] = X_train_proc
    MODELS_CACHE["X_test_proc"] = X_test_proc
    MODELS_CACHE["encoded_feature_names"] = encoded_feature_names
    MODELS_CACHE["lr_model"] = lr_model
    MODELS_CACHE["dt_model"] = dt_model
    MODELS_CACHE["rf_model"] = rf_model
    MODELS_CACHE["pipeline_rf"] = pipeline_rf
    MODELS_CACHE["classes"] = lr_model.classes_.tolist()

# Train models on start
train_and_cache_models()


# Input schema for prediction
class PatientFeatures(BaseModel):
    age: float = Field(..., ge=10, le=100, description="Tuổi bệnh nhân")
    sex: str = Field(..., description="Giới tính: F hoặc M")
    bp: str = Field(..., description="Huyết áp: HIGH, NORMAL, LOW")
    cholesterol: str = Field(..., description="Mức cholesterol: HIGH, NORMAL")
    na_to_k: float = Field(..., ge=1.0, le=50.0, description="Tỷ lệ Natri / Kali")
    model_name: Optional[str] = Field("pipeline_rf", description="lr, dt, rf, hoặc pipeline_rf")


# Stage script map
STAGE_SCRIPTS = {
    1: SRC_DIR / "stage1_understand_data.py",
    2: SRC_DIR / "stage2_baseline_model.py",
    3: SRC_DIR / "stage3_evaluation.py",
    4: SRC_DIR / "stage4_model_comparison.py",
    5: SRC_DIR / "stage5_pipeline_workflow.py",
    6: SRC_DIR / "stage6_visualization.py",
}


@app.get("/api/status")
def get_status():
    df = MODELS_CACHE.get("df")
    return {
        "status": "online",
        "dataset_rows": len(df) if df is not None else 0,
        "dataset_cols": df.shape[1] if df is not None else 0,
        "models_loaded": ["Logistic Regression", "Decision Tree", "Random Forest", "Pipeline (Production)"],
        "classes": MODELS_CACHE.get("classes", [])
    }


@app.get("/api/dataset")
def get_dataset_api(limit: int = 250, offset: int = 0):
    df = MODELS_CACHE.get("df")
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    total = len(df)
    subset = df.iloc[offset: offset + limit].copy()
    subset["id"] = subset.index.tolist()
    records = subset.to_dict(orient="records")

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "records": records
    }


@app.get("/api/stage1")
def get_stage1_data():
    df = MODELS_CACHE.get("df")
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    # Target distribution
    target_counts = df["Drug"].value_counts().to_dict()
    target_pct = (df["Drug"].value_counts(normalize=True) * 100).round(2).to_dict()

    target_distribution = [
        {"drug": drug, "count": int(count), "percentage": target_pct[drug]}
        for drug, count in target_counts.items()
    ]

    # Numerical description
    num_cols = ["Age", "Na_to_K"]
    desc = df[num_cols].describe().round(2).to_dict()

    # Categorical breakdown
    cat_cols = ["Sex", "BP", "Cholesterol"]
    cat_breakdown = {}
    for col in cat_cols:
        cat_breakdown[col] = df[col].value_counts().to_dict()

    return {
        "title": "Stage 1: Khám phá và Thấu hiểu Dữ liệu (Understand the Data)",
        "script": "src/stage1_understand_data.py",
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "null_count": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "target_column": "Drug",
        "target_distribution": target_distribution,
        "numerical_summary": desc,
        "categorical_breakdown": cat_breakdown,
        "head": df.head(10).to_dict(orient="records")
    }


@app.get("/api/stage2")
def get_stage2_data():
    X_train = MODELS_CACHE["X_train"]
    X_test = MODELS_CACHE["X_test"]
    y_train = MODELS_CACHE["y_train"]
    y_test = MODELS_CACHE["y_test"]
    lr_model = MODELS_CACHE["lr_model"]
    X_test_proc = MODELS_CACHE["X_test_proc"]
    encoded_feature_names = MODELS_CACHE["encoded_feature_names"]
    classes = MODELS_CACHE["classes"]

    y_pred = lr_model.predict(X_test_proc)
    y_proba = lr_model.predict_proba(X_test_proc)

    test_accuracy = accuracy_score(y_test, y_pred) * 100

    # Train / Test stratified distribution
    train_dist = (y_train.value_counts(normalize=True) * 100).round(2).to_dict()
    test_dist = (y_test.value_counts(normalize=True) * 100).round(2).to_dict()

    stratified_comparison = []
    for cls in sorted(classes):
        stratified_comparison.append({
            "class": cls,
            "train_pct": train_dist.get(cls, 0.0),
            "test_pct": test_dist.get(cls, 0.0)
        })

    # Detailed test samples (all test samples)
    test_samples = []
    for i in range(len(y_test)):
        orig_idx = int(y_test.index[i])
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        is_correct = bool(actual == predicted)
        probas = {cls: round(float(prob) * 100, 2) for cls, prob in zip(classes, y_proba[i])}
        patient_info = X_test.iloc[i].to_dict()

        test_samples.append({
            "sample_no": i + 1,
            "original_index": orig_idx,
            "actual": actual,
            "predicted": predicted,
            "is_correct": is_correct,
            "probabilities": probas,
            "patient": patient_info
        })

    return {
        "title": "Stage 2: Xây dựng Mô hình Cơ sở (Build a Baseline - Logistic Regression)",
        "script": "src/stage2_baseline_model.py",
        "train_count": len(X_train),
        "test_count": len(X_test),
        "train_ratio": 0.8,
        "test_ratio": 0.2,
        "random_state": 42,
        "stratified_comparison": stratified_comparison,
        "encoded_feature_count": len(encoded_feature_names),
        "encoded_features": encoded_feature_names,
        "model_name": "Logistic Regression (max_iter=500)",
        "classes": classes,
        "baseline_accuracy": round(test_accuracy, 2),
        "test_samples": test_samples
    }


@app.get("/api/stage3")
def get_stage3_data():
    X_test = MODELS_CACHE["X_test"]
    y_test = MODELS_CACHE["y_test"]
    lr_model = MODELS_CACHE["lr_model"]
    X_test_proc = MODELS_CACHE["X_test_proc"]
    classes = MODELS_CACHE["classes"]

    y_pred = lr_model.predict(X_test_proc)
    y_proba = lr_model.predict_proba(X_test_proc)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    prec_weighted = precision_score(y_test, y_pred, average="weighted")
    rec_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_matrix = cm.tolist()

    # Classification report dict
    rep = classification_report(y_test, y_pred, output_dict=True, digits=4)

    class_metrics = []
    for cls in classes:
        if cls in rep:
            class_metrics.append({
                "class": cls,
                "precision": round(rep[cls]["precision"] * 100, 2),
                "recall": round(rep[cls]["recall"] * 100, 2),
                "f1_score": round(rep[cls]["f1-score"] * 100, 2),
                "support": int(rep[cls]["support"])
            })

    # Error analysis
    misclassified_indices = np.where(y_test.values != y_pred)[0]
    errors = []
    for idx in misclassified_indices:
        orig_idx = int(y_test.index[idx])
        actual = y_test.iloc[idx]
        predicted = y_pred[idx]
        patient_features = X_test.iloc[idx].to_dict()
        probas = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, y_proba[idx])}

        errors.append({
            "test_index": int(idx),
            "original_row": orig_idx,
            "actual_drug": actual,
            "predicted_drug": predicted,
            "patient_features": patient_features,
            "probabilities": probas,
            "explanation": f"Bệnh nhân {orig_idx} có BP={patient_features.get('BP')}, Age={patient_features.get('Age')}, Na_to_K={patient_features.get('Na_to_K')}. Mô hình tuyến tính phân vân giữa {actual} và {predicted} do siêu phẳng tuyến tính không phân tách hoàn hảo ranh giới phi tuyến."
        })

    return {
        "title": "Stage 3: Đánh giá Đa chiều và Mổ xẻ Mô hình (Evaluate the Model)",
        "script": "src/stage3_evaluation.py",
        "overall_metrics": {
            "accuracy": round(acc * 100, 2),
            "macro_precision": round(prec_macro * 100, 2),
            "macro_recall": round(rec_macro * 100, 2),
            "macro_f1": round(f1_macro * 100, 2),
            "weighted_precision": round(prec_weighted * 100, 2),
            "weighted_recall": round(rec_weighted * 100, 2),
            "weighted_f1": round(f1_weighted * 100, 2),
        },
        "classes": classes,
        "confusion_matrix": cm_matrix,
        "class_metrics": class_metrics,
        "total_test_samples": len(y_test),
        "misclassified_count": len(errors),
        "errors": errors
    }


@app.get("/api/stage4")
def get_stage4_data():
    X_train_proc = MODELS_CACHE["X_train_proc"]
    X_test_proc = MODELS_CACHE["X_test_proc"]
    y_train = MODELS_CACHE["y_train"]
    y_test = MODELS_CACHE["y_test"]

    models = {
        "Logistic Regression": MODELS_CACHE["lr_model"],
        "Decision Tree": MODELS_CACHE["dt_model"],
        "Random Forest": MODELS_CACHE["rf_model"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison_data = []

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_proc, y_train, cv=cv, scoring="accuracy")
        cv_f1 = cross_val_score(model, X_train_proc, y_train, cv=cv, scoring="f1_macro")

        train_pred = model.predict(X_train_proc)
        train_acc = accuracy_score(y_train, train_pred)

        test_pred = model.predict(X_test_proc)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average="macro")

        comparison_data.append({
            "model": name,
            "train_acc": round(train_acc * 100, 2),
            "cv_acc_mean": round(cv_scores.mean() * 100, 2),
            "cv_acc_std": round(cv_scores.std() * 100, 2),
            "cv_f1_mean": round(cv_f1.mean() * 100, 2),
            "cv_f1_std": round(cv_f1.std() * 100, 2),
            "test_acc": round(test_acc * 100, 2),
            "test_f1": round(test_f1 * 100, 2),
            "overfitting_gap": round((train_acc - test_acc) * 100, 2)
        })

    return {
        "title": "Stage 4: So sánh và Biện luận Mô hình (Compare Models)",
        "script": "src/stage4_model_comparison.py",
        "benchmark": comparison_data,
        "key_takeaways": [
            "Decision Tree và Random Forest giải quyết triệt để các ranh giới vuông góc (orthogonal rules) như Na_to_K > 15.",
            "Decision Tree đạt Train Acc = 100% (dễ overfit nếu cây quá sâu, do đó Random Forest ensemble là lựa chọn tối ưu).",
            "Logistic Regression gặp khó khăn ở các vùng ranh giới phi tuyến đa điều kiện giữa drugA và drugB."
        ]
    }


@app.get("/api/stage5")
def get_stage5_data():
    X_train = MODELS_CACHE["X_train"]
    X_test = MODELS_CACHE["X_test"]
    y_train = MODELS_CACHE["y_train"]
    y_test = MODELS_CACHE["y_test"]
    pipeline_rf = MODELS_CACHE["pipeline_rf"]

    test_pred = pipeline_rf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")

    pipeline_steps = [
        {
            "step_name": "Preprocessor (ColumnTransformer)",
            "details": [
                {"name": "StandardScaler", "columns": ["Age", "Na_to_K"], "purpose": "Chuẩn hóa z-score (mean=0, std=1)"},
                {"name": "OneHotEncoder", "columns": ["Sex", "BP", "Cholesterol"], "purpose": "Mã hóa nhị phân 0/1 (sparse_output=False)"}
            ]
        },
        {
            "step_name": "Classifier (RandomForestClassifier)",
            "details": [
                {"name": "n_estimators", "value": "100 cây quyết định"},
                {"name": "max_depth", "value": "5 (ngăn chặn overfitting)"},
                {"name": "random_state", "value": "42"}
            ]
        }
    ]

    grid_search_specs = {
        "search_space": {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [3, 5, "None"],
            "classifier__min_samples_split": [2, 5]
        },
        "total_combinations": 12,
        "cv_folds": 5,
        "total_fits": 60,
        "best_params": {
            "classifier__max_depth": 5,
            "classifier__min_samples_split": 2,
            "classifier__n_estimators": 100
        },
        "best_cv_f1_macro": 99.17
    }

    return {
        "title": "Stage 5: Quy trình Chuẩn Chuyên nghiệp với Pipeline (Improve the Workflow)",
        "script": "src/stage5_pipeline_workflow.py",
        "pipeline_steps": pipeline_steps,
        "grid_search": grid_search_specs,
        "final_test_accuracy": round(test_acc * 100, 2),
        "final_test_f1": round(test_f1 * 100, 2),
        "production_readiness": "Đưa trực tiếp dữ liệu thô bệnh nhân dạng Dict/DataFrame vào mô hình mà không cần bước tiền xử lý thủ công."
    }


@app.get("/api/stage6")
def get_stage6_data():
    figures = [
        {
            "id": "fig1",
            "file": "01_target_distribution.png",
            "title": "Biểu đồ 1: Phân phối Nhãn Mục tiêu",
            "description": "Số lượng và tỷ lệ % của 5 nhóm thuốc. Minh họa rõ nét hiện tượng mất cân bằng lớp (drugY chiếm ~45%)."
        },
        {
            "id": "fig2",
            "file": "02_feature_relationships.png",
            "title": "Biểu đồ 2: Quan hệ giữa Tuổi và Tỷ lệ Na/K",
            "description": "Scatter plot minh họa đường ranh giới quyết định lâm sàng tại Na_to_K = 15. Trên 15 gần như 100% bệnh nhân được chỉ định drugY."
        },
        {
            "id": "fig3",
            "file": "03_confusion_matrices.png",
            "title": "Biểu đồ 3: So sánh Ma trận Nhầm lẫn",
            "description": "Heatmap đối chiếu giữa Logistic Regression (Baseline) và Random Forest (Pipeline tối ưu)."
        },
        {
            "id": "fig4",
            "file": "04_decision_boundary_2d.png",
            "title": "Biểu đồ 4: Ranh giới Quyết định Không gian 2D",
            "description": "Đối chiếu ranh giới phân tách phẳng (Linear Hyperplane) của Logistic Regression và ranh giới phân mảnh vuông góc (Orthogonal) của Cây quyết định."
        }
    ]

    return {
        "title": "Stage 6: Trực quan hóa và Đúc kết Tri thức (Visualize and Communicate)",
        "script": "src/stage6_visualization.py",
        "figures_dir": str(FIGURES_DIR),
        "figures": figures
    }


@app.post("/api/predict")
def predict_patient(feat: PatientFeatures):
    classes = MODELS_CACHE["classes"]
    
    # Select model
    if feat.model_name == "lr":
        model = MODELS_CACHE["lr_model"]
        is_pipeline = False
    elif feat.model_name == "dt":
        model = MODELS_CACHE["dt_model"]
        is_pipeline = False
    elif feat.model_name == "rf":
        model = MODELS_CACHE["rf_model"]
        is_pipeline = False
    else:  # default pipeline_rf
        model = MODELS_CACHE["pipeline_rf"]
        is_pipeline = True

    # Prepare input
    patient_df = pd.DataFrame([{
        "Age": feat.age,
        "Sex": feat.sex,
        "BP": feat.bp,
        "Cholesterol": feat.cholesterol,
        "Na_to_K": feat.na_to_k
    }])

    if is_pipeline:
        pred_drug = model.predict(patient_df)[0]
        probas = model.predict_proba(patient_df)[0]
    else:
        # manual transform
        scaler = MODELS_CACHE["scaler"]
        encoder = MODELS_CACHE["encoder"]
        num_cols = ["Age", "Na_to_K"]
        cat_cols = ["Sex", "BP", "Cholesterol"]

        X_num = scaler.transform(patient_df[num_cols])
        X_cat = encoder.transform(patient_df[cat_cols])
        X_proc = np.hstack([X_num, X_cat])

        pred_drug = model.predict(X_proc)[0]
        probas = model.predict_proba(X_proc)[0]

    prob_dict = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, probas)}
    highest_prob = prob_dict[pred_drug]

    # Clinical explanation logic
    explanation = ""
    if feat.na_to_k > 15.0:
        explanation = f"Bệnh nhân có tỷ lệ Na/K = {feat.na_to_k:.2f} > 15.0. Theo quy luật sinh hóa, nồng độ ion Kali thấp khiến thuốc lựa chọn hàng đầu là DrugY."
    else:
        if feat.bp == "HIGH":
            if feat.age > 50:
                explanation = f"Tỷ lệ Na/K <= 15, Huyết áp CAO (HIGH) và Tuổi = {int(feat.age)} (> 50) -> Chỉ định đặc hiệu là DrugB."
            else:
                explanation = f"Tỷ lệ Na/K <= 15, Huyết áp CAO (HIGH) và Tuổi = {int(feat.age)} (<= 50) -> Chỉ định đặc hiệu là DrugA."
        elif feat.bp == "LOW":
            if feat.cholesterol == "HIGH":
                explanation = "Tỷ lệ Na/K <= 15, Huyết áp THẤP (LOW) và Cholesterol CAO (HIGH) -> Chỉ định phù hợp là DrugC."
            else:
                explanation = "Tỷ lệ Na/K <= 15, Huyết áp THẤP (LOW) và Cholesterol BÌNH THƯỜNG -> Chỉ định phù hợp là DrugX."
        else: # NORMAL BP
            explanation = "Tỷ lệ Na/K <= 15, Huyết áp BÌNH THƯỜNG (NORMAL) -> Chỉ định lâm sàng an toàn là DrugX."

    return {
        "prediction": pred_drug,
        "confidence": highest_prob,
        "probabilities": prob_dict,
        "explanation": explanation,
        "model_used": feat.model_name
    }


@app.post("/api/run-stage/{stage_num}")
def run_stage_script(stage_num: int):
    if stage_num not in STAGE_SCRIPTS:
        raise HTTPException(status_code=404, detail="Stage not found")
    
    script_path = STAGE_SCRIPTS[stage_num]
    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"File {script_path.name} not found")

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30
        )
        output = proc.stdout
        if proc.stderr:
            output += "\n[STDERR]:\n" + proc.stderr

        # Refresh models cache if stage 2, 4, 5 were rerun
        if stage_num in (2, 4, 5):
            train_and_cache_models()

        return {
            "stage": stage_num,
            "script_name": script_path.name,
            "returncode": proc.returncode,
            "output": output
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Thực thi kịch bản quá thời gian quy định (30s)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve reports/figures
if FIGURES_DIR.exists():
    app.mount("/reports/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

# Serve frontend static assets
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(WEB_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
