"""Stage 5: Improve the Workflow (Quy trình Chuẩn Chuyên nghiệp với Pipeline)"""

from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "drug200.csv"
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Drug"])
    y = df["Drug"]

    # Giữ tập Test độc lập để chấm điểm ở bước cuối cùng
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = ["Age", "Na_to_K"]
    cat_cols = ["Sex", "BP", "Cholesterol"]

    print("=" * 65)
    print("1. KHỞI TẠO COLUMNTRANSFORMER")
    print("=" * 65)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )
    print(preprocessor)

    print("\n" + "=" * 65)
    print("2. ĐÓNG GÓI THÀNH PIPELINE HOÀN CHỈNH")
    print("=" * 65)
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42))
        ]
    )
    print("Cấu trúc của Pipeline:")
    print(full_pipeline)

    print("\n" + "=" * 65)
    print("3. TÌM KIẾM SIÊU THAM SỐ (GRIDSEARCHCV) TRÊN TẬP TRAIN")
    print("=" * 65)
    param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [3, 5, None],
        "classifier__min_samples_split": [2, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    print("Đang chạy Grid Search 5-Fold Cross Validation trên tập Train...")
    grid_search.fit(X_train, y_train)

    print(f"✓ Siêu tham số tối ưu tìm được: {grid_search.best_params_}")
    print(f"✓ Điểm F1-Macro trung bình trên 5 folds: {grid_search.best_score_ * 100:.2f}%")

    print("\n" + "=" * 65)
    print("4. ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST ĐỘC LẬP")
    print("=" * 65)
    best_model = grid_search.best_estimator_

    # Dự đoán trực tiếp từ dữ liệu thô X_test (Pipeline tự động transform)
    test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Độ chính xác (Accuracy) cuối cùng trên Test set: {test_acc * 100:.2f}%\n")
    print("Báo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, test_pred, digits=4))

    print("=" * 65)
    print("TẠI SAO PIPELINE LÀ TIÊU CHUẨN VÀNG TRONG MACHINE LEARNING?")
    print("=" * 65)
    print("1. Tiện lợi: Dữ liệu thô đưa vào pipeline tự động được scale & encode.")
    print("2. Chống Data Leakage: Trong GridSearchCV, StandardScaler chỉ fit trên từng train fold.")
    print("3. Sẵn sàng triển khai: Có thể dự đoán trực tiếp dữ liệu thô bệnh nhân mới.")


if __name__ == "__main__":
    main()
