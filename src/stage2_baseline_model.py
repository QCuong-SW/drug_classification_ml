"""Stage 2: Build a Baseline (Xây dựng Mô hình Cơ sở)"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "drug200.csv"
    df = pd.read_csv(data_path)

    print("=" * 65)
    print("BƯỚC 1: TÁCH ĐẶC TRƯNG (X) VÀ NHÃN MỤC TIÊU (y)")
    print("=" * 65)
    X = df.drop(columns=["Drug"])
    y = df["Drug"]

    print(f"Kích thước ma trận X (Features): {X.shape} (200 mẫu, 5 đặc trưng)")
    print(f"Kích thước vector y (Target):    {y.shape} (200 nhãn tương ứng)")

    print("\n" + "=" * 65)
    print("BƯỚC 2: CHIA TẬP DỮ LIỆU TRAIN / TEST (80% / 20%)")
    print("=" * 65)
    # stratify=y để giữ nguyên tỷ lệ các nhãn giữa Train và Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"- Tập Train (Học): {X_train.shape[0]} mẫu (80%)")
    print(f"- Tập Test (Thi):  {X_test.shape[0]} mẫu (20%)")

    train_dist = (y_train.value_counts(normalize=True) * 100).round(1)
    test_dist = (y_test.value_counts(normalize=True) * 100).round(1)
    dist_cmp = pd.DataFrame({"Train (%)": train_dist, "Test (%)": test_dist})
    print("\nKiểm tra tỷ lệ phân phối nhãn sau khi stratify:")
    print(dist_cmp)

    print("\n" + "=" * 65)
    print("BƯỚC 3: TIỀN XỬ LÝ (PREPROCESSING) CHỐNG DATA LEAKAGE")
    print("=" * 65)
    num_cols = ["Age", "Na_to_K"]
    cat_cols = ["Sex", "BP", "Cholesterol"]

    # Fit trên Train, Transform trên cả Train và Test
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    X_train_processed = np.hstack([X_train_num, X_train_cat])
    X_test_processed = np.hstack([X_test_num, X_test_cat])

    encoded_feature_names = num_cols + encoder.get_feature_names_out(cat_cols).tolist()
    print(f"Số lượng đặc trưng sau khi mã hóa: {len(encoded_feature_names)}")
    print(f"Danh sách đặc trưng: {encoded_feature_names}")

    print("\n" + "=" * 65)
    print("BƯỚC 4: KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH (LOGISTIC REGRESSION)")
    print("=" * 65)
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train_processed, y_train)
    print("✓ Mô hình đã huấn luyện xong thành công (fit completed)!")
    print(f"Các lớp nhãn mà mô hình đã học được: {model.classes_}")

    print("\n" + "=" * 65)
    print("BƯỚC 5: DỰ ĐOÁN VÀ PHÂN BIỆT predict() VS predict_proba()")
    print("=" * 65)
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)

    print("Soi chi tiết các bệnh nhân trong tập Test:")
    for i in range(min(40, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        probas = {cls: f"{prob * 100:.1f}%" for cls, prob in zip(model.classes_, y_proba[i])}
        is_correct = "✓ ĐÚNG" if actual == predicted else "✗ SAI"
        print(f"\nBệnh nhân #{i+1}:")
        print(f"  - Thực tế:  {actual}")
        print(f"  - Dự đoán:  {predicted} ({is_correct})")
        print(f"  - Xác suất chi tiết: {probas}")

    baseline_acc = (y_pred == y_test).mean() * 100
    print("\n" + "-" * 65)
    print(f"Độ chính xác sơ bộ (Baseline Accuracy) trên tập Test: {baseline_acc:.2f}%")
    print("-" * 65)


if __name__ == "__main__":
    main()
