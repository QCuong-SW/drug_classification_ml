"""Stage 3: Evaluate the Model (Đánh giá Đa chiều và Mổ xẻ Mô hình)"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "drug200.csv"
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Drug"])
    y = df["Drug"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = ["Age", "Na_to_K"]
    cat_cols = ["Sex", "BP", "Cholesterol"]

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    X_train_processed = np.hstack([X_train_num, X_train_cat])
    X_test_processed = np.hstack([X_test_num, X_test_cat])

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)

    print("=" * 65)
    print("1. CÁC ĐỘ ĐO TỔNG QUAN (OVERALL METRICS)")
    print("=" * 65)
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    prec_weighted = precision_score(y_test, y_pred, average="weighted")
    rec_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"- Accuracy (Độ chính xác chung):        {acc * 100:.2f}%")
    print(f"- Macro Precision:                    {prec_macro * 100:.2f}%")
    print(f"- Macro Recall:                       {rec_macro * 100:.2f}%")
    print(f"- Macro F1-Score:                     {f1_macro * 100:.2f}%")
    print(f"- Weighted F1-Score:                  {f1_weighted * 100:.2f}%")

    print("\n" + "=" * 65)
    print("2. BÁO CÁO PHÂN LOẠI CHI TIẾT THEO TỪNG LỚP (CLASSIFICATION REPORT)")
    print("=" * 65)
    print(classification_report(y_test, y_pred, digits=4))

    print("=" * 65)
    print("3. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)")
    print("=" * 65)
    classes = model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Thực tế: {c}" for c in classes],
        columns=[f"Dự đoán: {c}" for c in classes]
    )
    print(cm_df)

    print("\n" + "=" * 65)
    print("4. TRUY VẾT CÁC MẪU BỊ ĐOÁN SAI (ERROR ANALYSIS)")
    print("=" * 65)
    misclassified_indices = np.where(y_test.values != y_pred)[0]
    print(f"Tổng số ca đoán sai trong tập test: {len(misclassified_indices)} / {len(y_test)}")

    for idx in misclassified_indices:
        orig_idx = y_test.index[idx]
        actual_drug = y_test.iloc[idx]
        pred_drug = y_pred[idx]
        patient_info = X_test.iloc[idx].to_dict()
        probas = {cls: f"{p * 100:.1f}%" for cls, p in zip(classes, y_proba[idx])}

        print(f"\n[Bệnh nhân dòng #{orig_idx} trong dataset gốc]:")
        print(f"  - Thông số lâm sàng: {patient_info}")
        print(f"  - Thuốc thực tế:    {actual_drug}")
        print(f"  - Mô hình đoán:     {pred_drug}")
        print(f"  - Phân bổ xác suất: {probas}")

    print("\n" + "=" * 65)
    print("5. BÀI HỌC VỀ MACRO VS WEIGHTED VÀ ACCURACY")
    print("=" * 65)
    print("• Weighted Average bị chi phối mạnh bởi lớp lớn (drugY có nhiều mẫu trong test).")
    print("• Macro Average coi 5 lớp thuốc quan trọng BÌNH ĐẲNG như nhau.")
    print("• Nếu mô hình đoán sai hoàn toàn lớp thiểu số, Accuracy vẫn cao nhưng Macro Recall sẽ giảm mạnh.")


if __name__ == "__main__":
    main()
