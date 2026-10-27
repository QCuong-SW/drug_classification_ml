"""Stage 4: Compare Models (So sánh và Biện luận Mô hình)"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

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

    X_train_proc = np.hstack([X_train_num, X_train_cat])
    X_test_proc = np.hstack([X_test_num, X_test_cat])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print("=" * 70)
    print("BẢNG SO SÁNH HIỆU NĂNG 3 MÔ HÌNH (MODEL COMPARISON)")
    print("=" * 70)

    for name, model in models.items():
        # Đo lường 5-Fold Stratified Cross-Validation trên tập Train
        cv_scores = cross_val_score(model, X_train_proc, y_train, cv=cv, scoring="accuracy")
        cv_f1_scores = cross_val_score(model, X_train_proc, y_train, cv=cv, scoring="f1_macro")

        # Huấn luyện trên toàn bộ tập Train
        model.fit(X_train_proc, y_train)

        # Đánh giá trên Train (đo Overfitting)
        train_pred = model.predict(X_train_proc)
        train_acc = accuracy_score(y_train, train_pred)

        # Đánh giá trên Test độc lập
        test_pred = model.predict(X_test_proc)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average="macro")

        results.append({
            "Model": name,
            "Train Acc (%)": round(train_acc * 100, 2),
            "5-Fold CV Acc (%)": f"{cv_scores.mean() * 100:.2f} ± {cv_scores.std() * 100:.2f}",
            "5-Fold CV F1-Macro (%)": f"{cv_f1_scores.mean() * 100:.2f} ± {cv_f1_scores.std() * 100:.2f}",
            "Test Acc (%)": round(test_acc * 100, 2),
            "Test F1-Macro (%)": round(test_f1 * 100, 2),
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("KIỂM TRA LẠI CÁC CA BỊ ĐOÁN SAI Ở STAGE 3 VỚI CÂY QUYẾT ĐỊNH (DECISION TREE)")
    print("=" * 70)
    dt_model = models["Decision Tree"]
    dt_test_pred = dt_model.predict(X_test_proc)

    for orig_idx in [138, 34]:
        if orig_idx in y_test.index:
            test_pos = list(y_test.index).index(orig_idx)
            actual = y_test.iloc[test_pos]
            pred_lr = models["Logistic Regression"].predict(X_test_proc[[test_pos]])[0]
            pred_dt = dt_test_pred[test_pos]

            print(f"Bệnh nhân #{orig_idx}:")
            print(f"  - Triệu chứng: {X_test.iloc[test_pos].to_dict()}")
            print(f"  - Đáp án thật:          {actual}")
            print(f"  - Logistic Regression:  {pred_lr} ({'✓' if pred_lr == actual else '✗'})")
            print(f"  - Decision Tree:        {pred_dt} ({'✓' if pred_dt == actual else '✗'})")

    print("\n" + "=" * 70)
    print("KẾT LUẬN VỀ HÀNH VI CỦA CÁC MÔ HÌNH")
    print("=" * 70)
    print("1. Decision Tree và Random Forest đạt hiệu năng cao vì bài toán tuân theo các ngưỡng luật (rule-based).")
    print("2. Cây quyết định học các nhát cắt vuông góc (orthogonal splits), trong khi Logistic Regression dùng siêu phẳng tuyến tính.")
    print("3. Cần kiểm soát max_depth để hạn chế Overfitting trên các tập dữ liệu có nhiễu.")


if __name__ == "__main__":
    main()
