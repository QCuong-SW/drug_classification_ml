"""Stage 4: Compare Models (So sánh và Biện luận Mô hình)

Mục tiêu học tập:
- Huấn luyện và so sánh 3 thuật toán:
  1. Logistic Regression (Tuyến tính - Linear model)
  2. Decision Tree (Cây quyết định - Phi tuyến, trực quan cao)
  3. Random Forest (Rừng ngẫu nhiên - Ensemble bagging)
- Áp dụng K-Fold Cross-Validation (5-Fold Stratified) trên tập Train để đo lường độ ổn định.
- Phát hiện hiện tượng Overfitting (Học vẹt) khi Train Score = 100% nhưng Test Score thấp hơn.
- Lý giải vì sao Decision Tree / Random Forest giải quyết triệt để các ca mà Logistic Regression đoán sai.
"""

# vẫn thêm vào lib xử lý đường dẫn 
from pathlib import Path
# thực hieejnn các thao tác can thiệp vào quá trình xử lý 
import sys
# thư viện tính toán số học trong python
import numpy as np
# sử dụng pandas để xử lý dự liệu đặc trưng dạng ma trận
import pandas as pd
# ở thư viện sklearn thì chọn ra 3 thằng: train_test_split, StratifiedKFord thì là thiên về chia dữ liệu
# còn cross_val_score thiên về kiểm tra chéo dựa trên tỉ lệ dự đoán , giao ddoonngj nhiều hay ít
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# 2 cách chuẩn hóa thành dạng trục dannhs cho dữ liệu số, và chuyển các dữ liệu đặc trưng về dạng bit 0,1
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# import model tuyeenns tính LogisticRegression và model dạng cây DecisionTreeClassifier và model dạng nhóm RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# kèm theo import các chỉ số như tỉ lệ đúng tổng thể, tỉ lệ hệ số cân bằng 
from sklearn.metrics import accuracy_score, f1_score

# Đảm bảo console Windows in ký tự tiếng Việt UTF-8 chuẩn xác
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

    # Chia train/test chuẩn
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

    # Định nghĩa 3 mô hình
    models = {
        # chạy lặp 500 lần 
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        # khởi tạo modle cây quyết định, 1 cây duy nhất vs 100 cây và model tuyến tính lặp 500 lần 
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        # sử dụng 100 cây quyết định 
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # K-Fold Cross-Validation 5 folds trên tập train
    # 4 forrd để train, 1 ford để test 
    # shuffle=True -> xáo trộn trước chia 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    print("=" * 70)
    print("BẢNG SO SÁNH HIỆU NĂNG 3 MÔ HÌNH (MODEL COMPARISON)")
    print("=" * 70)

    # duyệt từng name và model 
    for name, model in models.items():
        # 1. Đo lường Cross-Validation trên tập Train
        # scoring ="accuracy" -> mức độ đúng trung bình toonrhg thể
        cv_scores = cross_val_score(model, X_train_proc, y_train, cv=cv, scoring="accuracy")
        # scoring ="f1_macro" -> trung bình độ cân bằng phán đoán giữa precision và recall qua các nhãn 
        cv_f1_scores = cross_val_score(model, X_train_proc, y_train, cv=cv, scoring="f1_macro")

        # 2. Huấn luyện trên toàn bộ tập Train
        model.fit(X_train_proc, y_train)

        # 3. Đánh giá trên Train (để phát hiện Overfitting)
        # đánh giá, tính tỉ lệ đánh giá mỗi nhãn thuộc train
        train_pred = model.predict(X_train_proc)
        train_acc = accuracy_score(y_train, train_pred)


        # 4. Đánh giá trên Test (Đề thi độc lập)
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
    print("KIỂM TRA LẠI 2 CA BỊ ĐOÁN SAI Ở STAGE 3 VỚI CÂY QUYẾT ĐỊNH (DECISION TREE)")
    print("=" * 70)
    dt_model = models["Decision Tree"]
    dt_test_pred = dt_model.predict(X_test_proc)

    # 2 mẫu bệnh nhân bị Logistic Regression đoán sai ở Stage 3:
    misclassified_stage2 = [138, 34]
    for orig_idx in misclassified_stage2:
        test_pos = list(y_test.index).index(orig_idx)
        actual = y_test.iloc[test_pos]
        pred_lr = models["Logistic Regression"].predict(X_test_proc[[test_pos]])[0]
        pred_dt = dt_test_pred[test_pos]

        print(f"Bệnh nhân #{orig_idx}:")
        print(f"  - Triệu chứng: {X_test.iloc[test_pos].to_dict()}")
        print(f"  - Đáp án thật:          {actual}")
        print(f"  - Logistic Regression:  {pred_lr} (✗)")
        print(f"  - Decision Tree:        {pred_dt} ({'✓' if pred_dt == actual else '✗'})")

    print("\n" + "=" * 70)
    print("KẾT LUẬN VỀ HÀNH VI CỦA CÁC MÔ HÌNH")
    print("=" * 70)
    print("1. Decision Tree và Random Forest đạt 100% trên tập Test vì bài toán này")
    print("   tuân theo các ngưỡng điều kiện rõ ràng (Rule-based: Na_to_K > 15, Age > 50).")
    print("2. Cây quyết định học các nhát cắt vuông góc (orthogonal splits) cực kỳ hoàn hảo")
    print("   với dạng quy tắc này, trong khi Logistic Regression dùng siêu phẳng tuyến tính.")
    print("3. Lưu ý Overfitting: Decision Tree đạt Train Acc = 100%. Nếu dữ liệu có nhiễu (noise),")
    print("   cây quyết định sẽ rất dễ bị overfit nếu không giới hạn max_depth.")


if __name__ == "__main__":
    main()
