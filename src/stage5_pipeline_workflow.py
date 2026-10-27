"""Stage 5: Improve the Workflow (Quy trình Chuẩn Chuyên nghiệp với Pipeline)

Mục tiêu học tập:
- Đóng gói quy trình tiền xử lý bằng `ColumnTransformer`.
- Kết hợp Preprocessor và Classifier thành một thực thể duy nhất bằng `Pipeline`.
- Chứng minh cách Pipeline ngăn chặn triệt để Data Leakage trong K-Fold Cross-Validation.
- Tinh chỉnh siêu tham số (Hyperparameter Tuning) bằng `GridSearchCV` an toàn trên tập Train.
- Giữ tập Test nguyên vẹn, chỉ chấm điểm đúng 1 lần duy nhất ở bước cuối cùng.
"""

from pathlib import Path
import sys
import pandas as pd
# GridSearchCV có chức năng thử nhiều cấu hình và lọc ra cấu hình nào tốt nhất cho mỗi model 
# StratifiedKFold  sẽ chia dữ liệu thành các fold và giữ tỉ lệ các nhãn tương tự nhau để chạy và đánh giá
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# ColumnTransformerp thì giúp điều phối các công việc với OneHotEncoder, StandardScaler mọt cách nhất quán hơn,tập trunng hơn
# đưa từng nhóm cột đến đúng coonng cụ rồi ghép kết quả
from sklearn.compose import ColumnTransformer
# chuyển cột số sang dạng trục, và đặc trưng khác thành dạng bit 0/1
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# giúp cho workflow ML gọn hơn, dễ bảo trì và chuyên nghiệp hơn thay vì phải tự khai báo thủ công như các stage trc
from sklearn.pipeline import Pipeline
# ensemble là module chứa nhiều thuật toán kết hợp, 
from sklearn.ensemble import RandomForestClassifier
# từ metrics import thêm các thông số tính toán  như tổng hợp báo cáo các thông tin, cũng như điểm chính xác tổng thể 
from sklearn.metrics import classification_report, accuracy_score

# Đảm bảo console Windows in ký tự tiếng Việt UTF-8 chuẩn xác
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

    # Chia train/test ban đầu: Giữ tập Test làm "kho bảo mật"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = ["Age", "Na_to_K"]
    cat_cols = ["Sex", "BP", "Cholesterol"]

    
    # khởi tạo column tranformer.
    print("=" * 65)
    print("1. KHỞI TẠO COLUMNTRANSFORMER")
    print("=" * 65)
    # Tự động điều hướng: Cột số vào StandardScaler, Cột chữ vào OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )
    # in ra các thành phần có trong preprocessor để ktra
    print(preprocessor)

    # bước đóng gói thành một pipeline hoàn chỉnh
    print("\n" + "=" * 65)
    print("2. ĐÓNG GÓI THÀNH PIPELINE HOÀN CHỈNH")
    print("=" * 65)
    # Pipeline xâu chuỗi: [Tiền xử lý] -> [Huấn luyện mô hình]
    # lấy một biến, tham chiếu đến đối tượng pipeline vừa khởi tạo
    full_pipeline = Pipeline(
        # các bước được đóng gói lại để phù hợp với 1 luồng hoạt động duy nhất
        steps=[
            # đặt tên nhãn cho bước preprocessor để tiền xử lý các cột số, cột đặc trưng thành các hệ tương ứng
            ("preprocessor", preprocessor),
            # tương tự, nhưng 
            ("classifier", RandomForestClassifier(random_state=42))
        ]
    )
    # inn ra cấu trúc của pipeline
    print("Cấu trúc của Pipeline:")
    print(full_pipeline)

    print("\n" + "=" * 65)
    print("3. TÌM KIẾM SIÊU THAM SỐ (GRIDSEARCHCV) TRÊN TẬP TRAIN")
    print("=" * 65)

    # tạo ra danh sách các bộ cấu hình để thử nghiệm 
    param_grid = {
        # classifier_n_estimators: bộ này có bao nhiêu cây, 50-100
        "classifier__n_estimators": [50, 100],
        # các cây có độ sâu bao nhiêu
        "classifier__max_depth": [3, 5, None],
        # cannaf bao nhiêu mẫu để được phép phân nhánh 
        "classifier__min_samples_split": [2, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # lọc ra cấu hình nào tốt nhất cho mỗi một model
    # đoạn này cấu hình trước full pipeline + các cấu hinfnh + cũng như chia fold derdtrain
    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        # mục đích là để các cấu hình chạy được song song,1 = 1 cấu hình chạy, -1 = 16 cấu hình chạy, 4 bằng 4 cấu hình chạy 
        n_jobs=-1
    )

    # giờ bất đầu dùng cấu hình cùng method fit để train với X và Y, nhãn và đặc trưng
    print("Đang chạy Grid Search 5-Fold Cross Validation trên tập Train...")
    grid_search.fit(X_train, y_train)

    # đứa ra tham số tối ưu tìm được với
    # best_params_ : chính là tham số tối ưu nhất tìm được
    print(f"✓ Siêu tham số tối ưu tìm được: {grid_search.best_params_}")
    # đưa ra điểm f1 macro trung bình trên 5 folds, nhân cho 100 để ra hệ số /100%
    print(f"✓ Điểm F1-Macro trung bình trên 5 folds: {grid_search.best_score_ * 100:.2f}%")

    print("\n" + "=" * 65)
    print("4. ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST ĐỘC LẬP")
    print("=" * 65)
    best_model = grid_search.best_estimator_

    # Dự đoán trực tiếp từ dữ liệu thô X_test! Pipeline sẽ tự động biến đổi
    test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Độ chính xác (Accuracy) cuối cùng trên Test set: {test_acc * 100:.2f}%\n")
    print("Báo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, test_pred, digits=4))

    print("=" * 65)
    print("TẠI SAO PIPELINE LÀ TIÊU CHUẨN VÀNG TRONG MACHINE LEARNING?")
    print("=" * 65)
    print("1. Tiện lợi: Dữ liệu thô đưa vào pipeline tự động được scale & encode,")
    print("   không cần viết code biến đổi thủ công lặp đi lặp lại.")
    print("2. Chống Data Leakage tuyệt đối: Trong GridSearchCV, ở mỗi fold,")
    print("   StandardScaler chỉ fit trên 4 phần train fold, fold validation thứ 5")
    print("   hoàn toàn không bị lộ thông số.")
    print("3. Sẵn sàng triển khai (Production-ready): Khi có bệnh nhân mới,")
    print("   chỉ cần gọi `best_model.predict(new_patient_dataframe)`.")


if __name__ == "__main__":
    main()
