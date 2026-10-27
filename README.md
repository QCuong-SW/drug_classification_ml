# Scikit-learn Learning Lab — Drug Classification

> **Lưu ý quan trọng (Medical Disclaimer):**  
> Đây là dự án học tập (educational project) nhằm rèn luyện tư duy Machine Learning thực chiến từ nền tảng. Dự án **không** phải là hệ thống ra quyết định y khoa hay kê đơn thuốc. Các dự đoán của mô hình tuyệt đối không được coi là chỉ định điều trị y khoa có giá trị lâm sàng.

---

## 1. Mục tiêu dự án (Project Context & Goals)

Dự án này được thiết kế dành cho người học đã nắm vững Python cơ bản, NumPy và Pandas, bắt đầu học Machine Learning từ con số 0 với thư viện Scikit-learn. 

Trọng tâm không phải là "gọi hàm thư viện cho chạy", mà là:
- Nắm vững quy trình phát triển mô hình học có giám sát (Supervised ML Workflow) hoàn chỉnh.
- Hiểu cách xử lý đặc trưng dạng số (numerical) và dạng phân loại (categorical).
- Học cách huấn luyện, đánh giá đa chiều, so sánh và giải thích mô hình phân loại đa lớp (multiclass classification).
- Nhận diện và phòng tránh các lỗi sai kinh điển trong ML (như Data Leakage, Overfitting, lạm dụng Accuracy).
- Xây dựng tư duy khoa học dữ liệu sẵn sàng áp dụng cho các bài toán thực tế phức tạp hơn.

---

## 2. Thông tin bộ dữ liệu (Proposed Dataset)

- **Tên dữ liệu:** Drug Classification (`drug200.csv`).
- **Nguồn gốc:** Bộ dữ liệu mẫu kinh điển từ IBM SPSS Modeler / Watson Studio, phổ biến rộng rãi trong cộng đồng học tập quốc tế (Kaggle, Coursera, IBM Data Science).
- **Giấy phép (License):** CC0 / Open Educational Use.
- **Kích thước dự kiến:** 200 bản ghi (rows), 6 cột (columns).
- **Các đặc trưng (Features):**
  - `Age` (Numerical): Tuổi của bệnh nhân.
  - `Sex` (Categorical): Giới tính (`F`, `M`).
  - `BP` (Categorical/Ordinal): Huyết áp (`HIGH`, `NORMAL`, `LOW`).
  - `Cholesterol` (Categorical): Mức cholesterol (`HIGH`, `NORMAL`).
  - `Na_to_K` (Numerical): Tỷ lệ nồng độ Natri trên Kali trong máu.
- **Nhãn mục tiêu (Target Label):**
  - `Drug` (Multiclass): Loại thuốc thích hợp (`DrugY`, `drugA`, `drugB`, `drugC`, `drugX`).

### Hạn chế & Thiên lệch của dữ liệu (Dataset Limitations & Bias)
1. **Quy mô nhỏ:** Chỉ với 200 mẫu, phương sai khi chia tập dữ liệu (train/test split) có thể lớn nếu không cẩn thận.
2. **Mất cân bằng lớp (Class Imbalance):** `DrugY` chiếm tỷ lệ lớn nhất (~45%), đòi hỏi phải sử dụng kỹ thuật phân tầng (`stratify`) và các chỉ số đánh giá phù hợp (F1-macro, Confusion Matrix) thay vì chỉ nhìn vào Accuracy.
3. **Tính chất dữ liệu đồ chơi (Toy / Synthetic Dataset):** Các chỉ số huyết áp và cholesterol được đơn giản hóa thành các mức rời rạc (`HIGH`, `NORMAL`, `LOW`), không phản ánh đầy đủ tính biến thiên sinh học liên tục trong y học thực tế.

---

## 3. Cấu trúc thư mục dự án (Project Structure)

```text
Drug_classification_ML/
├── data/
│   └── raw/                    # Chứa file dữ liệu gốc (drug200.csv)
├── src/                        # Mã nguồn Python chia theo từng stage học tập
│   ├── __init__.py
│   ├── stage1_understand_data.py
│   ├── stage2_baseline_model.py
│   ├── stage3_evaluation.py
│   ├── stage4_model_comparison.py
│   ├── stage5_pipeline_workflow.py
│   └── stage6_visualization.py
├── reports/
│   └── figures/                # Lưu biểu đồ, ma trận nhầm lẫn và trực quan hóa
├── .gitignore                  # Bỏ qua virtualenv, cache, file tạm hệ điều hành
├── requirements.txt            # Danh sách thư viện tối thiểu cần thiết
└── README.md                   # Tài liệu hướng dẫn & ghi nhận kết quả thực nghiệm
```

---

## 4. Lộ trình học tập 7 giai đoạn (Learning Stages)

- [x] **Stage 1 — Understand the Data (Khám phá dữ liệu):** Đọc dữ liệu, kiểm tra cấu trúc, kiểu dữ liệu, phân phối nhãn, giá trị khuyết thiếu và phát biểu bài toán.
- [x] **Stage 2 — Build a Baseline (Xây dựng mô hình cơ sở):** Tách `X`/`y`, chia tập train/test có `stratify`, mã hóa/chuẩn hóa cơ bản và huấn luyện Logistic Regression. Giải thích cơ chế `fit()`, `predict()`, `predict_proba()`.
- [x] **Stage 3 — Evaluate the Model (Đánh giá mô hình):** Phân tích Confusion Matrix, Accuracy, Precision, Recall, F1-score (Macro vs Weighted). Phân tích mẫu dự đoán sai và lý do tại sao không thể chỉ dựa vào Accuracy.
- [x] **Stage 4 — Compare Models (So sánh mô hình):** Huấn luyện và so sánh Logistic Regression, Decision Tree và Random Forest. Giới thiệu K-Fold Cross-Validation và hiện tượng Overfitting.
- [x] **Stage 5 — Improve the Workflow (Tối ưu quy trình chuẩn):** Sử dụng `Pipeline` và `ColumnTransformer` (`OneHotEncoder`, `StandardScaler`). Giải thích vì sao phải fit trên tập train và cách ngăn chặn Data Leakage triệt để.
- [x] **Stage 6 — Visualize and Communicate (Trực quan hóa & Đúc kết):** Vẽ phân phối đặc trưng, decision regions trên không gian 2D, ghi nhận kết quả thực tế vào báo cáo.
- [ ] **Stage 7 — Real-World Extension (Đề xuất thử thách nâng cao):** Giới thiệu bài toán thực tế có độ phức tạp cao hơn (missing values, imbalance nặng hơn, noisy data) để mở rộng kỹ năng.

---

## 5. Kết quả thực nghiệm thực tế (Actual Experimental Results)

Toàn bộ kết quả dưới đây được ghi nhận trực tiếp từ mã nguồn thực thi trên máy:

### 5.1. Bảng so sánh hiệu năng các mô hình (Model Benchmark)

| Mô hình (Model) | Train Accuracy | 5-Fold CV Accuracy (Train) | 5-Fold CV F1-Macro (Train) | Test Accuracy (Unseen) | Test F1-Macro (Unseen) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 98.75% | 95.00 ± 1.53% | 93.39 ± 2.61% | 95.00% | 92.69% |
| **Decision Tree** | 100.00% | 100.00 ± 0.00% | 100.00 ± 0.00% | 97.50% | 94.18% |
| **Random Forest** | 100.00% | 100.00 ± 0.00% | 100.00 ± 0.00% | 97.50% | 94.18% |
| **Random Forest + Pipeline (GridSearchCV)** | 100.00% | 100.00 ± 0.00% | 100.00 ± 0.00% | **97.50%** | **94.18%** |

### 5.2. Báo cáo phân loại chi tiết của Mô hình Pipeline tối ưu trên tập Test (40 bệnh nhân)

```text
              precision    recall  f1-score   support

       drugA     0.8333    1.0000    0.9091         5
       drugB     1.0000    0.6667    0.8000         3
       drugC     1.0000    1.0000    1.0000         3
       drugX     1.0000    1.0000    1.0000        11
       drugY     1.0000    1.0000    1.0000        18

    accuracy                         0.9750        40
   macro avg     0.9667    0.9333    0.9418        40
weighted avg     0.9792    0.9750    0.9736        40
```

### 5.3. Ma trận nhầm lẫn (Confusion Matrix) trên tập Test

| Thực tế \ Dự đoán | drugA | drugB | drugC | drugX | drugY |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **drugA** | **5** | 0 | 0 | 0 | 0 |
| **drugB** | 1 | **2** | 0 | 0 | 0 |
| **drugC** | 0 | 0 | **3** | 0 | 0 |
| **drugX** | 0 | 0 | 0 | **11** | 0 |
| **drugY** | 0 | 0 | 0 | 0 | **18** |

### 5.4. Phân tích nguyên nhân sai sót (Error Analysis)
- Chỉ có đúng **1 ca duy nhất** bị dự đoán sai trên toàn bộ 40 bệnh nhân kiểm tra:
  - **Bệnh nhân dòng #138**: `{Age: 51, Sex: 'M', BP: 'HIGH', Cholesterol: 'NORMAL', Na_to_K: 11.343}`
  - Thực tế dùng `drugB`, mô hình dự đoán `drugA`.
  - **Nguyên nhân:** Cả `drugA` và `drugB` đều dùng cho bệnh nhân có `BP=HIGH` và `Na_to_K <= 15`. Ngưỡng phân tách giữa drugA và drugB là độ tuổi (`Age`). Bệnh nhân này 51 tuổi — nằm sát ranh giới phân tách độ tuổi mà mô hình học được từ tập huấn luyện.

---

## 6. Danh mục hình ảnh trực quan hóa (Generated Visualizations)

Các hình ảnh xuất ra tại `reports/figures/`:
1. `01_target_distribution.png`: Biểu đồ cột thể hiện phân phối 5 lớp nhãn (phát hiện mất cân bằng lớp).
2. `02_feature_relationships.png`: Tương quan giữa `Age` và `Na_to_K`, làm nổi bật ngưỡng $Na\_to\_K = 15$ tách biệt hoàn toàn nhóm `drugY`.
3. `03_confusion_matrices.png`: So sánh trực quan ma trận nhầm lẫn giữa Baseline Logistic Regression (95%) và Pipeline Random Forest (97.5%).
4. `04_decision_boundary_2d.png`: Ranh giới quyết định 2D so sánh đường biên tuyến tính của Logistic Regression và các nhát cắt vuông góc của Cây quyết định.

---

## 7. Đúc kết và Bài học kinh nghiệm (Key Takeaways)

1. **Hiểu dữ liệu trước khi chọn mô hình:** Bài toán Drug Classification vốn được tạo ra theo quy tắc logic dạng cây (`If-Else`). Do đó, các mô hình dạng cây (Decision Tree, Random Forest) giải quyết tốt hơn hẳn các mô hình tuyến tính (Logistic Regression).
2. **Đừng để Accuracy đánh lừa:** Một mô hình đạt Accuracy 95% có thể vẫn bỏ sót 33% số ca của lớp thiểu số (`drugB` Recall chỉ 66.7%). Luôn đối chiếu với Macro F1-Score và Confusion Matrix.
3. **Quy chuẩn Pipeline là bất di bất dịch:** Việc dùng `Pipeline` và `ColumnTransformer` bảo vệ dự án khỏi Data Leakage trong Cross-Validation và biến code thành một hệ thống sẵn sàng triển khai thực tế.

---

## 8. Hướng dẫn chạy từng Stage độc lập (Execution Guide)

```bash
# Stage 1: Khám phá dữ liệu
python src/stage1_understand_data.py

# Stage 2: Huấn luyện mô hình cơ sở
python src/stage2_baseline_model.py

# Stage 3: Đánh giá chuyên sâu và soi lỗi
python src/stage3_evaluation.py

# Stage 4: So sánh 3 mô hình & Cross-Validation
python src/stage4_model_comparison.py

# Stage 5: Đóng gói Pipeline chuẩn & GridSearchCV
python src/stage5_pipeline_workflow.py

# Stage 6: Xuất biểu đồ trực quan hóa
python src/stage6_visualization.py
```
