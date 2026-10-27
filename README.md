# Drug Classification ML Lab

Dự án Machine Learning phân loại đa lớp (Multiclass Classification) có giám sát sử dụng thư viện **Scikit-learn** trên bộ dữ liệu lâm sàng `drug200.csv`.

---

## 1. Tổng quan Dữ liệu (`data/raw/drug200.csv`)

- **Quy mô:** 201 bản ghi, 6 thuộc tính (0 missing values, 1 dòng duplicate).
- **Đặc trưng đầu vào (Features):**
  - Số học (`Numerical`): `Age` (Tuổi: 15–74), `Na_to_K` (Tỷ lệ Natri/Kali: 6.27–38.25).
  - Phân loại (`Categorical`): `Sex` (`F`, `M`), `BP` (`HIGH`, `NORMAL`, `LOW`), `Cholesterol` (`HIGH`, `NORMAL`).
- **Nhãn mục tiêu (`Target`):** 5 loại thuốc — `drugY` (45.8%), `drugX` (26.9%), `drugA` (11.4%), `drugB` (8.0%), `drugC` (8.0%).
- **Lưu ý:** Dữ liệu mất cân bằng lớp (`Imbalanced`), cần sử dụng `stratify=y` và đánh giá qua `Macro F1-Score` thay vì chỉ dùng `Accuracy`.

---

## 2. Cấu trúc Dự án

```text
drug_classification_ml/
├── data/raw/drug200.csv          # Dữ liệu gốc
├── src/                          # 6 giai đoạn học máy độc lập
│   ├── stage1_understand_data.py # Khám phá EDA, kiểu dữ liệu, phân phối nhãn
│   ├── stage2_baseline_model.py  # Stratified split, StandardScaler + OneHot, Logistic Regression
│   ├── stage3_evaluation.py      # Confusion Matrix, Macro/Weighted F1, Error Analysis
│   ├── stage4_model_comparison.py# Benchmark 3 mô hình (LR vs Decision Tree vs Random Forest)
│   ├── stage5_pipeline_workflow.py# ColumnTransformer + Pipeline + GridSearchCV (Anti-leakage)
│   └── stage6_visualization.py   # Xuất biểu đồ phân tích và ranh giới quyết định 2D
├── reports/figures/              # 4 biểu đồ trực quan hóa
├── requirements.txt              # Thư viện phụ thuộc
└── README.md
```

---

## 3. Kết quả Thực nghiệm Thực tế

| Mô hình (Model) | Train Acc | 5-Fold CV Acc (Train) | 5-Fold CV F1-Macro | Test Acc | Test F1-Macro |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 98.75% | 94.38 ± 3.06% | 92.35 ± 4.31% | 95.12% | 92.77% |
| **Decision Tree** | 100.00% | 99.38 ± 1.25% | 99.60 ± 0.81% | 97.56% | 94.18% |
| **Random Forest + Pipeline (Tối ưu)** | 100.00% | 98.75 ± 2.50% | 98.35 ± 3.30% | **97.56%** | **94.18%** |

### Đúc kết chuyên môn:
1. **Bản chất bài toán dạng luật (Rule-based):** Ngưỡng $Na\_to\_K > 15$ tách biệt hoàn toàn nhóm `drugY`. Các mô hình dạng cây học các nhát cắt vuông góc (orthogonal splits) tối ưu hơn ranh giới phẳng của Logistic Regression.
2. **Không phụ thuộc vào Accuracy:** Với lớp thiểu số như `drugB`, mô hình tuyến tính có thể đoán nhầm sang `drugA`. Macro F1 phản ánh chính xác hiệu năng đồng đều giữa các lớp.
3. **Pipeline chống Data Leakage:** Đóng gói `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) cùng `RandomForestClassifier` qua `GridSearchCV` đảm bảo chuẩn hóa độc lập trong từng Fold.

---

## 4. Hướng dẫn Chạy

### Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Chạy từng Stage Python
```bash
python src/stage1_understand_data.py
python src/stage2_baseline_model.py
python src/stage3_evaluation.py
python src/stage4_model_comparison.py
python src/stage5_pipeline_workflow.py
python src/stage6_visualization.py
```

### Chạy Giao diện Web Trực quan (Tùy chọn)
```bash
python server.py
# Hoặc nhấp đúp file run_web.bat -> Mở http://127.0.0.1:8000
```
