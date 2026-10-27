"""Stage 1: Understand the Data (Khám phá và Thấu hiểu Dữ liệu)"""

from pathlib import Path
import sys
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Tải dữ liệu từ đường dẫn file CSV."""
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {file_path}")
    return pd.read_csv(file_path)


def inspect_dataset(df: pd.DataFrame) -> None:
    """Kiểm tra cấu trúc, kiểu dữ liệu, missing values và trùng lặp."""
    print("=" * 60)
    print("1. TỔNG QUAN KÍCH THƯỚC & 5 DÒNG ĐẦU TIÊN")
    print("=" * 60)
    print(f"- Số lượng quan sát (hàng): {df.shape[0]}")
    print(f"- Số lượng thuộc tính (cột): {df.shape[1]}")
    print("\n5 dòng đầu tiên:")
    print(df.head())

    print("\n" + "=" * 60)
    print("2. THÔNG TIN KIỂU DỮ LIỆU & GIÁ TRỊ KHUYẾT THIẾU")
    print("=" * 60)
    info_df = pd.DataFrame({
        "Dtype": df.dtypes,
        "Non-Null Count": df.notnull().sum(),
        "Null Count": df.isnull().sum(),
        "Null Ratio (%)": (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(info_df)
    print(f"\n- Số dòng trùng lặp (duplicates): {df.duplicated().sum()}")

    print("\n" + "=" * 60)
    print("3. PHÂN LOẠI CÁC CỘT (FEATURE TYPES)")
    print("=" * 60)
    target_col = "Drug"
    feature_cols = [c for c in df.columns if c != target_col]
    num_features = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    cat_features = df[feature_cols].select_dtypes(include=["object", "string", "category"]).columns.tolist()

    print(f"- Cột nhãn mục tiêu (Target): ['{target_col}']")
    print(f"- Đặc trưng dạng số (Numerical features): {num_features}")
    print(f"- Đặc trưng dạng phân loại (Categorical features): {cat_features}")

    print("\n" + "=" * 60)
    print("4. THỐNG KÊ MÔ TẢ ĐẶC TRƯNG DẠNG SỐ (NUMERICAL)")
    print("=" * 60)
    print(df[num_features].describe().round(2))

    print("\n" + "=" * 60)
    print("5. GIÁ TRỊ ĐỘC BẢN CỦA ĐẶC TRƯNG PHÂN LOẠI (CATEGORICAL)")
    print("=" * 60)
    for col in cat_features:
        unique_vals = df[col].unique().tolist()
        print(f"Cột '{col}': {len(unique_vals)} giá trị khác nhau -> {unique_vals}")

    print("\n" + "=" * 60)
    print("6. PHÂN PHỐI NHÃN MỤC TIÊU (TARGET CLASS DISTRIBUTION)")
    print("=" * 60)
    target_counts = df[target_col].value_counts()
    target_ratios = (df[target_col].value_counts(normalize=True) * 100).round(2)
    target_dist = pd.DataFrame({
        "Số lượng (Count)": target_counts,
        "Tỷ lệ (%)": target_ratios
    })
    print(target_dist)


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "drug200.csv"

    print(f"Đang đọc dữ liệu từ: {data_path}")
    df = load_dataset(data_path)
    inspect_dataset(df)


if __name__ == "__main__":
    main()
