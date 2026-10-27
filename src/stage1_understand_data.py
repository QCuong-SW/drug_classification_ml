"""Stage 1: Understand the Data (Khám phá và Thấu hiểu Dữ liệu)

Mục tiêu học tập:
- Đọc và kiểm tra cấu trúc dữ liệu bằng Pandas (shape, head, info).
- Phát hiện missing values (dữ liệu khuyết thiếu) và duplicate rows (dữ liệu trùng lặp).
- Phân biệt đặc trưng số (Numerical features) và đặc trưng phân loại (Categorical features).
- Phân tích phân phối của nhãn mục tiêu (Target class distribution) để phát hiện mất cân bằng lớp.
- Phát biểu bài toán Machine Learning có giám sát (Supervised Multiclass Classification).
"""

# from pathlib import Path : được dùng để xử lý file và thư mục
from pathlib import Path
# import sys : dùng để tương tác với môi truownfng chạy chuonwng trình 
import sys
# import pandas để xử lý dữ liệu trước khi đưa vào sử dụng
import pandas as pd

# Đảm bảo console Windows in ký tự tiếng Việt UTF-8 chuẩn xác
# đảm bảo chuyển đổi thành dạng in ký tự  UTF-8 
# trường hợp không cần chuyển đổi hoặc chuyển đổi không được vẫn cho pass
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# trước hết 
def load_dataset(file_path: Path) -> pd.DataFrame:
    """Tải dữ liệu từ đường dẫn file CSV."""

    # nếu không tồn tại giá trị được lưu tại path 
    if not file_path.exists():
            # in ra console không tìm thấy file dữ liệu tại "đường dẫn"
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {file_path}")
    # còn trường hợp có, thì đọc file csv từ đường dẫn đó rồi lưu vào df
    df = pd.read_csv(file_path)
    # return df
    return df

# kiểm duyệt tập dữ liệu với đầu vào là file với dạng dataframe của panndas define
def inspect_dataset(df: pd.DataFrame) -> None:
    """Kiểm tra tổng quan cấu trúc, kiểu dữ liệu, missing values và trùng lặp."""
    print("=" * 60)
    print("1. TỔNG QUAN KÍCH THƯỚC & 5 DÒNG ĐẦU TIÊN")
    print("=" * 60)
    # số lượng hàng và cột ( hàng: 0, cột: 1)
    print(f"- Số lượng quan sát (hàng): {df.shape[0]}")
    print(f"- Số lượng thuộc tính (cột): {df.shape[1]}")
    print("\n5 dòng đầu tiên:")
    # dùng head khoonng có tham số, mặc định là in 5 dòng đầu tiên 
    print(df.head())

    print("\n" + "=" * 60)
    print("2. THÔNG TIN KIỂU DỮ LIỆU & GIÁ TRỊ KHUYẾT THIẾU")
    print("=" * 60)

    # thống kê các giá trị khuyết thiếu 
    info_df = pd.DataFrame({
        # đưa ra kiểu dữ liệu của toàn bộ df
        "Dtype": df.dtypes,
        # tạo cột mới để lưu dữ liệu được tish toán, thống kê vào đó 
        # không rỗng, tổng hết toàn bộ, tùy theo giá trị cần truy xuất
        # ở đây đòi lấy nnull thế nên sẽ thống kê theo cấp từng giá trị có thỏa không
        "Non-Null Count": df.notnull().sum(),
        # tương tự, các giá trị ko rỗng cũng được tổng 
        "Null Count": df.isnull().sum(),
        # sau đó tính % , round(2) làm tronnf đến 2 chũ số thập phân
        "Null Ratio (%)": (df.isnull().sum() / len(df) * 100).round(2)
    })
    # in ra thông tin của dataframe mà ta làm việc 
    print(info_df)

    # lấy ra số lượng dòng bị trùng lặp trong df rồi tổng lại
    num_duplicates = df.duplicated().sum()
    print(f"\n- Số dòng trùng lặp (duplicates): {num_duplicates}")

    print("\n" + "=" * 60)
    print("3. PHÂN LOẠI CÁC CỘT (FEATURE TYPES)")
    print("=" * 60)
    # sau đó sẽ phân loại các cột, tạo thành danh sách các cột trừ cột thuốc 
    target_col = "Drug"
    feature_cols = [c for c in df.columns if c != target_col]

    # cột dạng số: từ cột tính nnawng đã lọc , tiếp tục chọn cột có type được include là dạng number lấy các cột đó thành một danh sách
    num_features = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    # tương tự ở đây lấy các cột gồm các kiểu dữ liệu như obj, strinng, category  và cũng tạo thành list các cột có kiểu dữ liệu 
    # tương ứng 
    cat_features = df[feature_cols].select_dtypes(include=["object", "string", "category"]).columns.tolist()

    # in ra cột mục tiêu đó là cột nào ,đặc trưng dạng số và đặc trưng dạng phân loại

    print(f"- Cột nhãn mục tiêu (Target): ['{target_col}']")
    print(f"- Đặc trưng dạng số (Numerical features): {num_features}")
    print(f"- Đặc trưng dạng phân loại (Categorical features): {cat_features}")

    print("\n" + "=" * 60)
    print("4. THỐNG KÊ MÔ TẢ ĐẶC TRƯNG DẠNG SỐ (NUMERICAL)")
    print("=" * 60)
    # in ra cột mô tả làm tròn
    # in sô luonngjw, % thống kêm,... rất nhiều chỉ số hay
    print(df[num_features].describe().round(2))

    print("\n" + "=" * 60)
    print("5. GIÁ TRỊ ĐỘC BẢN CỦA ĐẶC TRƯNG PHÂN LOẠI (CATEGORICAL)")
    print("=" * 60)
    # duyệt các dòng trong các cột phân loại
    # lấy giá trị duy nhất trong từng cột thuộc category dataframe 
    for col in cat_features:
        # với mỗi cột, tạo thành list các giá trị duy nhất trong cột đó, ví dụ cột giới tính thì có male và female 
        unique_vals = df[col].unique().tolist()
        #  in ra cột nào, và số lượng giá trị độc nhất và in ra các giá trị khác nhau đó. 
        print(f"Cột '{col}': {len(unique_vals)} giá trị khác nhau -> {unique_vals}")

    print("\n" + "=" * 60)
    print("6. PHÂN PHỐI NHÃN MỤC TIÊU (TARGET CLASS DISTRIBUTION)")
    print("=" * 60)
    # đầu tiên: đếm số lần xuất hiện của từng giá trị khác nhau trong 1 cột 
    target_counts = df[target_col].value_counts()
    # tiếp tục là tinsnh tỉ lệ, cũng với cột mục tiêu nhưng thêm tính tỉ trọng xuát hiện theo dạng % cho các dữ liệu
    # ví dụ 1 loại xuất hiện nhiều thì tỉ trọng cao, còn xuất hiện ít thì tỉ trọng ít, đơn giản thế thôi. 
    # do value count vẫn tồn tại nnene nó vẫn đếm số lần xuất hiện của các giá trị 
    target_ratios = (df[target_col].value_counts(normalize=True) * 100).round(2)

    # bên dưới thì số lượng vẫn được in ra: 
    # tạo thành một bảng mới với tên cột mới được định nghĩa, kèm theo gán giá trị cho cột 
    target_dist = pd.DataFrame({
        "Số lượng (Count)": target_counts,
        "Tỷ lệ (%)": target_ratios
    })
    # in ra phân phối nhãn thống kê mục tiêu 
    print(target_dist)


def main():
    # Đường dẫn file dữ liệu tương đối so với thư mục gốc dự án
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "drug200.csv"

    print(f"Đang đọc dữ liệu từ: {data_path}")
    df = load_dataset(data_path)
    inspect_dataset(df)


if __name__ == "__main__":
    main()
