"""Stage 2: Build a Baseline (Xây dựng Mô hình Cơ sở)

Mục tiêu học tập:
- Tách ma trận đặc trưng (X) và vector nhãn mục tiêu (y).
- Chia tập dữ liệu Train / Test có sử dụng `stratify=y` và `random_state`.
- Tiền xử lý dữ liệu chuẩn khoa học: Encode biến phân loại và Scale biến số.
- Huấn luyện mô hình cơ sở Logistic Regression.
- Hiểu và phân biệt rõ 3 phương thức then chốt:
  + fit(): Học trọng số từ tập train.
  + predict(): Dự đoán nhãn lớp cụ thể.
  + predict_proba(): Dự đoán phân phối xác suất (%) giữa các lớp.
"""

# import mục xử lý đường dẫn trong python
from pathlib import Path
# để can thiệp vào hệ thoonnsg đang hoạt động 
import sys
# khai báo thư viện xử lý số học của py
import numpy as np
# lib pandas được khai báo để xử lý dữ liệu dưới dạng dataframe 
import pandas as pd

# bắt đầu import scikit learn để dưa vào tạo ra model thực tế 
# sklearn là thư viện machine learning, model_selection chính là module chứa công cụ hỗ trợ lựa chọn và đánh giá mô hình 
# còn train_test_split là hàm chia dữ liệu 
from sklearn.model_selection import train_test_split
# import 2 công cụ: preprocessing: tiền xử lý dữ liệu
# với OneHotEncoder nó sẽ mã hóa dữ liệu phân loại 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# sklearn.linear_model là module trong scikitlearn chứa các thuật toán mô hình tuyến tính 
from sklearn.linear_model import LogisticRegression

# Đảm bảo console Windows in ký tự tiếng Việt UTF-8 chuẩn xác
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    # đưa vào file gốc, lấy file thư mcuj gốc bằng cách đi lên 2 cấp 
    project_root = Path(__file__).resolve().parent.parent
    # lấy đường dẫn đến file data linh động theo file cha , rồi sau đó lấy đến file csv để có được dữ liệu 
    data_path = project_root / "data" / "raw" / "drug200.csv"
    # tạo một biến là một dataframe lưu dữ liệu đọc được từ file csv 
    df = pd.read_csv(data_path)

    print("=" * 65)
    print("BƯỚC 1: TÁCH ĐẶC TRƯNG (X) VÀ NHÃN MỤC TIÊU (y)")
    print("=" * 65)
    # phiên bản dataframe x copy sẽ bị bỏ phần drug đi 
    X = df.drop(columns=["Drug"])
    # với y thì chỉ giữ riêng cột drug ở bảng gốc 
    y = df["Drug"]

        # tính số dòng và số cột của X và Y, và số dòng bănnfg nhau, còn số cột thfi bên X nhiều hơn
    print(f"Kích thước ma trận X (Features): {X.shape} (200 mẫu, 5 đặc trưng)")
    print(f"Kích thước vector y (Target):    {y.shape} (200 nhãn tương ứng)")

    print("\n" + "=" * 65)

    # chia tập dữ liệu để train
    print("BƯỚC 2: CHIA TẬP DỮ LIỆU TRAIN / TEST (80% / 20%)")
    print("=" * 65)
    # stratify=y đảm bảo tỷ lệ 5 loại thuốc ở tập train và test đều giống tập gốc
    # random_state=42 giúp kết quả chia luôn cố định mỗi lần chạy (reproducibility)

    
    # train test split này có nhiệm vụ tách bài toán thành 2 phần 1 phần để train và 1 phần để test dựa trên model đã trainn
    X_train, X_test, y_train, y_test = train_test_split(
        # đưa vào 1 phần để train, và tìm ra 3 phần còn lại
        X, y,
        # chấp nhận phạm vi test chỉ 20% của dữ liệu chính 
        test_size=0.2,
        # random state quy định phép chia cố định, ví dụ như các tập dữ liệu khác nhau thì ko nói, nhưng các tập giống
        # thì vẫn cần cố định để từ đó tìm ra được điểm sai lệch, bất ổn từ tập dữ liệu cũ hoặc tuonngw đối giống 
        random_state=42,
        # giữ được tỉ lệ các loại thuốc gần giống với tỉ lệ gốc nhất có thể để dễ dàng so sánh và thực thi
        stratify=y
    )

    # học 80% đó và tìm ra quyy luật cho 20% còn lại
    # in ra số donfgh của train và của test(20%)
    print(f"- Tập Train (Học): {X_train.shape[0]} mẫu (80%)")
    print(f"- Tập Test (Thi):  {X_test.shape[0]} mẫu (20%)")

    # So sánh tỷ lệ phân phối nhãn để chứng minh vai trò của stratify=y
    
    # đếm số lượng train rồi chuyển thành dạng tỉ lệ từ 0-1 rồi sau đó nhân với 100 để ra số %
    # mục đích 2 thằng này là lấy ra tỉ lệ % của train và test 
    train_dist = (y_train.value_counts(normalize=True) * 100).round(1)
    test_dist = (y_test.value_counts(normalize=True) * 100).round(1)

        #lưu lại dataframe chứa tỉ lệ phần trăm của train và của test
        # tỉ lệ phần trăm này là của tuwfg giá trị rienneg nhau, ví dụ giá trị này xuất hiện với tần suất bao nhiêu % trong df
    dist_cmp = pd.DataFrame({"Train (%)": train_dist, "Test (%)": test_dist})
    print("\nKiểm tra tỷ lệ phân phối nhãn sau khi stratify:")
    # hiển thị nname col và cả value đã được tính toán 
    print(dist_cmp)


        # bước tiền xử lý dữ liệu để model có thể sử dụng
    print("\n" + "=" * 65)
    print("BƯỚC 3: TIỀN XỬ LÝ (PREPROCESSING) CHỐNG DATA LEAKAGE")
    print("=" * 65)

    # trước hết là gom lsst các tên coojjt lại(các cột kiểu số, các cột kiểu đặc trưng)
    num_cols = ["Age", "Na_to_K"]
    cat_cols = ["Sex", "BP", "Cholesterol"]

    # 1. Scale đặc trưng số (StandardScaler: đưa về trung bình = 0, độ lệch chuẩn = 1)
    # đưa về dạng dễ xử lý 
    scaler = StandardScaler()
    # chủa hóa về dạng trục số để dễ dàng xử lý honnw cho model
    # NGUYÊN TẮC VÀNG: Chỉ fit trên X_train!
    # học thoonng số từ train và chuyển đổi thành trục dễ xử lý bằng fit_transform
    X_train_num = scaler.fit_transform(X_train[num_cols])
    # còn thằng này thì chỉ cần chuyển đổi trục số là được, còn việc học thoonng sốp thì đã học từ phần traiun trước đó
    X_test_num = scaler.transform(X_test[num_cols])

    # 2. Encode đặc trưng phân loại (OneHotEncoder: biến chuỗi thành các cột nhị phân 0/1)
    # chuyển dạng dữ liệu thành các danngj cột nhị phân 0 và 1 để model dễ dafnng xử lý hơn
    # bảo encoder trả kết quả thành mảng số thoonng thường thay vì dùng ma trận thưa, để tiết kiệm bộ nhớ hơn
    # và handle_unknown="ignore" tức là đảm bảo enncoder sẽ khoonng bị lỗi khi nhận vô giá trị lạ mới, mà chỉ chuyển nó thành dạng
    # số tương ứng 
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # tương tự với train và test của các cột phân loại, học thông số và chuyển đổi thành trục 
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    # Gộp đặc trưng số và phân loại lại thành ma trận đầu vào hoàn chỉnh
    # sau khi đã chuyển về dạng dễ xử lý, sau đó sẽ gộp lại thành đủ phần  train và phần test ở cả phần số và phần đặc truwnng của chúng
    X_train_processed = np.hstack([X_train_num, X_train_cat])
    # hstack dùng để ghép 2 phần đặc trưng với nhau 
    X_test_processed = np.hstack([X_test_num, X_test_cat])

    # tạo danh sách tên cho các cột sau biến đổi 
    # rồi ghép tên các cột tipes tục lại với nhau 
    encoded_feature_names = num_cols + encoder.get_feature_names_out(cat_cols).tolist()
    # đưa ra số lượng đặc trưng đã ghép sau khi mã hóa
    print(f"Số lượng đặc trưng sau khi mã hóa: {len(encoded_feature_names)}")
    # và đưa ra danh sách đặc trưng theo cột và luonngwj đặc trunwng, trừ các cột số thì giữ nnguyeenn name
    print(f"Danh sách đặc trưng: {encoded_feature_names}")


        # bước này là thời điểm cho model học thật sự
        # khởi tạo và huấn luyện model theo tt tuyeenns tính logistic regression
    print("\n" + "=" * 65)
    print("BƯỚC 4: KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH (LOGISTIC REGRESSION)")
    print("=" * 65)
    # max_iter=500 để thuật toán tối ưu có đủ số vòng lặp hội tụ

    # model cần tối ưu hệ số để tìm ra được cách phân loại phù hợp và max_inter = 500 cho phép quá trinnfh tối ưu chạy
    # tối đa 500 vòng lặp để tìm nnghieemj tối ưu tốt hơn, chưa chắc chawnsn khiến model tốt hơn nhưng hội tụ nhiều lần honnw thì 
    # có khả năng ra nghiệm tối ưu tốt
    # random_state là cố dịnh seed ngẫu nhiên để kết quả được tái lập kêt quả khi thuật toán dùng yếu tố ngẫu nhiên
    model = LogisticRegression(max_iter=500, random_state=42)


    # fit(): Mô hình học mối quan hệ giữa X_train_processed và y_train
    # method fit() này dufnng để model học từ X là gồm các đặc trưng, còn y là đáp án để so sasnnh và học cách đưa ra đáp án từ data
    model.fit(X_train_processed, y_train)
    # hiển thị mô hinnfh đã huấn luyện thành coonng với hàm fit
    print("✓ Mô hình đã huấn luyện xong thành công (fit completed)!")
    # in ra các lớp nhãn mà mô hinnfh đã học được , ví dụ drugA, B ,,..... 
    # các đầu ra mà nó có thể xác định từ đặc trưng được đưa vào
    print(f"Các lớp nhãn mà mô hình đã học được: {model.classes_}")

        # đây là bước đưa mô hình test trên một phần tập dữ liệu thực tế sau khi đã train
        # để đánh giá xem có thực sự hiệu quả ra sao
    print("\n" + "=" * 65)
    print("BƯỚC 5: DỰ ĐOÁN VÀ PHÂN BIỆT predict() VS predict_proba()")
    print("=" * 65)
    # Dự đoán nhãn cuối cùng
    # fit là trước đó ta dùng để khởi tạo model và train dựa trên X và y rồi, còn ta dùng predict để đưa vào các đặc trunngw cần
    # phán đoán dựa trên model đã train
    y_pred = model.predict(X_test_processed)

    # Dự đoán xác suất chi tiết cho từng lớp thuốc (%)
    # dựa vào đặc trưng đã thêm vào để dự đoán, nó sẽ đưa ra tỉ lệ mà các đặc truwnng được xác định vào nhãn nào 
    # xét theo mỗi bệnh nhân, xem với các đặc trunwgh của họ thì tỉ lệ rơi vô các nhãn thuốc để uoonnsg ra sao
    y_proba = model.predict_proba(X_test_processed)

    # Hiển thị 5 mẫu đầu tiên của tập test để so sánh
    # inn ra soi chi tiết 5 bệnh nhân đầu tiên tronng tập test 
    print("Soi chi tiết 5 bệnh nhân đầu tiên trong tập Test:")
    # in ra 5 bệnh nhanna đầu tienen trong tập testing 
    for i in range(40):
        # lấy phần nnhaxn thực tế của bệnh nhân đó 
        actual = y_test.iloc[i]
        # lấy 1 nhaxnn mà model dự đoán 
        predicted = y_pred[i]
        # cls sẽ lưu 
        probas = {cls: f"{prob * 100:.1f}%" for cls, prob in zip(model.classes_, y_proba[i])}
        # không khớp với nhãn thực tế thì suy ra sai 
        is_correct = "✓ ĐÚNG" if actual == predicted else "✗ SAI"
        # in ra bệnh nhân nào, thực tế là nhãn nào, va dựa đoánn nhãn nào, kết quả đúng hay sai, và xác suất chi tiết là bao nhiêu ở các
        # nhãn khi model dự đoán cho từng bệnh nhân
        print(f"\nBệnh nhân #{i+1}:")
        print(f"  - Thực tế:  {actual}")
        print(f"  - Dự đoán:  {predicted} ({is_correct})")
        print(f"  - Xác suất chi tiết: {probas}")

    # 
    baseline_acc = (y_pred == y_test).mean() * 100
    print("\n" + "-" * 65)
    print(f"Độ chính xác sơ bộ (Baseline Accuracy) trên tập Test: {baseline_acc:.2f}%")
    print("-" * 65)


if __name__ == "__main__":
    main()
