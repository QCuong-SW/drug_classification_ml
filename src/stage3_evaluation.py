"""Stage 3: Evaluate the Model (Đánh giá Đa chiều và Mổ xẻ Mô hình)

Mục tiêu học tập:
- Tính toán toàn diện: Accuracy, Precision, Recall, F1-Score.
- Phân biệt bản chất của Macro Average và Weighted Average trong phân loại đa lớp.
- Vẽ và phân tích Ma trận nhầm lẫn (Confusion Matrix).
- Truy tìm các mẫu bệnh nhân bị đoán sai (Error Analysis) và lý giải nguyên nhân.
- Giải thích vì sao Accuracy cao (95%) vẫn có thể là "cú lừa" nếu không nhìn vào Precision/Recall.
"""

# vaannx khai báo mục xử lý đường dẫn
from pathlib import Path
# can thiệp vào quá trinfhy hoạt đọng của hệ thống 
import sys
# khai báo numpy để dùng cho các tính toán số học với python
import numpy as np
# khai báo pandas xử lý dữ liệu dạng dataframe
import pandas as pd
# tiếp tục khai báo module chuyên dufnng để lựa chọn và chia dữ liệu cho ML model
# train_test_split là hàm cụ thể trong model_selection để chia dữ liệu thành các tập con
# dữ liệu train, nhãn train tương tự với test và % test, cố định cách chia, phân chia đồng bộ
from sklearn.model_selection import train_test_split
# preprocessing là dạng tiền xử lý dữ liệu để model dễ dàng tiếp cận hơn, thành các dạng mốc và dạng số bit để dễ xử lý
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# lựa chọn mô hình tuyến tính là logisticregression
from sklearn.linear_model import LogisticRegression
# là module chứa các công cụ đánh giá hiệu suất và đo lường ML model:
from sklearn.metrics import (
    # hàm tính độ chính xác tổng thể của model, nằm trong module sklearn.metrics
    accuracy_score,
    # tạo thành bảng thống kê số lượng dự đoán của model theo từng nhãn, cho biết đúng bao nhiêu và nhầm sang nhaxnn nào bao nhiêu
    confusion_matrix,
    # bảng tổng hợp báo cáo gồm model này dự đoánn đúng bao nhiêu, trong các mẫu thực tế nó tìm được bao nhiêu
    # 2 thawnfng recall và precision kết hợp thì đạt mức nào, support sẽ nêu ra có bao nhiêu mẫu A thực tế trong tập test 
        # precision_score,
        # recall_score,
        # f1_score
    # 3 thằng treenn được classification_report tổng hợp cùng lúc thành bảng chứa các thành phần đó
    classification_report,
    # 3thawnfg hàm tính toán precission_score và recall_store và f1_score
    precision_score, # tính xem % dự đoán kết quả mẫu đúng là bao nhiêu 
    recall_score,   # tìm được bao nhiêu % mẫu đúng trong mẫu thực tế
    f1_score        # cân bằng 2 thawnfng chỉ số có cao không , cả 2 có tốt cùng nhau hay không 
)

# Đảm bảo console Windows in ký tự tiếng Việt UTF-8 chuẩn xác
#  đảm bảo đúng mã định dạnh của hệ điều hành window 
if sys.platform == "win32":
    try:
    # chuyển đổi danngj ra thành danngj utf-8 giúp đầu ra tiếng việt chuẩn xác hơn 
        sys.stdout.reconfigure(encoding="utf-8")
    # nếu thuộc ngoại lệ và không chuyển đổi được, vẫn cho passs
    except Exception:
        pass

# cấu hình pandas hiển thị dataframe ra màn hình 
pd.set_option("display.max_columns", None)
# tạo giới hạn chiều rộng kí tự là 1000 kí tự cho bảng, độ rộng bảng
pd.set_option("display.width", 1000)


# hàm main 
def main():
    # lấy đường dẫn root (cha) linh hoạt 
    project_root = Path(__file__).resolve().parent.parent
    # gán folder cha vào đầu để linh hoạt hơn khi kiếm file dù folder name cha có bị đổi 
    data_path = project_root / "data" / "raw" / "drug200.csv"
    # sau đó cho biến dataframe đọc dữ liệu trong csv lưu lại để xử lý
    df = pd.read_csv(data_path)

    # lấy mọi cột khác, chỉ bỏ cột nhãn đi thôi
    X = df.drop(columns=["Drug"])
    # còn y thì là nhãn gốc, lấy nhãn Drug thôi
    y = df["Drug"]

    # Chia tập train/test cố định y hệt Stage 2
    # vẫn là chia tập train và test, với test chỉ chiếm 20% dữ liệu chính 
    # cũng như tách ra phần train và nhãn , tuonnwg tự với test
    # hệ số random cũnng dudojcw quy định để các truonnwfg hợp lặp lại được đồng nhất, đỡ đánh giá sai lệch do biến đổi của random 
    # stratify=y đảm bảo rằng có sự cân bằng giữa test và train thực dù k đảm bảo tuyệt đối
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # tạo 2 list là cột về số lượng và cột về đặc trưng,phục vụ việc chuyển đổi thành định dạng mà model 
    # dễ xử lý 
    num_cols = ["Age", "Na_to_K"]
    cat_cols = ["Sex", "BP", "Cholesterol"]
    # đối tuonnwgj scaler từ class StandardScaler 
    scaler = StandardScaler()
    # từ đạng cột số ở train, chuyeennr đổi thành dạng trục hệ số phù hợp cho bên model xử lý
    X_train_num = scaler.fit_transform(X_train[num_cols])
    # tương tự train, ở test cuxnng vậy, chuyển đổi thành dạng trục hệ số phù hợp 
    X_test_num = scaler.transform(X_test[num_cols])

    # tiếp tục chuyển các đặc truwnng dạng khác thành dạng bit 0/1
    # không lưu thafngh các matrix riêng biệt, mà lưu thành mảng dể tiết kiệm bộ nhớ, 
    # handle_unknown="ignore" tức là các dữ liệu rõ ràng vẫn sẽ được chuyển thành dạng bit để xử lý luôn
    # tức là dữ liệu dù chưa xuất hiện ở đã train nhưng vẫn được thêm vào xử lý nếu mới có 
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    # chuyển thành định dạng mà model dễ đọc honwn đối với dữ liệu trừ dạng số 
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    # tiếp tục chuyển với phần test, do đã học cách chuyển đổi phù hợp với bên trainn nên có thể dùng lại cho test
    # teswst chỉ cần chueyenr đổi
    X_test_cat = encoder.transform(X_test[cat_cols])

    # ghép 2 tập đã chueyenr đổi lại thành dạng cho model dễ xử lý rồi 
    # làm train num với train cat, tuionwg ứng với testnum và testcat
    X_train_processed = np.hstack([X_train_num, X_train_cat])
    X_test_processed = np.hstack([X_test_num, X_test_cat])

    # Huấn luyện mô hình cơ sở
    # model tuyến tinnsh được lựa chọn đó là LogisticRegression 
    model = LogisticRegression(max_iter=500, random_state=42)
    # và cho model học từ đặc trưng và nhãn train chính thức
    # nên nhớ các đặc trưng cần chuyển đổi sang dạng mà model dễ xử lý
    model.fit(X_train_processed, y_train)

    # dùng model đã học để đoán ra nhãn từ phần dữ liệu test chưa được nó trainn nhưng thuộc chung tập dữ liệu 
    y_pred = model.predict(X_test_processed)
    # tính ra số % dự đoán cho mỗi nhãn dự đoán của mỗi đối tượng dữ liệu
    # vị dụ bên A có đặc trưng x1 x2 x3 thì sẽ đưa ra dự đoán nhãn kiểu nhãn 1 40% , nhãn 2 10% ,....tương tự vậy để trực quan hơn 
    y_proba = model.predict_proba(X_test_processed)

    print("=" * 65)
    print("1. CÁC ĐỘ ĐO TỔNG QUAN (OVERALL METRICS)")
    print("=" * 65)
    # đầu tienen là lưu ra accuracy_score : là % chính xác chung của model trên toàn bộ các nhãn ở test, dựa vào gốc và test thực tế
    acc = accuracy_score(y_test, y_pred)
    # precision_score là tỉ lệ doán trúng của từng nhãn trên số lần đoán toonnrg thể
    # ví dụ đoánn 5 lần mà chỉ có 7 lần trúng A , lấy trunng bình số % của 3 bên đoán, tính được tỉ lệ chính xác của tuwnfng nhãn
    prec_macro = precision_score(y_test, y_pred, average="macro")
    # tiếp theo là điểm recall average = "macro" tức cũng lấy trung bình cộng tỉ lệ chính xác
    rec_macro = recall_score(y_test, y_pred, average="macro")
    # về macro cũng được xét đều 
    f1_macro = f1_score(y_test, y_pred, average="macro")
    
    prec_weighted = precision_score(y_test, y_pred, average="weighted")
    rec_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # sau đó in ra các chỉ số:
    print(f"- Accuracy (Độ chính xác chung):        {acc * 100:.2f}%")
    print(f"- Macro Precision:                    {prec_macro * 100:.2f}%")
    print(f"- Macro Recall:                       {rec_macro * 100:.2f}%")
    print(f"- Macro F1-Score:                     {f1_macro * 100:.2f}%")
    print(f"- Weighted F1-Score:                  {f1_weighted * 100:.2f}%")


    # mục tiếp theo là báo cáo phân loại chi tiết theo tuwnfdg lớp
    print("\n" + "=" * 65)
    print("2. BÁO CÁO PHÂN LOẠI CHI TIẾT THEO TỪNG LỚP (CLASSIFICATION REPORT)")
    print("=" * 65)
    # dựa vào nhãn test và y đã chuyển đổi 
    # tổng hợp các chỉ số đánh giá trên thành 1 bảng duy nhất
    print(classification_report(y_test, y_pred, digits=4))

    # 
    print("=" * 65)
    print("3. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)")
    print("=" * 65)
    # tất cả các nhãn mà model đã học được để đự đoán trong quá trình train
    classes = model.classes_
    # cm = tức là ma trận sẽ được lưu theo dạng mảng số
    # với y_test là các nhãn thực tế, đầu vào tiếp theo là y_pred tức là chỉ số dự đoán đối với các nhãn, cũng như 
    # labels=classes tức là thứ tự các nhãn phải tuân theo danh sách tuownng ứng với classes 
    # lấy các nhá
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    # sau đó in ra dataframe được chuyển theo dạng 
    # 
    cm_df = pd.DataFrame(
        cm,
        index=[f"Thực tế: {c}" for c in classes], # in ra ở dạng thực tế với mỗi nhãn là số lần dự đoán đúng theo đường chéo
        # các số lần k nằm trong đường chéo là leehcj, hoặc có ggias trị 0 là không đoán vào lần vào tại nhãn đó 
        columns=[f"Dự đoán: {c}" for c in classes]
    )
    # in ra bảng đó. chỉ ra số lần model chọn đúng và số lần model chọn sai
    print(cm_df)

    # tiếp tục là truy vết các mẫu bị dự đoán sai 
    print("\n" + "=" * 65)
    print("4. TRUY VẾT CÁC MẪU BỊ ĐOÁN SAI (ERROR ANALYSIS)")
    print("=" * 65)
    
    # donfngt nnafy là tổng hợp các giá trị, ở đâu có giá trị khác với mục dự đoán 
    # tính số lần thực tế không khớp với dự đoán ở nhãn kết quả 
    # tổng số lần không khớp
    misclassified_indices = np.where(y_test.values != y_pred)[0]
    # in ra toonrgh số lần không khớp treenn/ toàn bộ nhãn dự đoán 
    print(f"Tổng số ca đoán sai trong tập test: {len(misclassified_indices)} / {len(y_test)}")

    # duyệt vòng lặp sau đó hiển thị các ca sai khi dự đoán 
    for idx in misclassified_indices:
        # lấy ra index của bệnh nhanan 
        orig_idx = y_test.index[idx]
        # thuốc thực tế 
        actual_drug = y_test.iloc[idx]
        # mô hình dự đoán, đưa ra index đún mà mô hình dự đoán
        pred_drug = y_pred[idx]
        # thgông tin bệnh nhân chi tiết, X là từ đặc trưng
        patient_info = X_test.iloc[idx].to_dict()
        # tên nhãn kèm số % dự đoán của model với các nhãn (phân bổ xác suất)
        probas = {cls: f"{p * 100:.1f}%" for cls, p in zip(classes, y_proba[idx])}

        print(f"\n[Bệnh nhân dòng #{orig_idx} trong dataset gốc]:")
        print(f"  - Thông số lâm sàng: {patient_info}")
        print(f"  - Thuốc thực tế:    {actual_drug}")
        print(f"  - Mô hình đoán:     {pred_drug}")
        print(f"  - Phân bổ xác suất: {probas}")

    print("\n" + "=" * 65)
    print("5. BÀI HỌC VỀ MACRO VS WEIGHTED VÀ ACCURACY")
    print("=" * 65)
    print("• Weighted Average bị chi phối mạnh bởi lớp lớn (drugY có 18 mẫu trong test).")
    print("• Macro Average coi 5 lớp thuốc quan trọng BÌNH ĐẲNG như nhau.")
    print("• Nếu mô hình đoán sai hoàn toàn lớp thiểu số (như drugB chỉ có 3 mẫu),")
    print("  Accuracy vẫn có thể đạt ~92% nhưng Macro Recall sẽ giảm mạnh!")


if __name__ == "__main__":
    main()
