import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from color_histogram_extractor import extract_color_histogram


def evaluate_model(test_data_path, model_path="traffic_sign_model.pkl"):
    print("--- BẮT ĐẦU CHẤM ĐIỂM MÔ HÌNH ---")

    try:
        clf = joblib.load(model_path)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy model. Hãy train trước!")
        return

    y_true = []  # Chứa nhãn thực tế
    y_pred = []  # Chứa nhãn AI dự đoán

    # Duyệt qua thư mục test
    for label in os.listdir(test_data_path):
        label_path = os.path.join(test_data_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                if img_file.endswith(('.jpg', '.png')):
                    img_path = os.path.join(label_path, img_file)

                    # Trích xuất đặc trưng
                    feat = extract_color_histogram(img_path, bins=16)
                    if feat is not None:
                        # Lưu lại đáp án đúng
                        y_true.append(label)
                        # AI dự đoán và lưu lại
                        prediction = clf.predict([feat])[0]
                        y_pred.append(prediction)

    # Nếu có dữ liệu, bắt đầu tính toán
    if len(y_true) > 0:
        accuracy = accuracy_score(y_true, y_pred)
        print("\n" + "=" * 50)
        print(f"KẾT QUẢ ĐÁNH GIÁ TRÊN {len(y_true)} BỨC ẢNH TEST")
        print("=" * 50)
        print(f"ĐỘ CHÍNH XÁC TỔNG THỂ (Accuracy): {accuracy * 100:.2f}%\n")

        print("BÁO CÁO CHI TIẾT (Classification Report):")
        print(classification_report(y_true, y_pred))

        print("MA TRẬN NHẦM LẪN (Confusion Matrix):")
        print(confusion_matrix(y_true, y_pred))
    else:
        print("Không tìm thấy ảnh test hợp lệ!")


if __name__ == "__main__":
    # ĐIỀN ĐƯỜNG DẪN THƯ MỤC TEST CỦA BẠN VÀO ĐÂY
    # Ví dụ: r"D:\Downloads\traffic signs.v7i.folder\valid"
    test_directory = "data/cam_test/"
    evaluate_model(test_directory)