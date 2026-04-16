import joblib
from color_histogram_extractor import extract_color_histogram


def predict_new_image(image_path, model_path="traffic_sign_model.pkl"):
    # 1. Nạp "bộ não" đã học
    try:
        clf = joblib.load(model_path)
    except:
        print("Lỗi: Chưa tìm thấy file model. Hãy chạy train_svm.py trước!")
        return

    # 2. Trích xuất đặc trưng ảnh mới
    feat = extract_color_histogram(image_path, bins=16)

    if feat is not None:
        # 3. AI đưa ra dự đoán
        prediction = clf.predict([feat])[0]
        # Tính toán xác suất (độ tự tin)
        prob = clf.predict_proba([feat])
        confidence = max(prob[0]) * 100

        print("\n" + "=" * 40)
        print("KẾT QUẢ TỪ MÔ HÌNH SVM")
        print("=" * 40)
        print(f"DỰ ĐOÁN: >> {prediction.upper()} <<")
        print(f"ĐỘ TỰ TIN: {confidence:.2f}%")
        print("=" * 40)


if __name__ == "__main__":
    predict_new_image("data/cam/nguoc_chieu/test.png")