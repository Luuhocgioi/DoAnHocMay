import os
import numpy as np
import joblib  # Dùng để lưu file model
from sklearn import svm
from color_histogram_extractor import extract_color_histogram


def train_model(data_path, model_name="traffic_sign_model.pkl"):
    features_list = []
    labels_list = []

    print("--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN AI ---")

    # Duyệt qua các thư mục trong data/cam/
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            print(f"Đang học loại biển: {label}...")
            for img_file in os.listdir(label_path):
                if img_file.endswith(('.jpg', '.png')):
                    img_path = os.path.join(label_path, img_file)

                    # Trích xuất đặc trưng
                    feat = extract_color_histogram(img_path, bins=16)
                    if feat is not None:
                        features_list.append(feat)
                        labels_list.append(label)

    # Khởi tạo mô hình SVM (Support Vector Machine)
    # kernel='linear' giúp AI tìm đường thẳng phân chia các nhóm màu
    clf = svm.SVC(kernel='linear', probability=True)

    print("\nAI đang tính toán ranh giới toán học...")
    clf.fit(features_list, labels_list)

    # Lưu "bộ não" đã học xong vào file .pkl
    joblib.dump(clf, model_name)
    print(f"--- THÀNH CÔNG: Đã xuất file model '{model_name}' ---")


if __name__ == "__main__":
    train_model("data/cam/")