import cv2
import numpy as np
import os
from color_histogram_extractor import extract_color_histogram


def load_database(data_folder):
    """Quét thư mục data để lấy tất cả đặc trưng mẫu."""
    database = {}
    print("--- ĐANG KHỞI TẠO CƠ SỞ DỮ LIỆU MẪU ---")

    # Duyệt qua từng thư mục con (ví dụ: nguoc_chieu, cam_dung)
    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            # Lấy ảnh đầu tiên trong thư mục đó làm mẫu chuẩn
            for file_name in os.listdir(folder_path):
                if file_name.endswith(('.jpg', '.png')):
                    img_path = os.path.join(folder_path, file_name)
                    # Trích xuất đặc trưng
                    features = extract_color_histogram(img_path, bins=16)
                    database[folder_name] = features
                    print(f"-> Đã nạp mẫu chuẩn cho nhãn: [{folder_name}]")
                    break  # Chỉ cần 1 ảnh chuẩn nhất cho mỗi loại
    return database


def predict_traffic_sign(test_image_path, database, threshold=0.5):
    """Nhận diện ảnh mới dựa trên Database."""
    print(f"\n[PHÂN TÍCH] Đang kiểm tra ảnh: {test_image_path}")
    test_feat = extract_color_histogram(test_image_path, bins=16)

    best_match = None
    min_dist = float('inf')

    # So sánh với từng mẫu trong Database
    for label, template_feat in database.items():
        dist = np.linalg.norm(template_feat - test_feat)
        print(f" - So sánh với [{label}]: Khoảng cách = {dist:.4f}")

        if dist < min_dist:
            min_dist = dist
            best_match = label

    print("\n" + "=" * 40)
    print("KẾT QUẢ NHẬN DIỆN CUỐI CÙNG")
    print("=" * 40)
    if best_match and min_dist < threshold:
        print(f"DANH TÍNH: >> {best_match.upper()} <<")
        print(f"Độ tin cậy (Khoảng cách): {min_dist:.4f}")
    else:
        print("KẾT QUẢ: KHÔNG NHẬN DIỆN ĐƯỢC (Quá lạ so với mẫu)")
    print("=" * 40)


# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # 1. Tự động nạp tất cả các loại biển cấm có trong thư mục data/cam/
    db = load_database("data/cam/")

    if not db:
        print("Lỗi: Thư mục data/cam trống rỗng!")
    else:
        # 2. Dán ảnh test vào đây
        predict_traffic_sign("data/cam/nguoc_chieu/test2.jpg", db)