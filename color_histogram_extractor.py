import cv2
import numpy as np


def extract_color_histogram(image_path, bins=16):
    """
    Hàm trích xuất đặc trưng màu sắc với Log giải trình chi tiết.
    """
    # 1. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"--- LỖI: Không tìm thấy ảnh tại {image_path} ---")
        return None

    # 2. Tiền xử lý (Chuẩn hóa kích thước)
    # Chúng ta dùng 64x64 để mọi ảnh đều có tổng số pixel bằng nhau (4096 pixels)
    img_resized = cv2.resize(img, (64, 64))

    # 3. Chuyển đổi không gian màu sang HSV
    # Bản chất: Tách riêng màu sắc (Hue) khỏi độ sáng (Value)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # 4. Tính toán Histogram cho kênh Hue (H)
    # Range của H trong OpenCV là [0, 180]
    hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])

    # 5. Chuẩn hóa Vector (Normalization)
    # Công thức: L1-Norm (Tổng các bin = 1) giúp tính toán tỉ lệ phần trăm
    hist_norm = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()

    # --- HỆ THỐNG LOG CHUYÊN GIA ---
    print("\n" + "=" * 60)
    print(f"LOG TOÁN HỌC: TRÍCH XUẤT ĐẶC TRƯNG CHO {image_path.upper()}")
    print("=" * 60)
    print(f"[THÔNG SỐ CẤU HÌNH]")
    print(f" - Kích thước ảnh chuẩn hóa: 64x64 pixels")
    print(f" - Tổng số điểm ảnh (N): {64 * 64}")
    print(f" - Số lượng thùng chứa (Bins): {bins}")
    print(f" - Độ rộng mỗi Bin: {180 / bins} độ (trên thang 180)")

    print(f"\n[CHI TIẾT VECTOR ĐẶC TRƯNG]")
    for i in range(len(hist_norm)):
        # Hiển thị biểu đồ cột bằng ký tự để minh họa
        bar = "|" * int(hist_norm[i] * 100)
        print(
            f" Bin {i:02d} ({int(i * (180 / bins)):>3d}° - {int((i + 1) * (180 / bins)):>3d}°): {hist_norm[i]:.4f} {bar}")

    print("\n[GIẢI TRÌNH BẢN CHẤT]")
    max_idx = np.argmax(hist_norm)
    print(f" - Giá trị lớn nhất tập trung tại Bin: {max_idx}")
    print(f" - Tỉ lệ màu sắc chiếm ưu thế: {hist_norm[max_idx] * 100:.2f}%")
    print("=" * 60 + "\n")

    return hist_norm

# Thử nghiệm với một tấm ảnh
# feature_vector = extract_color_histogram("bien_bao_test.jpg", bins=16)