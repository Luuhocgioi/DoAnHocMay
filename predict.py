import cv2
import numpy as np
import joblib
import os
import random
import matplotlib.pyplot as plt
# IMPORT hàm trích xuất gốc từ file của bạn
from color_histogram_extractor import extract_color_histogram

# 1. Tải model (Đã sửa lại tên cho khớp với file train_svm.py)
model_path = 'traffic_sign_model.pkl'
try:
    model = joblib.load(model_path)
    print(f"[INFO] Đã tải model '{model_path}' thành công!")
except FileNotFoundError:
    print(f"[LỖI] Không tìm thấy '{model_path}'. Hãy chạy train_svm.py trước!")
    exit()

# 2. TỰ ĐỘNG LẤY ẢNH TEST
# Đổi lại đường dẫn này tới thư mục valid/test của bạn nhé
valid_path = r"D:\Downloads\traffic signs.v7i.folder\valid"
all_images = []
for root, dirs, files in os.walk(valid_path):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            all_images.append(os.path.join(root, file))

if len(all_images) > 0:
    img_path = random.choice(all_images)
    actual_label = os.path.basename(os.path.dirname(img_path))

    # Đọc ảnh để hiển thị
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3. DỰ ĐOÁN (Gọi hàm truyền vào ĐƯỜNG DẪN ẢNH giống y lúc Train)
    features = extract_color_histogram(img_path, bins=16)

    if features is not None:
        # Sklearn yêu cầu mảng 2D (batch), nên ta bọc features trong list: [features]
        prediction = model.predict([features])[0]

        # 4. HIỂN THỊ KẾT QUẢ BẰNG MATPLOTLIB
        plt.figure(figsize=(10, 5))

        # Bên trái: Hiển thị ảnh
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        title_color = 'green' if actual_label == prediction else 'red'
        plt.title(f"Thực tế: {actual_label}\nAI Đoán: {prediction}", color=title_color, fontsize=14, fontweight='bold')
        plt.axis('off')

        # Bên phải: Hiển thị biểu đồ Histogram (Đặc trưng)
        plt.subplot(1, 2, 2)
        # Tạo trục X từ 0 đến 15 (vì bins=16)
        x_bins = np.arange(len(features))
        plt.bar(x_bins, features, color='teal')
        plt.title("Đặc trưng Color Histogram (16 Bins kênh Hue)")
        plt.xlabel("Thùng màu (Bins)")
        plt.ylabel("Tỉ lệ (Normalized)")
        plt.xticks(x_bins)  # Hiển thị đủ 16 vạch

        print(f"\n=> Nhãn gốc: {actual_label} | AI đoán: {prediction}")
        plt.tight_layout()
        plt.show()
else:
    print("Không tìm thấy ảnh test trong thư mục!")