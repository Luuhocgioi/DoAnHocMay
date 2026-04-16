import os
from color_histogram_extractor import extract_color_histogram

data_path = "data/cam/nguoc_chieu"
features_list = []

print(f"--- ĐANG TRÍCH XUẤT DỮ LIỆU TỪ: {data_path} ---")

for file_name in os.listdir(data_path):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        full_path = os.path.join(data_path, file_name)

        # Gọi hàm extractor bạn đã viết
        hist = extract_color_histogram(full_path, bins=16)

        if hist is not None:
            features_list.append(hist)

print(f"\n=> Đã xử lý xong {len(features_list)} mẫu cho biển Ngược Chiều.")