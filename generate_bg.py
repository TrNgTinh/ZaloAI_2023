from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys
import pandas as pd
import time

current_path = Path(__file__).resolve().parent

SD_path = current_path / 'SD_Thang'
sys.path.append(str(SD_path))
from SD_Thang.sd import BackgroundGenerator

#from Translate_Vi2En.vi2en import TranslatorModule

bg = BackgroundGenerator()


# Load data from the CSV file
csv_file = '/home/tinhtn/tinhtn/Banner/Zalo/Data/info_thuoc_en.csv'
data = pd.read_csv(csv_file)


# Biến để theo dõi số lượng ảnh đã xử lý
image_count = 0

# Biến để tính tổng thời gian
total_time = 0

for index, row in data.iterrows():
    # Kiểm tra xem đã xử lý đủ 100 ảnh chưa
    if image_count >= 2:
        break
    
    start_time = time.time()
    
    # Các bước xử lý ảnh và dịch ngôn ngữ ở đây
    
    output_filename = f"{row['bannerImage']}"

    # Process the image using the prompt and save it
    prompt = row['caption_en']
    bg.process_image(prompt, "Output/out_bg/" + output_filename)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_time += elapsed_time
    
    # Tăng biến đếm số lượng ảnh đã xử lý
    image_count += 1
    
    print(f"Processed {output_filename} in {elapsed_time:.2f} seconds")

# Tính trung bình thời gian chạy
average_time = total_time / image_count if image_count > 0 else 0
print(f"Average processing time for {image_count} images: {average_time:.2f} seconds")







