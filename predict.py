from translate_info import build_faiss_index, embed_single_value, relatedness_fn, TranslatorProcessor
import pandas as pd
from pathlib import Path
import sys

current_path = Path(__file__).resolve().parent
SD_path = current_path / 'SD_Thang'
sys.path.append(str(SD_path))
from SD_Thang.sd import BackgroundGenerator

def predict(row):
    pass


input_csv_file = '/data/thangdq/SDXL/info.csv'
output_csv_file = '/data/thangdq/SDXL/info_thuoc_en_phan_loai.csv'
cate_file = '/data/thangdq/SDXL/loc.txt'


translator_processor = TranslatorProcessor(input_csv_file, cate_file)
bg = BackgroundGenerator(output_directory="images/")

 
df = pd.read_csv(input_csv_file)

for i, row in df.iterrows():
    row = translator_processor.categorical_row(row, cate_file, output_csv_file) 
    bg.process_image_for_row(row)
    
    break




