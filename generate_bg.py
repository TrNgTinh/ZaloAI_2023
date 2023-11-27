from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys
import pandas as pd
import time

current_path = Path(__file__).resolve().parent

SD_path = current_path / 'SD'
sys.path.append(str(SD_path))
from SD.sd import BackgroundGenerator

from Translate_Vi2En.vi2en import TranslatorModule

#processor = BackgroundGenerator()

translator = TranslatorModule()

# Load data from the CSV file
csv_file = '/home/tinhtn/tinhtn/Banner/Zalo/Data/train/info.csv'
data = pd.read_csv(csv_file)

# Iterate through rows in the DataFrame
for index, row in data.iterrows():
    start_time = time.time()
    # Extract information from the row
    prompt_id = row['id']
    caption = row['caption']
    #caption_en = translator.translate_vi2en(row['caption'])
    output_filename = f"{prompt_id}.jpg"

    #data.at[index, 'caption_en'] = caption_en[0]

    # Check if description is not empty before translating
    if not pd.isnull(row['description']) and row['description'].strip() != "":
        description_en = translator.translate_vi2en(row['description'])
        data.at[index, 'description_en'] = description_en[0] if description_en else ""

    # Check if moreInfo is not empty before translating
    if not pd.isnull(row['moreInfo']) and row['moreInfo'].strip() != "":
        moreInfo_en = translator.translate_vi2en(row['moreInfo'])
        data.at[index, 'moreInfo_en'] = moreInfo_en[0] if moreInfo_en else ""

    # Check if caption is not empty before translating
    if not pd.isnull(row['caption']) and row['caption'].strip() != "":
        caption_en = translator.translate_vi2en(row['caption'])
        data.at[index, 'caption_en'] = caption_en[0] if caption_en else ""

    # Process the image using the prompt and save it
    #prompt = caption_en
    #processor.process_image(prompt, "ouput_bg/" + output_filename)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Processed {output_filename} in {elapsed_time:.2f} seconds")


output_csv_file = '/home/tinhtn/tinhtn/Banner/Zalo/Data/train/infor_english.csv'
data.to_csv(output_csv_file, index=False)
