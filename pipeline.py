from PIL import Image, ImageDraw, ImageFont
import torch
from pathlib import Path
import sys

current_path = Path(__file__).resolve().parent
bg_path = Path('/home/tinhtn/tinhtn/Banner/Zalo/Data/test/images/')
SD_path = current_path / 'SD'
sys.path.append(str(SD_path))
#from SD.sd import BackgroundGenerator
# Lấy đường dẫn của thư mục chứa b.py
LayoutDETR_path = current_path / 'LayoutDETR'
sys.path.append(str(LayoutDETR_path))
from LayoutDETR.generate import generate_images
import dnnlib
import legacy


network_pkl = str(LayoutDETR_path / "checkpoints/layoutdetr_ad_banner.pkl")

print('Loading networks from "%s"...' % network_pkl)
device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

import pandas as pd
csv_file = '/home/tinhtn/tinhtn/Banner/Zalo/Data/test/info.csv'
data = pd.read_csv(csv_file)
for index, row in data.iterrows():
    if index == 1:
        break
    bg = bg_path / f"{row['id']}.jpg"
    #strings='EVERYTHING 10% OFF|Friends & Family Savings Event|SHOP NOW|COE FRIEND10'
    #string_labels='header|body text|button|disclaimer / footnote'
    #strings = row['description'] + '|' + row['moreInfo']  
    description_str = str(row['description']) if not pd.isna(row['description']) else ''
    more_info_str = str(row['moreInfo']) if not pd.isna(row['moreInfo']) else ''
    strings = description_str + '|' + more_info_str
    print(strings)
    
    string_labels='body text|disclaimer / footnote'
    outfile= str(current_path / f"result_temp/output/{row['id']}.jpg")
    bg_preprocessing = 256
    out_jittering_strength = 0.0
    out_postprocessing = "horizontal_left_aligned"
    generate_images(network_pkl, bg, bg_preprocessing, strings, string_labels, outfile, out_jittering_strength, out_postprocessing, G)
    
#processor = BackgroundGenerator()
#prompt = ["Increase the luxury and elegance of your car. With high-quality 6d car floor mats."]
#processor.process_image(prompt)

#bg = str(current_path / "result_temp/bg.png")
#bg_preprocessing = 256
#strings='EVERYTHING 10% OFF|Friends & Family Savings Event|SHOP NOW|CODE FRIEND10'
#string_labels='header|body text|button|disclaimer / footnote'
#outfile= str(current_path / 'result_temp/output/test')
#out_jittering_strength = 0.0
#out_postprocessing = "horizontal_left_aligned"

#generate_images(network_pkl, bg, bg_preprocessing, strings, string_labels, outfile, out_jittering_strength, out_postprocessing)

#sys.path.append(str(current_path))