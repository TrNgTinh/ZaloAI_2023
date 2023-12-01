from PIL import Image, ImageDraw, ImageFont
import torch
from pathlib import Path
import sys
import time
from tqdm import tqdm
import pandas as pd

current_path = Path(__file__).resolve().parent
LayoutDETR_path = current_path / 'LayoutDETR'
sys.path.append(str(LayoutDETR_path))
from LayoutDETR.generate import generate_images
import legacy
import dnnlib

start_time = time.time() 

network_pkl = str(LayoutDETR_path / "checkpoints/layoutdetr_ad_banner.pkl")
print('Loading networks from "%s"...' % network_pkl)
device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

print(G)
print("type", type(G))

csv_file = '/home/tinhtn/tinhtn/Banner/Zalo/Data/test/info.csv'
bg_path = Path('/home/tinhtn/tinhtn/Banner/Zalo/Data/test/images/')
data = pd.read_csv(csv_file)

for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing Data"):

    bg = bg_path / f"{row['bannerImage']}"
    
    description_str = str(row['description']) if not pd.isna(row['description']) else ''
    more_info_str = str(row['moreInfo']) if not pd.isna(row['moreInfo']) else ''
    strings = description_str + '|' + more_info_str

    string_labels='body text|button'
    outfile= str(current_path / f"result_temp/images/{row['id']}")
    bg_preprocessing = 256
    out_jittering_strength = 0.0
    out_postprocessing = "horizontal_left_aligned"
    generate_images(network_pkl, bg, bg_preprocessing, strings, string_labels, outfile, out_jittering_strength, out_postprocessing, G)

end_time = time.time()
print("Time:", end_time - start_time)


    