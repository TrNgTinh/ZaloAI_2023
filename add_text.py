from PIL import Image, ImageDraw, ImageFont
import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
import sys
import time

current_path = Path(__file__).resolve().parent
LayoutDETR_path = current_path / 'LayoutDETR'
sys.path.append(str(LayoutDETR_path))
from LayoutDETR.generate import generate_images
import legacy
import dnnlib

class AddTextProcessor:
    def __init__(self, output_directory, network_pkl = str(LayoutDETR_path / "checkpoints/layoutdetr_ad_banner.pkl")):
        self.network_pkl = network_pkl
        self.result_path = output_directory
        self.G = self.load_network()

    def load_network(self):
        with dnnlib.util.open_url(self.network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(torch.device('cuda'))
        G.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return G

    def process_row(self, row, bg_path):
        bg_path = Path(bg_path) / f"{row['bannerImage']}"
        description_str = str(row['description']) if not pd.isna(row['description']) else ''
        more_info_str = str(row['moreInfo']) if not pd.isna(row['moreInfo']) else ''
        strings = description_str + '|' + more_info_str

        string_labels = 'body text|button'
        #outfile = str(self.result_path / f"images/{row['id']}")
        outfile = bg_path
        bg_preprocessing = 256
        out_jittering_strength = 0.0
        out_postprocessing = "horizontal_left_aligned"
        generate_images(self.network_pkl, bg_path, bg_preprocessing, strings, string_labels, outfile, out_jittering_strength, out_postprocessing, self.G)


