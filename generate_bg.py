from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys
import pandas as pd
import time

current_path = Path(__file__).resolve().parent

SD_path = current_path / 'SD_Thang'
sys.path.append(str(SD_path))
from SD_Thang.sd import BackgroundGenerator

def process_image_for_row(row, output_directory="Output/out_bg/"):
    # Process the image using the prompt and save it
    prompt = row['caption_en']
    output_filename = f"{row['bannerImage']}"
    bg.process_image(prompt, output_directory + output_filename)
    return elapsed_time

class BackgroundProcessor:
    def __init__(self, output_directory="Output/out_bg/"):
        self.output_directory = output_directory
        self.bg = BackgroundGenerator()

    def process_image_for_row(self, row):
        
        # Process the image using the prompt and save it
        prompt = row['caption_en']
        output_filename = f"{row['bannerImage']}"
        self.bg.process_image(prompt, self.output_directory + output_filename)
        








