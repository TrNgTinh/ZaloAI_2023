from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from pathlib import Path

import transformers
from diffusers.utils import load_image
from diffusers import (
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)
from controlnet_aux import PidiNetDetector, HEDdetector
class BackgroundGenerator:
    def __init__(self, checkpoint="lllyasviel/control_v11p_sd15_softedge", width=512, height=512, seed=0, device = "cuda"):
        # Set seed for torch and numpy
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.checkpoint = checkpoint
        self.generator = torch.Generator(device).manual_seed(1024)
        self.processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')

        # Load the controlnet model and pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None,
            requires_safety_checker=False, cache_dir="SD/checkpoint"
        ).to(device)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

    def process_image(self, prompt, output, negative_prompt = "low bad worst blurry", num_inference_steps = 20, height=528 , width=1024):
        processed_image = self.pipe(prompt, num_inference_steps = num_inference_steps, height = height, width = width, generator=self.generator,negative_prompt= negative_prompt).images[0]
        #processed_image = processed_image.resize((1024, 533), Image.ANTIALIAS)
        processed_image.save(output)
        return processed_image
