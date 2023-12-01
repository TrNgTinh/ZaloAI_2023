import locale
import torch
from diffusers import StableDiffusionXLPipeline

locale.getpreferredencoding = lambda: "UTF-8"

class BackgroundGenerator:
    def __init__(self, seed= 7183698734589870, output_directory="results/",  device="cuda", height=512, width=1024):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "segmind/SSD-1B",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir="cache"
        ).to(device)

        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.latents = torch.randn(
            (1, self.pipe.unet.in_channels, height // 8, width // 8),
            generator=self.generator,
            device=device
        )
        self.output_directory = output_directory

    def process_image(self, prompt, output, negative_prompt="ugly, blurry, poor quality", num_inference_steps=35, height=512, width=1024):
        prompt = f"A product advertising background banner with description {prompt}"
        with torch.autocast("cuda"):
            image = self.pipe(prompt, latents = self.latents,num_inference_steps=num_inference_steps, height=height, width=width, negative_prompt=negative_prompt).images[0]
            image = image.resize((1024, 533))   
            image.save(f"{output}")
            return image

    def process_image_for_row(self, row, nb=None):
        # Process the image using the prompt and save it
        prompt = row['caption_en']
        output_filename = f"{row['bannerImage']}"
        if nb:
            self.process_image(prompt, self.output_directory+ nb + output_filename)
        else:
            self.process_image(prompt, self.output_directory + output_filename)
        

    
