import locale
import torch
from diffusers import StableDiffusionXLPipeline

locale.getpreferredencoding = lambda: "UTF-8"

class BackgroundGenerator:
    def __init__(self, seed= 7183698734589870, device="cuda", height=512, width=1024):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "segmind/SSD-1B",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir="SD_Thang/checkpoint"
        ).to(device)

        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.latents = torch.randn(
            (1, self.pipe.unet.in_channels, height // 8, width // 8),
            generator=self.generator,
            device=device
        )

    def process_image(self, prompt, output, negative_prompt="ugly, blurry, poor quality", num_inference_steps=35, height=512, width=1024):
        prompt = f"A product advertising background banner with description {prompt}"

        image = self.pipe(prompt, num_inference_steps=num_inference_steps, height=height, width=width, negative_prompt=negative_prompt).images[0]
        image = image.resize((1024, 533))
        
        image.save(f"{output}")