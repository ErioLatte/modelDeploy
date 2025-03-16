from models import SimilarStructureControlNetModel, FlexibleModulatedControlNetModel
import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline
from preprocess import data_prep
# load model here
class ModelLoader():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n\nDEVICE DETECTED IS: {self.device}")
        self.weight_dir = "./weight/E07/"
        self.sd_model_path = "stabilityai/sd-turbo"
        self.original_controlnet_path = "thibaud/controlnet-sd21-canny-diffusers"

        self.load_flexible_modulated()
        pass

    def load_flexible_modulated(self):
        print("\n\nLOADING MODEL")
        # load controlnet weight
        controlnet = FlexibleModulatedControlNetModel.from_pretrained(
            self.weight_dir, torch_dtype=torch.float16, use_safetensors=True, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
            ).to(self.device)
        print("\n\nLOADING pipeline")
        #generate pipe stuff
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.sd_model_path, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
            ).to(self.device)

        torch.cuda.empty_cache()

    def generate_image(self, canny_image_path, prompt, upload_dir, image_name, extension):
        print(f"\n\nOPENING CANNY IMAGE: {canny_image_path}")
        canny_image = Image.open(canny_image_path)
        print(f"\nGENERATING OUTPUT:")
        output = self.pipe(
            prompt, image=canny_image, guidance_scale=9
        ).images[0]
        image_save_path = f'{upload_dir}{image_name}_rs.{extension}'
        print(f"\n\n\nSAVING OUTPUT:")
        output.save(image_save_path)
        return image_save_path


model_loader = ModelLoader()