import os
import torch
import uuid
from typing import Union, Optional, Iterable
from diffusers import StableDiffusionPipeline
from peft import PeftModel

path_lora_weights = "./Lora_folder/cyberpunk_lora_r64"
path_save_images = f"{path_lora_weights}/images"
path_sd_model = './models/tiny-sd-segmind'
num_inference_steps = 32
guidance_scale = 7.5
device = "cuda" if torch.cuda.is_available() else "cpu"


#TODO в идеале рассмотреть вариант отброса старой lora
#тоже формат использвоания многогранен
def pipeline_generate_images_and_save(path_model_sd: str,
                             path_save_image: str,
                             prompt: Union[str, list],
                             num_inference_steps=50,
                             path_sd_model: Optional[str] = None):
    os.makedirs(path_save_images, exist_ok=True)
    t2i_pipeline = StableDiffusionPipeline.from_pretrained(path_model_sd,
                                                           local_files_only=True).to(device)
    if path_sd_model:
        t2i_pipeline.unet = PeftModel.from_pretrained(t2i_pipeline.unet, path_lora_weights)
        t2i_pipeline.unet.set_adapter("default")
        t2i_pipeline.unet.set_adapters(["default"], weights=[1.0])
        t2i_pipeline.unet = t2i_pipeline.unet.merge_and_unload()

    if isinstance(prompt, str):
        image = t2i_pipeline(prompt,
                             num_inference_steps=num_inference_steps,
                             guidance_scale=guidance_scale).images[0]
        image.save(os.path.join(path_save_image), f"/{str(uuid.uuid4())}.png")
    elif isinstance(prompt, Iterable):
        for text in prompt:
            image = t2i_pipeline(text,
                                 num_inference_steps=num_inference_steps,
                                 guidance_scale=guidance_scale).images[0]
            image.save(os.path.join(path_save_image), f"/{str(uuid.uuid4())}.png")
