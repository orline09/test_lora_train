import torch
from typing import Optional
from diffusers import StableDiffusionPipeline
from peft import PeftModel


# TODO предусмотреть выпиливание прошлых лора и добавление новой
class PipelineRunLora:
    def __init__(self,
                 models_t2i_diffusion_path: str,
                 path_lora: Optional[str]):
        self.models_t2i_diffusion_path = models_t2i_diffusion_path
        self.path_lora = path_lora

    def generate_images(self,
                        prompts: list,
                        num_inference_steps: int = 50,
                        guidance_scale: float = 7.5,
                        weight_lora: float = 1.,
                        word_style: Optional[str] = None):
        """
        list[text] -> list[image]
        :param prompts: list[text] for generation
        :param num_inference_steps:
        :param guidance_scale:
        :param weight_lora:
        :return: list[image]
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t2i_pipeline = StableDiffusionPipeline.from_pretrained(self.models_t2i_diffusion_path,
                                                               local_files_only=True).to(device)
        if self.path_lora:
            t2i_pipeline.unet = PeftModel.from_pretrained(t2i_pipeline.unet, self.path_lora)
            t2i_pipeline.unet.set_adapter("default")
            t2i_pipeline.unet.set_adapters(["default"], weights=[weight_lora])
            t2i_pipeline.unet = t2i_pipeline.unet.merge_and_unload()

        images = []
        for prompt in prompts:
            if word_style:
                prompt = f"{word_style} " + prompt
            image = t2i_pipeline(prompt,
                                 num_inference_steps=num_inference_steps,
                                 guidance_scale=guidance_scale).images[0]
            images.append(image)

        return images
