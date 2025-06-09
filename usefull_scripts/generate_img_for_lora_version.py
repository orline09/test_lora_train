import os
import torch
from peft import PeftModel
from diffusers import StableDiffusionPipeline

path_lora_weights = "./Lora_folder/cyberpunk_lora"
path_images = f"{path_lora_weights}/images"
os.makedirs(path_images, exist_ok=True)
num_inference_steps = 50

test_prompts = ["lighthouse",
                "bedroom with lamp",
                "ship",
                "light",
                "city"
                "sitting cat"]


def main():
    pipe = StableDiffusionPipeline.from_pretrained("./tiny-sd-segmind",
                                                   torch_dtype=torch.float32,
                                                   local_files_only=True).to("cuda")
    for i, prompt in enumerate(test_prompts):
        image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=7.5).images[0]
        image.save(os.path.join(path_images) + f"/without_lora_{i}.png")

    for lora_version in sorted(os.listdir(path_lora_weights)):
        if os.path.isdir(os.path.join(path_lora_weights, lora_version)) and lora_version != 'images':
            pipe = StableDiffusionPipeline.from_pretrained("./tiny-sd-segmind",
                                                           torch_dtype=torch.float32,
                                                           local_files_only=True).to("cuda")
            pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.join(path_lora_weights, lora_version))
            pipe.unet.set_adapter("default")
            pipe.unet.set_adapters(["default"], weights=[1.0])
            print(pipe.unet.state_dict().keys())
            pipe.unet = pipe.unet.merge_and_unload()

            for i, prompt in enumerate(test_prompts):
                image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=7.5).images[0]
                image.save(os.path.join(path_images) + f"/{lora_version}_{i}.png")


if __name__ == "__main__":
    main()
