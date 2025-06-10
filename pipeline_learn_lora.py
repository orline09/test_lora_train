import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from loguru import logger
from tqdm.auto import tqdm

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from peft import LoraConfig, get_peft_model
from prodigyopt import Prodigy

from dataset import DatasetImageFromPath


class PipelineLearnLora:
    def __init__(self):
        self.dataset = None
        self.dataloader = None
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.scheduler = None

    def load_model(self,
                   models_t2i_diffusion_path,
                   device):
        self.tokenizer = AutoTokenizer.from_pretrained(models_t2i_diffusion_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(models_t2i_diffusion_path, subfolder="text_encoder").to(
            device)
        self.vae = AutoencoderKL.from_pretrained(models_t2i_diffusion_path, subfolder="vae").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(models_t2i_diffusion_path, subfolder="unet",
                                                         local_files_only=True)
        self.scheduler = DDPMScheduler.from_pretrained(models_t2i_diffusion_path, subfolder="scheduler")

    def create_dataset(self,
                       images_dir,
                       models_i2t,
                       device):
        self.dataset = DatasetImageFromPath.create_dataset(images_dir, models_i2t, device)

    def create_dataloader(self,
                          batch_size,
                          **params):
        preprocess = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        def collate_fn(examples):
            images = [preprocess(example["image"]) for example in examples]
            texts = [example["text"] for example in examples]
            pixel_values = torch.stack(images)  # [B, 3, 512, 512]
            text_inputs = self.tokenizer(texts, max_length=77, padding="max_length", truncation=True,
                                         return_tensors="pt")
            input_ids = text_inputs.input_ids
            batch = {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
            }
            return batch

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def learn_lora(self,
                   path_save_lora,
                   device,
                   lora_config: LoraConfig,
                   num_epochs: int = 128,
                   every_epoch_save_lora: int = 5,
                   start_epoch_save: int = 10,
                   **params
                   ):
        os.makedirs(path_save_lora, exist_ok=True)
        optimizer = Prodigy(self.unet.parameters(),
                            lr=1.0,
                            weight_decay=0.01,
                            betas=(0.9, 0.99),
                            safeguard_warmup=True,
                            use_bias_correction=True)
        self.unet = get_peft_model(self.unet, lora_config)
        logger.info(self.unet.print_trainable_parameters())
        self.unet.to(device)

        for epoch in range(num_epochs):
            total_train_loss = 0.
            self.unet.train()
            for batch in tqdm(self.dataloader):
                # latent and encoder_hidden_states
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                with torch.no_grad():
                    latents = self.vae.encode(pixel_values).latent_dist.sample() * self.vae.config.scaling_factor
                    encoder_hidden_states = self.text_encoder(input_ids)[0]

                # noise and predict
                noise = torch.randn_like(latents).to(device)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                noisy_images = self.scheduler.add_noise(latents, noise, timesteps)

                # set target
                noise_pred = self.unet(noisy_images, timesteps, encoder_hidden_states).sample
                if self.scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif self.scheduler.config.prediction_type == "v_prediction":
                    target = noisy_images.get_velocity(latents, noise, timesteps)

                # Loss и backward
                loss = torch.nn.functional.mse_loss(noise_pred, target, reduction="mean")
                loss.backward()
                optimizer.step()
                # lr_scheduler.step(
                optimizer.zero_grad()
                total_train_loss += loss.item()

            logger.info((f"Epoch {epoch + 1} Done"))

            if (epoch % every_epoch_save_lora == 0) and epoch >= start_epoch_save:
                os.makedirs(os.path.join(path_save_lora, f"{epoch}"))
                self.unet.save_pretrained(os.path.join(path_save_lora, f"{epoch}"))

        self.unet.save_pretrained(os.path.join(path_save_lora, f"{epoch}_last"))

    def run_pipeline_learn_lora(self,
                                images_dir,
                                models_t2i_diffusion_path,
                                models_i2t_path,
                                path_save_lora,
                                lora_config,
                                learn_lora_params
                                ):
        logger.info("start run pipeline")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model(models_t2i_diffusion_path, device)
        self.create_dataset(images_dir, models_i2t_path, device)
        self.create_dataloader(**learn_lora_params)
        self.learn_lora(path_save_lora, device, lora_config, **learn_lora_params)
        logger.info("pipeline done")


if __name__ == "__main__":
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
    )
    learn_lora_params = {"batch_size": 2,
                         "num_epochs": 128,
                         "every_epoch_save_lora": 5,
                         "start_epoch_save": 10}

    pipe = PipelineLearnLora()
    pipe.run_pipeline_learn_lora(images_dir="./small_datasets/cyberpunk",
                                 models_t2i_diffusion_path="./models/tiny-sd-segmind",
                                 models_i2t_path="./models/images2text_model",
                                 path_save_lora='./Lora_folder/cyberpunk_lora_r64_no_gausine',
                                 lora_config=lora_config,
                                 learn_lora_params=learn_lora_params
                                 )
