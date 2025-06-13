import os
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from loguru import logger
from tqdm.auto import tqdm

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler
from peft import LoraConfig, get_peft_model
from prodigyopt import Prodigy

from dataset import DatasetImageFromPath


# TODO need add interface
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
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(models_t2i_diffusion_path, subfolder="scheduler")

        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
        for param in self.vae.parameters():
            param.requires_grad_(False)

    def create_dataset(self,
                       images_dir: str,
                       models_i2t: Optional[str],
                       file_with_texts: Optional[str],
                       device: str):
        if file_with_texts:
            self.dataset = DatasetImageFromPath.create_dataset_from_file(path_file=file_with_texts,
                                                                         path_images=images_dir)
        elif models_i2t:
            self.dataset = DatasetImageFromPath.create_dataset_generate_caption(path_images=images_dir,
                                                                                path_models_i2t=models_i2t,
                                                                                device=device)
        else:
            raise Exception("no text in dataset, and dataset is None!!")

    def create_dataloader(self,
                          batch_size,
                          style_word: Optional[str],
                          **params):
        preprocess = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.Lambda(lambda x: x.convert("RGB")),
            # transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        def collate_fn(examples):
            images = [preprocess(example["image"]) for example in examples]
            if style_word:
                texts = [f"{style_word} " + example["text"]
                         if example["text"].find(style_word) > -1
                         else example["text"]
                         for example in examples]
            else:
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

    def fit_lora(self,
                 path_save_lora,
                 device,
                 lora_config: dict,
                 optimizator_params: dict,
                 num_epochs: int = 128,
                 every_epoch_save_lora: int = 5,
                 start_epoch_save: int = 10,
                 **params
                 ):
        """
        :param lora_config: https://huggingface.co/docs/peft/main/en/package_reference/lora
        :param optimizator_params: https://github.com/konstmish/prodigy/blob/main/prodigyopt/prodigy.py
        :return: None
        """
        lora_config = LoraConfig(**lora_config)
        os.makedirs(path_save_lora, exist_ok=True)
        optimizer = Prodigy(self.unet.parameters(),
                            **optimizator_params)
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
                timesteps = timesteps.long()
                noisy_images = self.scheduler.add_noise(latents, noise, timesteps)

                # set target
                noise_pred = self.unet(noisy_images, timesteps, encoder_hidden_states).sample
                if self.scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif self.scheduler.config.prediction_type == "v_prediction":
                    target = noisy_images.get_velocity(latents, noise, timesteps)

                # Loss и backward
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(noise_pred, target, reduction="mean")
                loss.backward()
                optimizer.step()
                # lr_scheduler.step(
                total_train_loss += loss.item()

            logger.info(f"Epoch {epoch + 1} Done, avg_loss={total_train_loss / len(self.dataloader)}")

            if (epoch % every_epoch_save_lora == 0) and epoch >= start_epoch_save:
                os.makedirs(os.path.join(path_save_lora, f"{epoch}"))
                self.unet.save_pretrained(os.path.join(path_save_lora, f"{epoch}"))

        self.unet.save_pretrained(os.path.join(path_save_lora, f"{epoch}_last"))

    def run_pipeline_fit_lora(self,
                              images_dir,
                              models_t2i_diffusion_path,
                              models_i2t_path,
                              path_save_lora,
                              lora_config,
                              optimizator_params,
                              fit_lora_params,
                              file_with_texts: Optional[str] = None
                              ):
        logger.info("start run pipeline")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model(models_t2i_diffusion_path, device)
        self.create_dataset(images_dir, models_i2t_path, file_with_texts, device)
        self.create_dataloader(**fit_lora_params)
        self.fit_lora(path_save_lora=path_save_lora,
                      device=device,
                      lora_config=lora_config,
                      optimizator_params=optimizator_params,
                      **fit_lora_params)
        logger.info("pipeline done")


if __name__ == "__main__":
    lora_config = {"r": 16,
                   "lora_alpha": 32,
                   "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
                   "lora_dropout": 0.1}

    optimizator_params = {"lr": 1.0,
                          "weight_decay": 0.01,
                          "betas": (0.9, 0.99),
                          "safeguard_warmup": True,
                          "use_bias_correction": True}

    fit_lora_params = {"batch_size": 2,
                       "style_word": None,
                       "num_epochs": 128,
                       "every_epoch_save_lora": 5,
                       "start_epoch_save": 10}

    images_dir = "./small_datasets/cyberpunk"
    models_t2i_diffusion_path = "./models/tiny-sd-segmind"
    models_i2t_path = "./models/images2text_model"
    path_save_lora = './Lora_folder/cyberpunk_lora_r64_no_gausine'
    image_save_path = './Lora_folder/cyberpunk_lora_r64_no_gausine/images'

    pipe = PipelineLearnLora()
    pipe.run_pipeline_fit_lora(images_dir=images_dir,
                               models_t2i_diffusion_path=models_t2i_diffusion_path,
                               models_i2t_path=models_i2t_path,
                               path_save_lora=path_save_lora,
                               lora_config=lora_config,
                               optimizator_params=optimizator_params,
                               fit_lora_params=fit_lora_params
                               )
