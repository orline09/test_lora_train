import os
import torch
from torchvision import transforms
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel
from prodigyopt import Prodigy
from loguru import logger
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from dataset import DatasetImageFromPath


#TODO параметры вверху файла, подтягивание можно сделать N способами
# что в env / что куда-то еще is diffrent
images_dir = "./small_datasets/cyberpunk"
models_t2i_diffusion_path = "./models/tiny-sd-segmind"
models_i2t = "./models/images2text_model"
path_save_lora = './cyberpunk_lora_r64_no_gausine'
batch_size = 2
num_epochs = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
every_epoch_save_lora = 8
start_epoch_save = 4

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
)

preprocess = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

optimizer = Prodigy(unet.parameters(),
                    lr=1.0,
                    weight_decay=0.01,
                    betas=(0.9, 0.99),
                    safeguard_warmup=True,
                    use_bias_correction=True)

def main():
    os.makedirs(path_save_lora, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(models_t2i_diffusion_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(models_t2i_diffusion_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(models_t2i_diffusion_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(models_t2i_diffusion_path, subfolder="unet", local_files_only=True)
    scheduler = DDPMScheduler.from_pretrained(models_t2i_diffusion_path, subfolder="scheduler")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    def collate_fn(examples):
        images = [preprocess(example["image"]) for example in examples]
        texts = [example["text"] for example in examples]
        pixel_values = torch.stack(images)  # [B, 3, 512, 512]
        text_inputs = tokenizer(texts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = text_inputs.input_ids
        batch = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }
        return batch

    train_dataset = DatasetImageFromPath.create_dataset(images_dir, models_i2t, device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    unet = get_peft_model(unet, lora_config)
    logger.info(unet.print_trainable_parameters())



    unet.to(device)
    for epoch in range(num_epochs):
        total_train_loss = 0.
        unet.train()
        for batch in tqdm(train_dataloader):
            # latent and encoder_hidden_states
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            # noise and predict
            noise = torch.randn_like(latents).to(device)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            noisy_images = scheduler.add_noise(latents, noise, timesteps)

            # set target
            noise_pred = unet(noisy_images, timesteps, encoder_hidden_states).sample
            if scheduler.config.prediction_type == "epsilon":
                target = noise
            elif scheduler.config.prediction_type == "v_prediction":
                target = noisy_images.get_velocity(latents, noise, timesteps)

            # Loss и backward
            loss = torch.nn.functional.mse_loss(noise_pred, target, reduction="mean")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
        # lr_scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        logger.info((f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}"))

        if (epoch % every_epoch_save_lora == 0) and epoch >= start_epoch_save:
            os.makedirs(os.path.join(path_save_lora, f"{epoch}"))
            unet.save_pretrained(os.path.join(path_save_lora, f"{epoch}"))

    unet.save_pretrained(os.path.join(path_save_lora, f"{epoch}_last"))


if __name__ == "__main__":
    main()
