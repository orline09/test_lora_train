from diffusers import DiffusionPipeline
from transformers import pipeline

def main():
    MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    SAVE_PATH = "./stable-diffusion-v1-5"
    DiffusionPipeline.from_pretrained(MODEL_NAME,
                                      cache_dir=SAVE_PATH)

    MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
    SAVE_PATH = "./images2text_model"
    pipe = pipeline("image-to-text", MODEL_NAME)
    pipe.save_pretrained(SAVE_PATH)


if __name__ == '__main__':
    main()
