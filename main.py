import os

from pipeline_fit_lora import PipelineLearnLora
from pipeline_run_lora import PipelineRunLora


def main():
    lora_config = {"r": 64,
                   "lora_alpha": 128,
                   "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
                   "lora_dropout": 0.1}

    fit_lora_params = {"batch_size": 2,
                       "num_epochs": 10,
                       "every_epoch_save_lora": 3,
                       "start_epoch_save": 3}

    # use prodigi optimizator
    optimizator_params = {"lr": 0.1,
                          "weight_decay": 0.01,
                          "betas": (0.9, 0.99),
                          "safeguard_warmup": True,
                          "use_bias_correction": True}

    images_dir = "./small_datasets/cyberpunk"
    models_t2i_diffusion_path = "./models/tiny-sd-segmind"
    models_i2t_path = "./models/images2text_model"
    path_save_lora = './Lora_folder/cyberpunk_lora_r64_no_gausine'
    image_save_path = './Lora_folder/cyberpunk_lora_r64_no_gausine/images'

    test_prompts = ["lighthouse",
                    "bedroom with lamp",
                    "ship",
                    "light",
                    "city"
                    "sitting cat"]

    pipe = PipelineLearnLora()
    pipe.run_pipeline_fit_lora(images_dir=images_dir,
                               models_t2i_diffusion_path=models_t2i_diffusion_path,
                               models_i2t_path=models_i2t_path,
                               path_save_lora=path_save_lora,
                               lora_config=lora_config,
                               optimizator_params=optimizator_params,
                               fit_lora_params=fit_lora_params
                               )

    for Lora_version_path in [None] + os.listdir(path_save_lora):
        if Lora_version_path:
            pipe_generate = PipelineRunLora(models_t2i_diffusion_path="./models/tiny-sd-segmind",
                                            path_lora=os.path.join(path_save_lora, Lora_version_path))
        else:
            pipe_generate = PipelineRunLora(models_t2i_diffusion_path="./models/tiny-sd-segmind",
                                            path_lora=Lora_version_path)
        os.makedirs(image_save_path, exist_ok=True)
        images = pipe_generate.generate_images(test_prompts, num_inference_steps=20)
        for num, image in enumerate(images):
            image.save(os.path.join(image_save_path) +
                       f"/{Lora_version_path if Lora_version_path else 'no_lora'}_{num}.png")

    print('done')


if __name__ == "__main__":
    main()
