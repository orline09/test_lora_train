import os
from transformers import pipeline


class ImageToText:
    def __init__(self,
                 path_local_model: str,
                 device: str):
        self.pipeline_i2t = pipeline("image-to-text", model=path_local_model,
                                     device=device,
                                     use_fast=True)

    def generate_image_captioning(self, path_image):
        """
        :param path_image: str_path
        :return: dict("path_image": "text")
        """
        image_text = {}
        if os.path.isdir(path_image):
            image_files = [f for f in os.listdir(path_image) if f.endswith(('.jpg', '.png', '.jpeg'))]
            for image_file in image_files:
                image_text[image_file] = self.pipeline_i2t(os.path.join(path_image, image_file))[0]['generated_text']
        else:
            image_text[path_image] = self.pipeline_i2t(os.path.join(path_image))[0]['generated_text']
        return image_text
