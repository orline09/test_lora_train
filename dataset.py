import os
import pandas as pd
from loguru import logger
from datasets import Dataset, Image
from image_to_text import ImageToText


class DatasetImageFromPath:
    @staticmethod
    def create_dataset_generate_caption(path_images: str,
                                        path_models_i2t: str = None,
                                        device: str = None):
        """
        use ImageToText model for generate caption (it may be a bad idea for some styles)
        :param path_images: path folder, where images
        :param path_models_i2t: str
        :return: Dataset
        """
        i2t = ImageToText(path_models_i2t, device)
        image_files = [f for f in os.listdir(path_images) if f.endswith(('.jpg', '.png', '.jpeg'))]
        data = []
        for img_file in image_files:
            image_path = os.path.join(path_images, img_file)
            text_image = i2t.generate_image_captioning(image_path)[image_path]
            data.append({"image": image_path, 'text': text_image})
        for i in data:
            print(f"'{i['text']}',")
        dataset = Dataset.from_pandas(pd.DataFrame(data)).cast_column("image", Image())
        logger.info(f"shape of dataset = {pd.DataFrame(data).shape}")
        return dataset

    @staticmethod
    def create_dataset_from_file(path_file: str,
                                 path_images: str):
        """
        :param path_file: path to csv file columns=[image, text]
        :param path_images: path to folder with images
        :return: Dataset
        """
        df_images = pd.read_csv(path_file)
        df_images['image'] = path_images + '/' + df_images['image']
        dataset = Dataset.from_pandas(df_images).cast_column("image", Image())
        logger.info(f"shape of dataset = {df_images.shape}")
        return dataset
