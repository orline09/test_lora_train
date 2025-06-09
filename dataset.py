import os
import pandas as pd
from datasets import Dataset, Image
from image_to_text import ImageToText


class DatasetImageFromPath:
    @staticmethod
    def create_dataset(path_images: str,
                       path_models_i2t: str = None,
                       device: str = None):
        """
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
        dataset = Dataset.from_pandas(pd.DataFrame(data)).cast_column("image", Image())
        return dataset
