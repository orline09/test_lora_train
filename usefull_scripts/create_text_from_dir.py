from image_to_text import ImageToText

i2t = ImageToText(path_local_model="images2text_model", device='cuda')
path_folder = "./cyberpunk"
file_text = i2t.generate_image_captioning(path_image=path_folder)
with open(f"{path_folder}/texts_from_image.txt", 'w') as f:
    for file in file_text:
        f.write(f"{file_text[file]}\n")
