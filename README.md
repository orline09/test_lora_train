# test_lora_train

classic install
python >= 3.11

<code>python3 -m venv .venv</code>

<code>source .venv/bin/activate</code>

<code>python3 -m pip install --upgrade pip</code>

<code>python3 -m pip install -r requirements.txt</code>

__download tiny models to local__

<code>python3 -m models/download_models_to_local</code>

заполнить верхние поля и запустить pipeline_learn_lora.py

<code>python3 -m pipeline_learn_lora</code>

__pipeline_run_lora.py__ - загатовка под переписывание дальше с TODO

__usefull_scripts/generate_img_for_lora_version.py__ - для разных версий лоры сгенерировать изображения по тестовым промптам и положить их рядышком

__usefull_scripts/create_text_from_dir.py__ - рядом с картинками положить файл с их описаниями

example (left no lora | add lora): 
style acrylic paints
![Логотип проекта](./impact_lora_example/acril_images_with_lora.png)

example (left no lora | and add lora): 
style multic anime
![Логотип проекта](./impact_lora_example/multi_anime.png)

example (left no lora | and add lora): 
style cyberpunk
![Логотип проекта](./impact_lora_example/cyberpank.png)

lets just for fun overfiting lora))