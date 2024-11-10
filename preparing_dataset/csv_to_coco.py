import pandas as pd
import json
import os
from PIL import Image

# Параметры
csv_file = '../datasets/dataset_original/annotation.csv'  # Путь к вашему CSV-файлу с разметкой
annotated_images_dir = '../datasets/dataset_original/images'  # Папка с размеченными изображениями
unannotated_images_dir = '../datasets/dataset_original/images_empty'  # Папка с неразмеченными изображениями
output_file = '../datasets/dataset_coco/coco_annotations.json'  # Имя выходного JSON-файла

# Загрузка данных
data = pd.read_csv(csv_file)

# Структура COCO
coco_format = {
    "info": {
        "description": "Dataset converted to COCO format",
        "version": "1.0",
        "year": 2024,
        "contributor": "Nikita Pogudalov",
        "date_created": "2024-11-08"
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

# Словарь для хранения уникальных категорий
categories = {}

# Генерация ID
annotation_id = 1
image_id = 1
image_ids = {}

coco_format['categories'].append({
    "id": 0,
    "name": 'animal',
    "supercategory": "none"
})

coco_format['categories'].append({
    "id": 1,
    "name": 'no animal',
    "supercategory": "none"
})

# Обработка размеченных изображений
for _, row in data.iterrows():
    image_name = row['Name']
    image_path = os.path.join(annotated_images_dir, image_name)
    
    if image_name not in image_ids:
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                width, height = img.size
            coco_format['images'].append({
                "id": image_id,
                "file_name": image_name,
                "height": height,
                "width": width
            })
            image_ids[image_name] = image_id
            image_id += 1
        else:
            print(f"Изображение {image_name} не найдено в папке {annotated_images_dir}. Пропуск...")
            continue
    
    # Преобразование Bbox
    bbox = [float(x) for x in row['Bbox'].split(',')]
    x_center, y_center, width, height = bbox
    x_min = (x_center - width / 2) * width
    y_min = (y_center - height / 2) * height
    
    # Добавление аннотации
    coco_format['annotations'].append({
        "id": annotation_id,
        "image_id": image_ids[image_name],
        "category_id": 0,
        "bbox": [x_min, y_min, width * width, height * height],
        "area": width * height,
        "iscrowd": 0
    })
    annotation_id += 1

# Обработка неразмеченных изображений
for unannotated_image in os.listdir(unannotated_images_dir):
    image_path = os.path.join(unannotated_images_dir, unannotated_image)
    if os.path.isfile(image_path) and unannotated_image not in image_ids:
        with Image.open(image_path) as img:
            width, height = img.size
            
        coco_format['images'].append({
            "id": image_id,
            "file_name": unannotated_image,
            "height": height,
            "width": width
        })
        
        # Добавление аннотации
        coco_format['annotations'].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [],
            "area": 0,
            "iscrowd": 0
        })
    
        image_ids[unannotated_image] = image_id
        image_id += 1
        annotation_id += 1
        
# Сохранение JSON
with open(output_file, 'w') as f:
    json.dump(coco_format, f, indent=4)

print(f'Аннотации в формате COCO сохранены в файл {output_file}')
