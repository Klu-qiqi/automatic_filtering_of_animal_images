import pandas as pd
import shutil
import os
from PIL import Image

def cut_and_sort_from_csv(train_data_path, output_dir):
    path_bad = os.path.join(output_dir, '0')
    path_good = os.path.join(output_dir, '1')
    
    os.makedirs(path_good, exist_ok=True)
    os.makedirs(path_bad, exist_ok=True)
    
    # Чтение CSV файла
    annotation_data = pd.read_csv(os.path.join(train_data_path, 'annotation.csv'))

    # Преобразование данных и сохранение в отдельные файлы
    for image_name, group in annotation_data.groupby('Name'):
        yolo_data = group.apply(lambda row: f"{row['Class']} {row['Bbox'].replace(',', ' ')}", axis=1)
        for line in yolo_data:
            line = list(map(float, line.split(' ')))
            image = Image.open(os.path.join(train_data_path, 'images', image_name))
            
            img_width, img_height = image.size  # Получаем размер изображения

            # Распаковываем относительные значения
            center_x, center_y, width, height = line[1:]

            # Преобразуем относительные значения в абсолютные пиксельные координаты
            left = (center_x - width / 2) * img_width
            top = (center_y - height / 2) * img_height
            right = (center_x + width / 2) * img_width
            bottom = (center_y + height / 2) * img_height

            # Обрезаем изображение
            cropped_image = image.crop((left, top, right, bottom))
            if (line[0] == 1):
                cropped_image.save(os.path.join(path_good, image_name))
            else:
                cropped_image.save(os.path.join(path_bad, image_name))

# Загрузка данных
train_data_path = '../datasets/dataset_original'
output_dir = '../datasets/dataset_classificator'

# Создание папки для сохранения, если она не существует
os.makedirs(output_dir, exist_ok=True)

cut_and_sort_from_csv(train_data_path, output_dir)

print(f'Файлы сохранены в папке: {output_dir}')