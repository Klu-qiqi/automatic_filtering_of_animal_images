import pandas as pd
import shutil
import os

def create_labels_from_csv(train_data_path, output_dir):
    # Чтение CSV файла
    annotation_data = pd.read_csv(os.path.join(train_data_path, 'annotation.csv'))

    # Преобразование данных и сохранение в отдельные файлы
    for image_name, group in annotation_data.groupby('Name'):
        # Формирование строк в формате YOLO (0)
        yolo_data = group.apply(lambda row: f"{row['Class']} {row['Bbox'].replace(',', ' ')}", axis=1)
        
        # Сохранение в файл с тем же именем, что и изображение, но с расширением .txt
        label_file = os.path.join(output_dir, 'labels', os.path.splitext(image_name)[0] + '.txt')
        with open(label_file, 'w') as f:
            for line in yolo_data:
                f.write(f"{line}\n")

def create_labels_from_folder(train_data_path, output_dir):
    for image_name in os.listdir(os.path.join(train_data_path, 'images_empty')):
        with open(os.path.join(output_dir, 'labels', os.path.splitext(image_name)[0] + '.txt'), 'w') as f:
            pass

def copy_images_to_dataset(train_data_path, folder_name, output_dir):
    source_folder = os.path.join(train_data_path, folder_name)  # Папка, откуда копируем файлы
    destination_folder = os.path.join(output_dir, 'images')  # Папка, куда копируем файлы

    # Создаем целевую папку, если ее нет
    os.makedirs(destination_folder, exist_ok=True)

    # Перебираем все файлы в исходной папке и копируем их в целевую
    for file_name in os.listdir(source_folder):
        source_file = os.path.join(source_folder, file_name)
        
        # Копируем только файлы (игнорируем папки)
        if os.path.isfile(source_file):
            shutil.copy(source_file, destination_folder)


# Загрузка данных
train_data_path = '../datasets/dataset_original'
output_dir = '../datasets/dataset_detection'

# Создание папки для сохранения, если она не существует
os.makedirs(output_dir, exist_ok=True)

create_labels_from_csv(train_data_path, output_dir)
create_labels_from_folder(train_data_path, output_dir)
copy_images_to_dataset(train_data_path, 'images', output_dir)
copy_images_to_dataset(train_data_path, 'images_empty', output_dir)

print(f'Файлы сохранены в папке: {output_dir}')