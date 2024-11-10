import os
import shutil
import random
from sklearn.model_selection import train_test_split

def get_all_image_files(images_dir, extensions=None):
    """
    Собирает список всех файлов изображений из указанной папки.

    :param images_dir: Путь к папке с изображениями.
    :param extensions: Список допустимых расширений файлов изображений. Если None, будут собраны все файлы.
    :return: Список путей к файлам изображений.
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # Добавьте или уберите расширения по необходимости

    image_files = []
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def split_dataset(
    image_files,
    test_size=0.2,
    val_size=0.2,
    shuffle=True,
    random_seed=42
):
    """
    Разделяет список файлов на train, validation и test выборки.

    :param image_files: Список путей к файлам изображений.
    :param test_size: Доля данных для тестовой выборки.
    :param val_size: Доля тренировочных данных для валидации.
    :param shuffle: Перемешивать данные перед разделением.
    :param random_seed: Случайное семя для воспроизводимости.
    :return: Три списка файлов: train, validation, test.
    """
    if shuffle:
        random.seed(random_seed)
        random.shuffle(image_files)

    # Первичное разделение на train_val и test
    train_val_files, test_files = train_test_split(
        image_files,
        test_size=test_size,
        random_state=random_seed
    )

    # Рассчитываем долю валидации от train_val
    val_ratio = val_size / (1 - test_size)

    # Вторичное разделение на train и validation
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=val_ratio,
        random_state=random_seed
    )

    return train_files, val_files, test_files

def copy_files(image_list, labels_dir, destination_images_dir, destination_labels_dir):
    """
    Копирует файлы изображений и соответствующие метки в целевые директории, сохраняя оригинальные имена.

    :param image_list: Список путей к файлам изображений для копирования.
    :param labels_dir: Путь к папке с метками.
    :param destination_images_dir: Путь к целевой папке для изображений.
    :param destination_labels_dir: Путь к целевой папке для меток.
    """
    if not os.path.exists(destination_images_dir):
        os.makedirs(destination_images_dir)
    if not os.path.exists(destination_labels_dir):
        os.makedirs(destination_labels_dir)

    for img_path in image_list:
        # Копирование изображения
        img_basename = os.path.basename(img_path)
        dest_img_path = os.path.join(destination_images_dir, img_basename)
        try:
            shutil.copy2(img_path, dest_img_path)
        except Exception as e:
            print(f"Ошибка копирования файла {img_path} в {dest_img_path}: {e}")
            continue

        # Обработка соответствующей метки
        label_name = os.path.splitext(img_basename)[0] + '.txt'
        src_label_path = os.path.join(labels_dir, label_name)

        if os.path.exists(src_label_path):
            dest_label_path = os.path.join(destination_labels_dir, label_name)
            try:
                shutil.copy2(src_label_path, dest_label_path)
            except Exception as e:
                print(f"Ошибка копирования метки {src_label_path} в {dest_label_path}: {e}")
        else:
            print(f"Метка для изображения {img_basename} не найдена. Пропускаем копирование метки.")

def split_and_copy_dataset(
    images_dir,
    labels_dir,
    dest_dir,
    test_size=0.2,
    val_size=0.2,
    shuffle=True,
    random_seed=42
):
    """
    Основная функция для разделения и копирования файлов изображений и меток.

    :param images_dir: Путь к исходной папке с изображениями.
    :param labels_dir: Путь к исходной папке с метками.
    :param dest_dir: Путь к папке назначения, где будут созданы train, validation, test.
    :param test_size: Доля данных для тестовой выборки.
    :param val_size: Доля тренировочных данных для валидации.
    :param shuffle: Перемешивать данные перед разделением.
    :param random_seed: Случайное семя для воспроизводимости.
    """
    # Собираем все файлы изображений
    image_files = get_all_image_files(images_dir)
    total_images = len(image_files)
    if total_images == 0:
        print("В папке с изображениями не найдено файлов.")
        return

    print(f"Всего найдено файлов изображений: {total_images}")

    # Разделяем данные
    train_files, val_files, test_files = split_dataset(
        image_files,
        test_size=test_size,
        val_size=val_size,
        shuffle=shuffle,
        random_seed=random_seed
    )

    print(f"Тренировочных изображений: {len(train_files)}")
    print(f"Валидационных изображений: {len(val_files)}")
    print(f"Тестовых изображений: {len(test_files)}")

    # Создаем целевые папки с подкаталогами images и labels
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split_name, split_files in splits.items():
        print(f"\nКопирование и сохранение {split_name} изображений и меток...")
        dest_images_dir = os.path.join(dest_dir, split_name, 'images')
        dest_labels_dir = os.path.join(dest_dir, split_name, 'labels')
        copy_files(
            split_files,
            labels_dir,
            dest_images_dir,
            dest_labels_dir
        )
        print(f"Копирование {split_name} завершено.")

    # Вывод итоговой информации
    for split_name in splits.keys():
        dest_images_dir = os.path.join(dest_dir, split_name, 'images')
        dest_labels_dir = os.path.join(dest_dir, split_name, 'labels')
        num_images = len(os.listdir(dest_images_dir)) if os.path.exists(dest_images_dir) else 0
        num_labels = len(os.listdir(dest_labels_dir)) if os.path.exists(dest_labels_dir) else 0
        print(f"Папка {split_name}: {num_images} изображений, {num_labels} меток")

    print("\nРазделение и копирование данных завершено.")

if __name__ == "__main__":
    # Укажите путь к исходной папке с изображениями
    source_images_dir = "../datasets/dataset_detection/images"  # Замените на ваш путь

    # Укажите путь к исходной папке с метками
    source_labels_dir = "../datasets/dataset_detection/labels"  # Замените на ваш путь

    # Укажите путь к папке назначения, где будут сохранены разделенные данные
    destination_directory = "../datasets/dataset_detection"  # Замените на ваш путь

    # Настройки разделения
    test_size = 0.2    # 20% тестовые данные
    val_size = 0.2     # 20% от train_val (итого 16% от всех данных)
    shuffle = True
    random_seed = 42

    # Вызов основной функции
    split_and_copy_dataset(
        images_dir=source_images_dir,
        labels_dir=source_labels_dir,
        dest_dir=destination_directory,
        test_size=test_size,
        val_size=val_size,
        shuffle=shuffle,
        random_seed=random_seed,
    )
