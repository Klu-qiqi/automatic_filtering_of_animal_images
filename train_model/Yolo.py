from ultralytics import YOLO

dataset_setting = '../datasets/dataset_detection/settings.yaml'
base_model = './yolo11x.pt'

imgsz = 640
epochs = 150
batch = 4
device = 0  #GPU

# Формируем имя для сохранения обученной модели, включающее параметры обучения
model_name = f'{base_model.split(".")[0].split("/")[0]}_imgsz_{imgsz}_epochs_{epochs}_batch_{batch}'

# Загружаем предобученную модель YOLO из указанного файла
model = YOLO(base_model)  

# Запуск процесса обучения
model.train(
    data=dataset_setting,       # Путь к YAML-файлу, содержащему описание датасета
    imgsz=imgsz,                # Размер входных изображений
    epochs=epochs,              # Количество эпох обучения
    batch=batch,                # Размер батча
    save=True,                  # Сохранение весов модели после каждой эпохи
    name=model_name,            # Имя папки, где будут сохраняться результаты
    show_labels=False,          # Отображение меток на изображениях отключено
    device=device,              # Устройство для обучения
    val=True                    # Выполнение валидации после каждой эпохи
)
