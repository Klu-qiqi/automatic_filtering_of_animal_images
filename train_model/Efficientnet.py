import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Класс Dataset для обработки данных
class AnimalDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_path = self.annotations.iloc[idx]['image']
        image = Image.open(image_path).convert('RGB')
        label = int(self.annotations.iloc[idx]['label'])
        labels = torch.tensor([label], dtype=torch.int64)

        target = {}
        target['labels'] = labels

        if self.transform:
            image = self.transform(image)

        return image, label  # Возвращаем label для классификации

# Функция для обучения и валидации модели.
def train_model(model, train_dataset, val_dataset, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    
    # История метрик для каждой эпохи
    train_acc_history = []
    train_precision_history = []
    train_recall_history = []
    val_acc_history = []
    val_precision_history = []
    val_recall_history = []
    train_los = []  # История потерь на обучении
    val_los = []    # История потерь на валидации

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Обучение
        model.train()  # Устанавливаем режим обучения
        running_loss = 0.0
        running_corrects = 0
        all_preds = []  # Предсказания модели
        all_labels = []  # Истинные метки

        for inputs, labels in tqdm(train_loader):  # Итерация по обучающим данным
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Обнуляем градиенты

            outputs = model(inputs)  # Прямой проход
            _, preds = torch.max(outputs, 1)  # Получаем предсказания
            loss = criterion(outputs, labels)  # Вычисляем потери

            loss.backward()  # Обратное распространение ошибки
            optimizer.step()  # Обновление параметров модели

            # Суммируем потери и корректные предсказания
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Сохраняем предсказания и метки
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        # Рассчитываем метрики для обучения
        epoch_loss = running_loss / len(train_dataset)  # Средние потери за эпоху
        train_los.append(epoch_loss)
        epoch_acc = running_corrects.double() / len(train_dataset)  # Точность
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        train_precision = precision_score(all_labels, all_preds, average='weighted')  # Точность
        train_recall = recall_score(all_labels, all_preds, average='weighted')  # Полнота
        
        # Сохраняем метрики в историю
        train_acc_history.append(epoch_acc.cpu())
        train_precision_history.append(train_precision)
        train_recall_history.append(train_recall)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {train_precision:.4f} Recall: {train_recall:.4f}')
        
        # Валидация
        model.eval()  # Устанавливаем режим валидации
        val_running_loss = 0.0
        val_running_corrects = 0
        val_all_preds = []
        val_all_labels = []

        with torch.no_grad():  # Отключаем расчет градиентов для валидации
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)  # Прямой проход
                _, preds = torch.max(outputs, 1)  # Получаем предсказания
                loss = criterion(outputs, labels)  # Вычисляем потери

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

                val_all_preds.append(preds.cpu())
                val_all_labels.append(labels.cpu())

        # Рассчитываем метрики для валидации
        val_loss = val_running_loss / len(val_dataset)
        val_los.append(val_loss)
        val_acc = val_running_corrects.double() / len(val_dataset)
        val_all_preds = torch.cat(val_all_preds)
        val_all_labels = torch.cat(val_all_labels)
        val_precision = precision_score(val_all_labels, val_all_preds, average='weighted')
        val_recall = recall_score(val_all_labels, val_all_preds, average='weighted')

        # Сохраняем метрики в историю
        val_acc_history.append(val_acc.cpu())
        val_precision_history.append(val_precision)
        val_recall_history.append(val_recall)

        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Precision: {val_precision:.4f} Recall: {val_recall:.4f}')
        
        # Сохраняем модель после каждой эпохи
        torch.save(model.state_dict(), f'model_{epoch}.pth')

    # Сохраняем финальную модель
    torch.save(model.state_dict(), 'model_end.pth')

    # *** Построение графиков метрик ***
    epochs = range(1, num_epochs + 1)

    # График точности
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_acc_history, label='Train Accuracy')
    plt.plot(epochs, val_acc_history, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # График precision
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_precision_history, label='Train Precision')
    plt.plot(epochs, val_precision_history, label='Val Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

    # График recall
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_recall_history, label='Train Recall')
    plt.plot(epochs, val_recall_history, label='Val Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()

    # График потерь
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_los, label='Train Loss')
    plt.plot(epochs, val_los, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model

# Функция настройки для обучения модели (с обучением)
def Setting(test_size, random_state, lr, filename):
    
    full_df=pd.read_csv(filename)

    data_transforms = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    # Нормализация с учетом статистик предобученной модели
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Разделение на обучающую и валидационную выборки
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['label'])
    
    train_df.to_csv('train_annotations.csv', index=False)
    val_df.to_csv('val_annotations.csv', index=False)
    
    train_dataset = AnimalDataset('train_annotations.csv', transform=data_transforms)
    val_dataset = AnimalDataset('val_annotations.csv', transform=data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=15)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=15)
    
    # Загрузка предобученной модели EfficientNet-B7
    model = models.efficientnet_b7(pretrained=True)
    
    # Замена последнего классификационного слоя
    num_ftrs = model.classifier[1].in_features  
    model.classifier[1] = nn.Linear(num_ftrs, 2)  # 2 класса: 0 - Плохой объект, 1 - Хороший объект
    
    model = model.to(device)
    
    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model = train_model(model, train_dataset, val_dataset, train_loader, val_loader, criterion, optimizer, num_epochs=15)
    
# Функция для предсказания на одном изображении
def predict_image(model, image_path, transform, device):
    # Загрузка изображения
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    
    # Добавление batch-димензии
    image = image.unsqueeze(0)
    image = image.to(device)
    
    # Предсказание
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    # Распознавание класса
    class_names = ['Bad ', 'Good ']  
    predicted_class = class_names[preds.item()]
    
    return predicted_class, preds.item()

# Функция для предсказания на нескольких изображении
def predict_all_images(model, annotation_file, transform, device):
    # Загрузка аннотаций
    df = pd.read_csv(annotation_file)
    
    # проверка
    if 'image' not in df.columns:
        raise ValueError("must be col 'image'")
    
    # Список для хранения результатов
    predictions = []
    
    # Проход по всем изображениям
    model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            image_path = row['image']
            try:
                class_, preds=predict_image(model, image_path, transform, device)
                predictions.append({'image': image_path, 'predicted_label': preds})
            except Exception as e:
                print(f"error predprocessing {image_path}: {e}")
    
    # Преобразование в DataFrame
    result_df = pd.DataFrame(predictions)
    
    # Сохранение результатов
    result_df.to_csv('predictions.csv', index=False)
    return result_df

if __name__ == '__main__':
    # Устройство
    Setting(0.2, 42, 0.001, 'annotations_augmented_with_bboxes.csv')

    # Загрузка модели
    model = models.efficientnet_b7(pretrained=False)
    num_ftrs = model.classifier[1].in_features  # Получаем количество входных признаков для линейного слоя
    model.classifier[1] = nn.Linear(num_ftrs, 2)  # 2 класса: 0 - Плохой объект, 1 - Хороший объект
    num = input("Enter epoch: ")
    model.load_state_dict(torch.load(f"model_{num}.pth"))
    model = model.to(device)

    # Преобразования для изображений
    test_transforms = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Проверка всех изображений из annotation.csv
    predictions_df = predict_all_images(model, 'annotations_augmented_with_bboxes.csv', test_transforms, device)
    print(predictions_df.head())

