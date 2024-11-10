from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn

# DETECTION_MODEL_PATH = '/home/dragonnp/projects/python/hackatons/automatic_filtering_of_animal_images/application/api/model/detection_210.pt'
# CLASSIFICATION_MODEL_PATH = '/home/dragonnp/projects/python/hackatons/automatic_filtering_of_animal_images/train_model/testing_model/metrica/classification_model.pth'

DETECTION_MODEL_PATH = '/home/dragonnp/projects/python/hackatons/automatic_filtering_of_animal_images/train_model/testing_model/metrica/detection_model.pt'
CLASSIFICATION_MODEL_PATH = '/home/dragonnp/projects/python/hackatons/automatic_filtering_of_animal_images/train_model/testing_model/metrica/animal_classification_model_7.pth'

class ModelDetectionHandler:
    def __init__(self, model_path):
        # Загрузка обученной модели YOLO
        self.model = YOLO(model_path)

    def predict(self, image):        
        # Получение предсказаний модели
        results = self.model.predict(image, conf=0.7)
        
        # Обработка результатов и преобразование в удобный формат
        predictions = []
        for pred in results:
            for box in pred.boxes:
                xc, yc, width, height = box.xywhn[0]  # Извлечение координат рамки
                predictions.append([float(xc), float(yc), float(width), float(height)])
        
        return predictions
    
class ModelClassificationHandler:
    def __init__(self, model_path):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        
        # Загрузка модели
        self.model = models.efficientnet_b7(pretrained=False)
        num_ftrs = self.model.classifier[1].in_features  # Получаем количество входных признаков для линейного слоя
        self.model.classifier[1] = nn.Linear(num_ftrs, 2)  # 2 класса: 0 - Плохой объект, 1 - Хороший объект
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
            _, preds = torch.max(outputs, 1)
        
        class_names = [0, 1]
        predicted_class = class_names[preds.item()]
        
        return predicted_class

class LogicModelHandler:
    def __init__(self):
        self.detector = ModelDetectionHandler(DETECTION_MODEL_PATH)
        self.classificator = ModelClassificationHandler(CLASSIFICATION_MODEL_PATH)
    
    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        draw = ImageDraw.Draw(image)

        result = []
        for bbox in self.detector.predict(image):            
            img_width, img_height = image.size  # Получаем размер изображения

            # Распаковываем относительные значения
            center_x, center_y, width, height = bbox

            # Преобразуем относительные значения в абсолютные пиксельные координаты
            left = (center_x - width / 2) * img_width
            top = (center_y - height / 2) * img_height
            right = (center_x + width / 2) * img_width
            bottom = (center_y + height / 2) * img_height

            # Обрезаем изображение
            cropped_image = image.crop((left, top, right, bottom))
            
            predicted_class = self.classificator.predict(cropped_image)
            
            if predicted_class == 0:
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
            else:
                draw.rectangle([left, top, right, bottom], outline="green", width=3)
            
            result.append([bbox, predicted_class])
            
        return [image, result]
