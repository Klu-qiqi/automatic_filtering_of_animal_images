import os
from models import LogicModelHandler
from shapely.geometry import box
from PIL import Image
import pandas as pd

PATH_TO_TEST_IMAGES = '/home/dragonnp/projects/python/hackatons/automatic_filtering_of_animal_images/train_model/testing_model/testing_folder'

logic_model = LogicModelHandler()

def getAbsXYWHcenter(norm_coords, image_width, image_height):    
    x_center_abs = float(norm_coords[0]) * image_width
    y_center_abs = float(norm_coords[1]) * image_height
    width_abs = float(norm_coords[2]) * image_width
    height_abs = float(norm_coords[3]) * image_height
    
    min_x = x_center_abs - (width_abs / 2)
    max_x = x_center_abs + (width_abs / 2)
    min_y = y_center_abs - (height_abs / 2)
    max_y = y_center_abs + (height_abs / 2)
    
    return [min_x, min_y, max_x, max_y]

def metricaIOU():
    summa_iou = 0
    count_cls_equals = 0
    count_objects = 0
    
    annotation_data = pd.read_csv(os.path.join('datasets/dataset_original', 'annotation.csv'))
    
    # Перебираем все файлы в исходной папке и копируем их в целевую
    for file_name in os.listdir(PATH_TO_TEST_IMAGES):
        source_image = os.path.join(PATH_TO_TEST_IMAGES, file_name)
            
        for _, row in annotation_data[annotation_data.iloc[:, 0] == file_name].iterrows():
            real_bbox = list(map(float, row.iloc[1].split(',')))
            real_cls = int(row.iloc[2])
            
            predicted_class = 0
            max_iou = 0
                
            with open(source_image, 'rb') as f:
                image_bytes = f.read()
            
            pred = logic_model.predict(image_bytes)
            for result in pred[1]:
                with Image.open(source_image) as img:
                    image_width, image_height = img.size
                    
                pred_coords = getAbsXYWHcenter(result[0], image_width, image_height)
                val_norm_coords = getAbsXYWHcenter(real_bbox, image_width, image_height)
                
                # Создание прямоугольника и многоугольника
                prediction_box = box(*pred_coords)
                label_box = box(*val_norm_coords)
                
                # Вычисление пересечения
                intersection = prediction_box.intersection(label_box).area
                union = prediction_box.area + label_box.area - intersection
                
                if intersection == 0: continue
                
                if (intersection / union) >= max_iou:
                    predicted_class = result[1]
                    max_iou = intersection / union
            
            if real_cls == predicted_class:
                count_cls_equals += 1
            
            summa_iou += max_iou
            count_objects += 1
            
    return [summa_iou / count_objects, count_cls_equals / count_objects]

print(metricaIOU())