if __name__ == '__main__':
    # Устройство
    # dodo()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загрузка модели
    model = models.efficientnet_b7(pretrained=False)
    num_ftrs = model.classifier[1].in_features  # Получаем количество входных признаков для линейного слоя
    model.classifier[1] = nn.Linear(num_ftrs, 2)  # 2 класса: 0 - Плохой объект, 1 - Хороший объект
    num = input("Enter epoch: ")
    model.load_state_dict(torch.load(f"animal_classification_model_{num}.pth"))
    model = model.to(device)

    # Преобразования для изображений
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Проверка всех изображений из annotation.csv
    predictions_df = predict_all_images(model, 'annotations_augmented_with_bboxes.csv', test_transforms, device)
    print(predictions_df.head())