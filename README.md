# Классификация рентгенограмм грудной клетки
## Целью этого проекта является классификация рентгеновских изображений грудной клетки на три категории: COVID-19, пневмония и норма, с использованием модели сверточной нейронной сети (CNN).

![Projectional_rendering_of_CT_scan_of_thorax_(thumbnail)](https://github.com/user-attachments/assets/99b47383-fff7-4741-aead-bf985f5512d6)



* Изображения для тестирования можно взять [отсюда](https://github.com/UzunDemir/Chest_X-Ray_Classification/tree/main/Test_images)

## Данные для обучения модели
Набор данных, используемый для этого проекта, состоит из коллекции рентгеновских изображений грудной клетки, полученных из [Набора данных рентгеновских изображений грудной клетки (Kaggle)](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset), включая случаи с положительным результатом на COVID-19, случаи пневмонии и нормальные случаи.

## Архитектура модели
* Сверточный уровень 1: 16 фильтров, ядро ​​3x3, активация ReLU, за которым следует слой MaxPooling2D (размер пула 2x2).
* Сверточный уровень 2: 64 фильтра, ядро ​​3x3, активация ReLU, заполнение установлено на «тот же», за которым следует слой MaxPooling2D (размер пула 2x2).
* Выпадающий слой (0,25) добавлен после сверточного слоя 2.
* Сверточный уровень 3: 128 фильтров, ядро ​​3x3, активация ReLU, заполнение установлено на «тот же», за которым следует слой MaxPooling2D (размер пула 2x2).
* Слой Dropout (0.3) добавлен после сверточного слоя 3.
* Сверточный уровень 4: 128 фильтров, ядро ​​3x3, активация ReLU, заполнение установлено на «тот же», за которым следует слой MaxPooling2D (размер пула 2x2).
* Слой Dropout (0.4) добавлен после сверточного слоя 4.
* Выходные данные сверточных слоев выравниваются с помощью слоя Flatten.
* Первый плотный слой: 128 нейронов, активация ReLU.
* Слой Dropout (0,25) добавлен после первого плотного слоя.
* Второй плотный слой: 64 нейрона, активация ReLU.
* Выходной слой: 3 нейрона (по одному на каждый класс), активация softmax для многоклассовой классификации.

## Данные обучения модели
![merge_from_ofoct](https://user-images.githubusercontent.com/97530517/231857828-bfd7ce92-2b2c-456f-a339-534a87d8da69.jpg)
![classification](https://user-images.githubusercontent.com/97530517/231857615-47340376-1d2d-4918-b2e6-f141b56273ce.PNG)
![confusion_matrix](https://user-images.githubusercontent.com/97530517/231857703-a2c9aac9-f217-4095-b63d-6145e7b95de8.PNG)

## Пример предсказания
![predict](https://user-images.githubusercontent.com/97530517/231856798-74574e8d-fb31-45b0-a681-b9579900924d.jpg)

## Модель развернута на streamlit и доступна по адресу: [https://chest-x-ray.streamlit.app/](https://chest-x-ray.streamlit.app/)



