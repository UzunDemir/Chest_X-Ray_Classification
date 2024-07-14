# # загрузка библиотек
# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from PIL import Image

# st.sidebar.write("[Uzun Demir](https://uzundemir.github.io/)") #[Github](https://github.com/UzunDemir)     [Linkedin](https://www.linkedin.com/in/uzundemir/)     
# st.sidebar.write("[Github](https://github.com/UzunDemir)")
# st.sidebar.write("[Linkedin](https://www.linkedin.com/in/uzundemir/)")
# st.sidebar.title("Описание проекта")
# st.sidebar.title("Handwritten Digits Classifier MNIST")
# st.sidebar.divider()
# st.sidebar.write(
#         """
                                       
#                      Эта приложение выполнено как самостоятельное исследование в рамках обучения по модулю Computer Vision курса Machine Learning Advanced от Skillbox. 
                     
#                      1. Вначале была обучена модель диагностики по рентгеновским снимкам следующих состояний: COVID-19, пневмония, нормальное состояние легких. 
#                      Я использовал много разных моделей и остановил свой выбор на сверточной нейронной сети (Convolutional Neural Network, CNN)
#                      которая показала точность на тестовом наборе данных: 0.86.
#                      Ноутбук с исследованиями можно посмотреть [здесь.](https://github.com/UzunDemir/Chest_X-Ray_Classification/blob/main/Model.ipynb)
#                      2. Вторым шагом я решил обернуть готовую модель в сервис и запустить её как веб-приложение, которое давало бы предсказания по загруженным рентгенограмам.
#                      3. 
                     
#                      """
#     )



# # загрузка модели
# model_path = "main.tflite"
# interpreter = tf.lite.Interpreter(model_path=model_path)
# interpreter.allocate_tensors()

# # Get input and output details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# input_shape = input_details[0]['shape']
# output_shape = output_details[0]['shape']
# input_dtype = input_details[0]['dtype']
# output_dtype = output_details[0]['dtype']

# # назначение классов предсказания
# class_names = ['Covid', 'Viral Pneumonia', 'Normal']

# st.set_page_config(page_title="Chest X-ray Classifier", layout="wide")
# col1, col2 = st.columns([1, 1])  # Divide the page into two columns

# # загруженное изображение будет слева
# with col1:
#     st.title('Chest X-Ray Classification')
#     st.markdown('<h3 style="font-weight:normal;">Classify Chest X-ray images into COVID-19, Pneumonia, or Normal.</h3>', unsafe_allow_html=True)

#     # загрузка изображения
#     uploaded_file = st.file_uploader("Upload an image and click the 'Predict Now' button.", type=["jpg", "jpeg", "png"])

#     # предсказание
#     if uploaded_file is not None:
#         # загрузка и препроцессинг изображения
#         image = Image.open(uploaded_file)
#         if image.mode != "RGB":
#             image = image.convert("RGB")  # конвертация в RGB 
#         image = image.resize((input_shape[1], input_shape[2]))
#         image = np.array(image, dtype=np.float32)
#         image /= 255.0
#         image = np.expand_dims(image, axis=0)

#         # предсказание
#         def predict(image):
#             interpreter.set_tensor(input_details[0]['index'], image.astype(input_dtype))
#             interpreter.invoke()
#             predictions = interpreter.get_tensor(output_details[0]['index'])
#             predicted_class_index = np.argmax(predictions, axis=1)
#             predicted_class_name = class_names[predicted_class_index[0]]
#             return predicted_class_name

#         # кнопка предсказания
#         if st.button('Predict Now'):
#             predicted_class_name = predict(image)
#             # Display prediction as a heading in bold font
#             st.markdown(f"<h2>Classified as: <span style='font-style: italic; font-weight: bold;'>{predicted_class_name}</span></h2>", unsafe_allow_html=True)

# # отображение загруженного
# with col2:
#     if uploaded_file is not None:
#         st.image(image, caption="Uploaded Image", use_column_width=True)
# Импорт необходимых библиотек
# Импорт необходимых библиотек
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Настройка страницы Streamlit в самом начале скрипта
st.set_page_config(page_title="Chest X-ray Classifier", layout="wide")

# Устанавливаем стиль для центрирования элементов
st.markdown("""
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        /height: 5vh;
        text-align: center;
        flex-direction: column;
        margin-top: 0vh;  /* отступ сверху */
    }
    .github-icon:hover {
        color: #4078c0; /* Изменение цвета при наведении */
    }
    </style>
    <div class="center">
        <img src="https://github.com/UzunDemir/mnist_777/blob/main/200w.gif?raw=true">        
    </div>
    """, unsafe_allow_html=True)
st.divider()

# Заголовок и ссылки в сайдбаре
st.sidebar.write("[Uzun Demir](https://uzundemir.github.io/)")
st.sidebar.write("[Github](https://github.com/UzunDemir)")
st.sidebar.write("[Linkedin](https://www.linkedin.com/in/uzundemir/)")
st.sidebar.title("Описание проекта")
st.sidebar.title("Диагностика по рентгенограммам грудной клетки")
st.sidebar.title('Chest X-Ray Classification')
st.sidebar.divider()
st.sidebar.write(
    """
    Это приложение разработано в рамках самостоятельного исследования по модулю Computer Vision курса Machine Learning Advanced от Skillbox.

    1. Вначале была обучена модель для диагностики по рентгеновским снимкам состояний: COVID-19, пневмония и нормальные легкие.
       Я использовал несколько моделей и выбрал сверточную нейронную сеть (CNN), которая показала точность на тестовых данных: 0.86.
       Ноутбук с исследованиями можно посмотреть [здесь.](https://github.com/UzunDemir/Chest_X-Ray_Classification/blob/main/Model.ipynb)
    
    2. Вторым шагом было обернуть готовую модель в веб-приложение для предсказания по загруженным рентгенограммам.
    """
)

# Загрузка модели TensorFlow Lite
model_path = "main.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Получение информации о входе и выходе модели
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']

# Назначение классов для предсказаний
class_names = ['Covid', 'Viral Pneumonia', 'Normal']

# Деление страницы на две колонки
col1, col2 = st.columns([1, 1])

# Блок с левой стороны, для загрузки изображения
with col1:
    st.title('Диагностика по рентгенограммам грудной клетки')
    st.markdown('<h3 style="font-weight:normal;">Определение COVID-19, пневмонии, или нормального состояния легких.</h3>', unsafe_allow_html=True)

    # Загрузка изображения пользователем
    uploaded_file = st.file_uploader("Загрузите изображение и нажмите кнопку 'Диагностика' ", type=["jpg", "jpeg", "png"])

    # Процесс предсказания
    if uploaded_file is not None:
        # Загрузка и предобработка изображения
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")  # Конвертация в RGB, если необходимо
        image = image.resize((input_shape[1], input_shape[2]))  # Изменение размера до нужного для модели
        image = np.array(image, dtype=np.float32) / 255.0  # Нормализация значений пикселей
        image = np.expand_dims(image, axis=0)  # Добавление измерения пакета

        # Функция для выполнения предсказания
        def predict(image):
            interpreter.set_tensor(input_details[0]['index'], image.astype(input_dtype))  # Установка входных данных
            interpreter.invoke()  # Выполнение предсказания
            predictions = interpreter.get_tensor(output_details[0]['index'])  # Получение результатов предсказания
            predicted_class_index = np.argmax(predictions, axis=1)  # Определение индекса предсказанного класса
            predicted_class_name = class_names[predicted_class_index[0]]  # Определение имени предсказанного класса
            return predicted_class_name

        # Кнопка для выполнения предсказания
        if st.button('Диагностика'):
            predicted_class_name = predict(image)
            # Отображение результата предсказания
            st.markdown(f"<h2>Classified as: <span style='font-style: italic; font-weight: bold;'>{predicted_class_name}</span></h2>", unsafe_allow_html=True)

# Блок с правой стороны для отображения загруженного изображения
with col2:
    if uploaded_file is not None:
        st.image(image, caption="Загруженная рентгенограмма", use_column_width=True)
