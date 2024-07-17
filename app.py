# # Импорт необходимых библиотек
# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from PIL import Image

# # Настройка страницы Streamlit в самом начале скрипта
# st.set_page_config(page_title="Chest X-ray Classifier", layout="wide")

# # Устанавливаем стиль для центрирования элементов
# st.markdown("""
#     <style>
#     .center {
#         display: flex;
#         justify-content: center;
#         align-items: center;
#         /height: 5vh;
#         text-align: center;
#         flex-direction: column;
#         margin-top: 0vh;  /* отступ сверху */
#     }
#     .github-icon:hover {
#         color: #4078c0; /* Изменение цвета при наведении */
#     }
#     </style>
#     <div class="center">
#         <img src="https://github.com/UzunDemir/Chest_X-Ray_Classification/blob/main/Projectional_rendering_of_CT_scan_of_thorax_(thumbnail).gif?raw=true">        
#     </div>
#     """, unsafe_allow_html=True)
# st.divider()

# # Заголовок и ссылки в сайдбаре
# st.sidebar.write("[Uzun Demir](https://uzundemir.github.io/)")
# st.sidebar.write("[Github](https://github.com/UzunDemir)")
# st.sidebar.write("[Linkedin](https://www.linkedin.com/in/uzundemir/)")
# st.sidebar.title("Описание проекта")
# st.sidebar.title("Диагностика по рентгенограммам грудной клетки")
# st.sidebar.title('Chest X-Ray Classification')
# st.sidebar.divider()
# st.sidebar.write(
#     """
#     Это приложение разработано в рамках самостоятельного исследования по модулю Computer Vision курса Machine Learning Advanced от Skillbox.

#     1. Вначале была обучена модель для диагностики по рентгеновским снимкам состояний: COVID-19, пневмония и нормальные легкие.
#        Я использовал несколько моделей и выбрал сверточную нейронную сеть (CNN), которая показала точность на тестовых данных: 0.86.
#        Ноутбук с исследованиями можно посмотреть [здесь.](https://github.com/UzunDemir/Chest_X-Ray_Classification/blob/main/Model.ipynb)
    
#     2. Вторым шагом было обернуть готовую модель в веб-приложение для предсказания по загруженным рентгенограммам.
#     """
# )

# # Загрузка модели TensorFlow Lite
# model_path = "main.tflite"
# interpreter = tf.lite.Interpreter(model_path=model_path)
# interpreter.allocate_tensors()

# # Получение информации о входе и выходе модели
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# input_shape = input_details[0]['shape']
# output_shape = output_details[0]['shape']
# input_dtype = input_details[0]['dtype']
# output_dtype = output_details[0]['dtype']

# # Назначение классов для предсказаний
# class_names = ['Covid', 'Viral Pneumonia', 'Normal']

# # Деление страницы на две колонки
# col1, col2 = st.columns([1, 1])

# # Блок с левой стороны, для загрузки изображения
# with col1:
#     st.title('Диагностика по рентгенограммам грудной клетки')
#     st.markdown('<h3 style="font-weight:normal;">Определение COVID-19, пневмонии, или нормального состояния легких.</h3>', unsafe_allow_html=True)

#     # Загрузка изображения пользователем
#     uploaded_file = st.file_uploader("Загрузите изображение и нажмите кнопку 'Диагностика' ", type=["jpg", "jpeg", "png"])

#     # Процесс предсказания
#     if uploaded_file is not None:
#         # Загрузка и предобработка изображения
#         image = Image.open(uploaded_file)
#         if image.mode != "RGB":
#             image = image.convert("RGB")  # Конвертация в RGB, если необходимо
#         image = image.resize((input_shape[1], input_shape[2]))  # Изменение размера до нужного для модели
#         image = np.array(image, dtype=np.float32) / 255.0  # Нормализация значений пикселей
#         image = np.expand_dims(image, axis=0)  # Добавление измерения пакета

#         # Функция для выполнения предсказания
#         def predict(image):
#             interpreter.set_tensor(input_details[0]['index'], image.astype(input_dtype))  # Установка входных данных
#             interpreter.invoke()  # Выполнение предсказания
#             predictions = interpreter.get_tensor(output_details[0]['index'])  # Получение результатов предсказания
#             predicted_class_index = np.argmax(predictions, axis=1)  # Определение индекса предсказанного класса
#             predicted_class_name = class_names[predicted_class_index[0]]  # Определение имени предсказанного класса
#             return predicted_class_name

#         # Кнопка для выполнения предсказания
#         if st.button('Диагностика'):
#             predicted_class_name = predict(image)
#             # Отображение результата предсказания
#             st.markdown(f"<h2>Classified as: <span style='font-style: italic; font-weight: bold;'>{predicted_class_name}</span></h2>", unsafe_allow_html=True)

# # Блок с правой стороны для отображения загруженного изображения
# with col2:
#     if uploaded_file is not None:
#         st.image(image, caption="Загруженная рентгенограмма", use_column_width=True)


# Импорт необходимых библиотек
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO

# Настройка страницы Streamlit в самом начале скрипта
st.set_page_config(page_title="Chest X-ray Classifier", layout="wide")

hide_github_icon = """
<style>
.css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK { 
    display: none !important; 
}
#MainMenu, footer, header { 
    display: none !important; 
}
</style>
"""

st.markdown(hide_github_icon, unsafe_allow_html=True)


st.divider()

# Заголовок и ссылки в сайдбаре
st.sidebar.divider()
st.sidebar.write("[Uzun Demir](https://uzundemir.github.io/)")
st.sidebar.write("[Github](https://github.com/UzunDemir/Chest_X-Ray_Classification)")
st.sidebar.write("[Linkedin](https://www.linkedin.com/in/uzundemir/)")
st.sidebar.title("Описание проекта")
st.sidebar.title('Chest X-Ray Classification')
st.sidebar.markdown('<img src="https://github.com/UzunDemir/Chest_X-Ray_Classification/blob/main/Projectional_rendering_of_CT_scan_of_thorax_(thumbnail).gif?raw=true" width="100%">', unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.write(
    """
    Это приложение разработано в рамках самостоятельного исследования по модулю Computer Vision курса Machine Learning Advanced от Skillbox.

    1. Вначале была обучена модель для диагностики по рентгеновским снимкам состояний: COVID-19, пневмония и нормальные легкие.
       Я использовал несколько моделей и выбрал сверточную нейронную сеть (CNN), которая показала точность на тестовых данных: 0.86.
       Ноутбук с исследованиями можно посмотреть [здесь.](https://github.com/UzunDemir/Chest_X-Ray_Classification/blob/main/Model.ipynb)
       Набор данных, используемый для этого проекта, состоит из коллекции рентгеновских изображений грудной клетки, полученных из 
       [Набора данных рентгеновских изображений грудной клетки (Kaggle)](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset), 
       включая случаи с положительным результатом на COVID-19, случаи пневмонии и нормальные случаи.
    2. Вторым шагом было обернуть готовую модель в веб-приложение для предсказания по загруженным рентгенограммам.
    3. Третьим шагом стало добавление возможности загрузки изображения по url.

    Модель предсказания будет в дальнейшем совершенствоваться и дообучаться на новых данных.  
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
    st.title('Диагностика грудной клетки по рентгенограммам ')
    st.markdown('<h3 style="font-weight:normal;">Определение COVID-19, пневмонии, или нормального состояния легких.</h3>', unsafe_allow_html=True)
    

    # Выбор между загрузкой файла и вводом URL
    option = st.radio("Выберите метод загрузки изображения", ("Загрузить файл", "Ввести URL"))

    if option == "Загрузить файл":
        # Загрузка изображения пользователем
        uploaded_file = st.file_uploader("Загрузите изображение и нажмите кнопку 'Диагностика'", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            if image.mode != "RGB":
                image = image.convert("RGB")  # Конвертация в RGB, если необходимо
            image = image.resize((input_shape[1], input_shape[2]))  # Изменение размера до нужного для модели
            image = np.array(image, dtype=np.float32) / 255.0  # Нормализация значений пикселей
            image = np.expand_dims(image, axis=0)  # Добавление измерения пакета
            st.session_state['image'] = image
            st.session_state['image_source'] = "upload"
            st.session_state['uploaded_file'] = uploaded_file

    elif option == "Ввести URL":
        # Поле для ввода URL изображения
        url = st.text_input("Введите URL изображения и нажмите кнопку 'Загрузить изображение'")
        st.markdown('<h5 style="font-weight:normal; color:blue; font-style:italic;">Изображения для тестирования можно взять <a href="https://github.com/UzunDemir/Chest_X-Ray_Classification/tree/main/Test_images" style="color:blue;">отсюда</a></h5>', unsafe_allow_html=True)
        # Кнопка для загрузки изображения по URL
        if st.button('Загрузить изображение'):
            if url:
                try:
                    response = requests.get(url)
                    image = Image.open(BytesIO(response.content))
                    if image.mode != "RGB":
                        image = image.convert("RGB")  # Конвертация в RGB, если необходимо
                    image = image.resize((input_shape[1], input_shape[2]))  # Изменение размера до нужного для модели
                    image = np.array(image, dtype=np.float32) / 255.0  # Нормализация значений пикселей
                    image = np.expand_dims(image, axis=0)  # Добавление измерения пакета
                    st.session_state['image'] = image  # Сохранение изображения в session_state
                    st.session_state['image_source'] = "url"
                    st.session_state['image_url_response'] = response.content
                except Exception as e:
                    st.error(f"Не удалось загрузить изображение по указанному URL: {e}")
            else:
                st.error("Пожалуйста, введите URL изображения.")

        #     # Кнопка для очистки поля ввода URL
        # if st.button('Очистить'):
        #     st.experimental_rerun()

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
        if 'image' in st.session_state:
            predicted_class_name = predict(st.session_state['image'])
            # Отображение результата предсказания
            st.markdown(f"<h2>Classified as: <span style='font-style: italic; font-weight: bold;'>{predicted_class_name}</span></h2>", unsafe_allow_html=True)
        else:
            st.error("Пожалуйста, загрузите изображение для диагностики.")

# Блок с правой стороны для отображения загруженного изображения
with col2:
    if 'image' in st.session_state:
        if st.session_state['image_source'] == "upload":
            st.image(Image.open(st.session_state['uploaded_file']), caption="Загруженная рентгенограмма", use_column_width=True)
        elif st.session_state['image_source'] == "url":
            st.image(Image.open(BytesIO(st.session_state['image_url_response'])), caption="Загруженная рентгенограмма по URL", use_column_width=True)


