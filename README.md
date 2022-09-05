# Повышение эффективности кодирования видеоданных в системах видеоконференцсвязи при помощи моделирования лица человека на основе речевого сигнала

Выпускная квалификационная работа, Университет ИТМО, 2022 год.

# Запуск

1. Скачать dlib shape_predictor (для 5-ти точек) и указать на него путь в main.py
2. Указать путь до папки с видеофайлами в формате .yuv и до аудиофайла
3. Запустить

На выходе будет получена папка с видеозаписями, на которых будет изображено смоделированное лицо.

Примечание: этот код предзначался для оценки оффективности кодирования при использовании различных
параметров квантования одной видеозаписи, поэтому на вход подается несколько разных видеофайлов (которые на самом деле были получены из одного исходного видео)
и один аудиофайл (соответсвующий исходной видеозаписи). Для тестирования можно просто положить в папку один yuv файл и соответсвующую ему аудиозапись.

# Структура

#### Preprocessing
В папке расположен весь код,
необходимый для предварительной обработки
датасета The Grid. Пример полученного датасета см. в `/dataset_small`

#### Training
В папке расположен весь код,
необходимый для обучения нейросети (код перекопирован из Colab, лучше это делать там).

#### Predicting
В папке расположен исходный код генератора, посволяющий по 
видеокадрам и аудиоданным смоделировать кадры и собрать и в видеозапись

Генератор принимает путь до видеофайла, аудиофайла и частоту ключевых кадров (например каждый 8)
и генерирует новое видео, используя ключевые кадры как опорные.

#### matlab
В папке расположен весь код (на языке matlab), позволяющий 
оценить повышение эффективности кодирования видеоданных.