from countries import get_countries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


def get_image(path):
    return np.array([img_to_array(load_img(path, target_size=(256, 256), interpolation="nearest"))])


def predict(img):
    model = load_model('models/model_02.h5')
    countries = sorted(get_countries('flags'))
    predictions = model.predict(img)[0]
    predictions = map(lambda prob: round(prob, 2), predictions)
    countries_and_prediction = list(zip(countries, predictions))
    countries_and_prediction = list(filter(lambda pair: pair[1] > 0.1, countries_and_prediction))
    countries_and_prediction.sort(key=lambda pair: pair[1], reverse=True)
    return countries_and_prediction[:min(len(countries_and_prediction), 5)]


print(predict(get_image('test_images/poland.png')))
