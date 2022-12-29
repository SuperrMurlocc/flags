import tensorflow as tf
import numpy as np
from model import IMG_HEIGHT, IMG_WIDTH, TF_MODEL_FILE_PATH
from countries import get_countries


def get_img_array(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    return tf.expand_dims(img_array, 0)  # Create a batch


def get_classify_lite():
    interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
    return interpreter.get_signature_runner('serving_default')


def predict(classify_lite, img_array):
    class_names = sorted(get_countries())
    predictions_lite = classify_lite(sequential_input=img_array)['dense_1']
    score_lite = tf.nn.softmax(predictions_lite)
    return f"{class_names[np.argmax(score_lite)]} : {100 * np.max(score_lite):.2f}%"


if __name__ == '__main__':
    classify_lite = get_classify_lite()
    img_array = get_img_array('tmp/tmp.png')
    print(predict(classify_lite, img_array))
