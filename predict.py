import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
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
    np_config.enable_numpy_behavior()
    pairs = sorted(list(filter(lambda pair: pair[1] >= 0.2, zip(class_names, score_lite.tolist()[0]))), key=lambda pair: pair[1], reverse=True)
    pairs = pairs[:min(3, len(pairs))]
    scores = {class_name: round(100 * score, 2) for class_name, score in pairs}
    return scores


if __name__ == '__main__':
    classify_lite = get_classify_lite()
    img_array = get_img_array('test_images/real_poland.jpeg')
    print(predict(classify_lite, img_array))
