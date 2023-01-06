import tensorflow as tf
from model import KERAS_MODEL_FILE_PATH, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH
from countries import get_countries
from models.report.confusion_matrix import confusion_matrix as cm
import numpy as np


def cross_validation(model) -> float:
    accuracies = []
    for seed in range(50):
        val_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
            'flags',
            validation_split=0.1,
            seed=seed,
            subset='validation',
            batch_size=BATCH_SIZE,
            image_size=(IMG_HEIGHT, IMG_WIDTH)
        )

        accuracies.append(model.evaluate(val_ds)[1])
    return sum(accuracies) / len(accuracies)


def get_confusion_matrix(model):
    confusion_matrix = np.zeros((len(get_countries()), len(get_countries())))
    for seed in range(50):
        val_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
            'flags',
            validation_split=0.2,
            seed=seed,
            subset='validation',
            batch_size=BATCH_SIZE,
            image_size=(IMG_HEIGHT, IMG_WIDTH)
        )

        labels = np.array([])
        predictions = np.array([])
        for x, y in val_ds:
            labels = np.concatenate([labels, y])
            predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=-1)])

        confusion_matrix = confusion_matrix + tf.math.confusion_matrix(labels, predictions)
    return confusion_matrix


def get_stats_from_confusion_matrix(confusion_matrix):
    total_samples = tf.reduce_sum(confusion_matrix)
    true_positives = tf.linalg.trace(confusion_matrix)
    row_sums = tf.reduce_sum(confusion_matrix, axis=1)
    col_sums = tf.reduce_sum(confusion_matrix, axis=0)

    tf_precision = true_positives / col_sums
    tf_recall = true_positives / row_sums
    tf_accuracy = true_positives / total_samples
    tf_f1_score = 2 * (tf_precision * tf_recall) / (tf_precision + tf_recall)

    return {
        'tf_accuracy': tf_accuracy,
        'tf_precision': tf_precision,
        'tf_recall': tf_recall,
        'tf_f1_score': tf_f1_score,
    }


if __name__ == '__main__':
    model = tf.keras.models.load_model(KERAS_MODEL_FILE_PATH)

    cross_validation_accuracy = cross_validation(model)
    with open('models/report/stats.cross_validation_accuracy.txt', 'w') as file:
        file.write(str(cross_validation_accuracy))

    print(cross_validation_accuracy)

    stats_from_confusion_matrix = '\n'.join([f"{k}: {v}" for k, v in get_stats_from_confusion_matrix(cm).items()])
    with open('models/report/stats.confusion_matrix.txt', 'w') as file:
        file.write(stats_from_confusion_matrix)

    print(stats_from_confusion_matrix)
