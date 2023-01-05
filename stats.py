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
    countries = sorted(get_countries())

    numpy_confusion_matrix = np.array(confusion_matrix)
    diagonal = np.diagonal(numpy_confusion_matrix)

    accuracy = sum(diagonal) / sum(sum(numpy_confusion_matrix))

    recall = []
    for col_i, column in enumerate(numpy_confusion_matrix.T):
        recall.append(diagonal[col_i] / sum(column))

    mean_recall = sum(recall) / len(recall)

    worst_by_recall_i = np.argmin(recall)
    worst_by_recall = countries[worst_by_recall_i]
    worst_by_recall_confused_most_often_with = countries[np.argsort(numpy_confusion_matrix.T[worst_by_recall_i])[-2]]

    precision = []
    for row_i, row in enumerate(numpy_confusion_matrix):
        precision.append(diagonal[row_i] / sum(row))

    mean_precision = sum(precision) / len(precision)

    worst_by_precision_i = np.argmin(recall)
    worst_by_precision = countries[worst_by_precision_i]
    worst_by_precision_confused_most_often_with = countries[np.argsort(numpy_confusion_matrix.T[worst_by_precision_i])[-2]]

    return {
        'accuracy': accuracy,
        'mean_recall': mean_recall,
        'worst_by_recall': worst_by_recall,
        'worst_by_recall_confused_most_often_with': worst_by_recall_confused_most_often_with,
        'mean_precision': mean_precision,
        'worst_by_precision': worst_by_precision,
        'worst_by_precision_confused_most_often_with': worst_by_precision_confused_most_often_with,
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
