import tensorflow as tf
import matplotlib.pyplot as plt
import visualkeras

BATCH_SIZE = 32
IMG_WIDTH = 300
IMG_HEIGHT = 200
TF_MODEL_FILE_PATH = 'models/model.tflite'
KERAS_MODEL_FILE_PATH = 'models/model.h5'

if __name__ == '__main__':
    train_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        'flags',
        validation_split=0.2,
        seed=123,
        subset='training',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH)
    )

    class_names = train_ds.class_names

    val_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        'flags',
        validation_split=0.2,
        seed=123,
        subset='validation',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH)
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    layers = tf.keras.layers

    data_augmentation = tf.keras.Sequential([layers.RandomRotation(0.1), layers.RandomZoom(0.1)])

    model = tf.keras.models.Sequential([
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names))
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    epochs = 15
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    model.summary()
    tf.keras.utils.plot_model(model, to_file="models/report/model.png", show_shapes=True)
    visualkeras.layered_view(model, to_file="models/report/model_layers.png", legend=True)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    model.save(KERAS_MODEL_FILE_PATH)
    with open(TF_MODEL_FILE_PATH, 'wb') as f:
        f.write(tflite_model)


