import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


dataset: tf.data.Dataset = image_dataset_from_directory(
    'flags',
    labels='inferred',
    validation_split=0.2,
    subset='training',
    seed=123
)

test_dataset: tf.data.Dataset = image_dataset_from_directory(
    'flags',
    labels='inferred',
    validation_split=0.2,
    subset='validation',
    seed=123
)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(dataset, epochs=10)

test_loss, test_acc = model.evaluate(test_dataset)

model.save('models/model_02.h5')

