import matplotlib.pyplot as plt
import numpy as np
import cv2
from predict import predict, get_classify_lite, get_img_array
from countries import get_representatives
from model import IMG_WIDTH, IMG_HEIGHT

ANALYZED_FILE_PATH = 'test_images/many_real.jpeg'

classify_lite = get_classify_lite()
TMP_FILE_PATH = 'tmp/tmp.png'

representatives = get_representatives()


def cut_and_stretch_frame(image, points):
    points = np.array(points, np.float32)
    dst_points = np.array([[0, 0], [IMG_WIDTH, 0], [IMG_WIDTH, IMG_HEIGHT], [0, IMG_HEIGHT]], np.float32)
    M = cv2.getPerspectiveTransform(points, dst_points)
    warped_image = cv2.warpPerspective(image, M, (IMG_WIDTH, IMG_HEIGHT))
    cv2.imwrite(TMP_FILE_PATH, cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR))
    return warped_image


image = plt.imread(ANALYZED_FILE_PATH)
fig, (user_ax, predictions_ax) = plt.subplots(2, 3, figsize=(12, 6))
user_ax[0].set_title('First point is left-top corner, then clockwise.')
user_ax[1].axis('off')
user_ax[2].axis('off')
user_ax[1].set_title('Wybrany wycinek')
user_ax[0].imshow(image)

predictions_ax[0].axis('off')
predictions_ax[1].axis('off')
predictions_ax[2].axis('off')


def onclick(event):
    global points
    points.append((event.xdata, event.ydata))
    if len(points) == 4:
        user_ax[0].clear()
        user_ax[0].imshow(image)
        user_ax[0].set_title('First point is left-top corner, then clockwise.')

        interpolated_frame = cut_and_stretch_frame(image, points)

        user_ax[1].imshow(interpolated_frame)
        predictions = predict(classify_lite, get_img_array(TMP_FILE_PATH))

        for i in range(3):
            predictions_ax[i].clear()
            predictions_ax[i].axis('off')
        for prediction_i, prediction in enumerate(predictions.keys()):
            predictions_ax[prediction_i].set_title(f"{prediction}: {predictions[prediction]}%")
            predictions_ax[prediction_i].imshow(representatives[prediction])

        points = []
        plt.draw()
    else:
        user_ax[0].scatter(event.xdata, event.ydata, c='b')
        plt.draw()


points = []
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
