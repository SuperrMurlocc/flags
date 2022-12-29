import matplotlib.pyplot as plt
import numpy as np
import cv2
from predict import predict, get_classify_lite, get_img_array
from model import IMG_WIDTH, IMG_HEIGHT


classify_lite = get_classify_lite()
TMP_FILE_PATH = 'tmp/tmp.png'


def cut_and_stretch_frame(image, points):
    points = np.array(points, np.float32)
    dst_points = np.array([[0, 0], [IMG_WIDTH, 0], [IMG_WIDTH, IMG_HEIGHT], [0, IMG_HEIGHT]], np.float32)
    M = cv2.getPerspectiveTransform(points, dst_points)
    warped_image = cv2.warpPerspective(image, M, (IMG_WIDTH, IMG_HEIGHT))
    cv2.imwrite(TMP_FILE_PATH,  cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR))
    return warped_image


image = plt.imread('test_images/many.jpeg')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
ax1.set_title('First point is left-top corner, then clockwise.')
ax2.axis('off')
ax1.imshow(image)


def onclick(event):
    global points
    points.append((event.xdata, event.ydata))
    if len(points) == 4:
        ax1.clear()
        ax1.imshow(image)
        ax1.set_title('First point is left-top corner, then clockwise.')

        interpolated_frame = cut_and_stretch_frame(image, points)

        ax2.imshow(interpolated_frame)
        ax2.set_title(predict(classify_lite, get_img_array(TMP_FILE_PATH)))

        points = []
        plt.draw()
    else:
        ax1.scatter(event.xdata, event.ydata, c='b')
        plt.draw()


points = []
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
