import matplotlib.pyplot as plt
from collections import Counter
from pickle import load
import imutils
from rozpoznawanie import Digit_predict, grayscale_inversion
from PIL import Image
import numpy as np
import image_slicer
from numpy.lib.stride_tricks import sliding_window_view
import argparse
import time
import cv2
import warnings
warnings.filterwarnings('ignore')


def load_image(img):
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.transpose(Image.ROTATE_270)
    img = np.asarray(img)
    img = img.reshape(-1, 28*28)
    img = grayscale_inversion(img)
    return img


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def predict_image(name):
    result = ''
    image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    (winW, winH) = (28, 28)
    for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        try:
            if y != previous_y:
                result += '\n'
        except NameError:
            previous_y = y

        window = Image.fromarray(window)
        window = window.resize(size=(28, 28))
        window = load_image(window)
        window_list = list(window[0])
        counts = window_list.count(0)
        if counts < 700:
            # window = np.array(window)
            # window = window.reshape(1,784)
            result += str(Digit_predict(window))
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow('show', clone)
        cv2.waitKey(1)
        time.sleep(0.025)
        previous_y = y
    return result


# img = load_image("test_digit/img0.png")

print(predict_image('img0.png'))

# print(Digit_predict(img))
