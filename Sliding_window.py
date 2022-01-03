from rozpoznawanie import Digit_predict, grayscale_inversion
from PIL import Image
import numpy as np


def load_image(name):
    img = Image.open(name).convert('L')
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.transpose(Image.ROTATE_270)
    img = np.asarray(img)
    img = img.reshape(-1, 82*28)
    img = grayscale_inversion(img)
    return img

def sliding_window(image, stepSize, windowSize):
  for y in range(0, image.shape[0], stepSize):
    for x in range(0, image.shape[1], stepSize):
      yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

img = load_image("img0.png")

windows = sliding_window(img, 4, (5,5))
for window in windows:
    print(Digit_predict(window))