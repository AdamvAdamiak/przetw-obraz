import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import sklearn.svm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from random import randint
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn import utils
from matplotlib.pyplot import imread
from PIL import Image
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier


def learn_digits():

    def load_data(path, size=784):

        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        X = []
        y = []
        f = pd.read_csv(path, sep=',')
        data = f.values.tolist()

        for row in data:
            id = int(row.pop(0))
            if id in digits:
                row = np.array(row)
                X.append(row)
                y.append(id)

        return X, y

    X_train, y_train = load_data('letters/emnist-balanced-train.csv')
    X_test, y_test = load_data('letters/emnist-balanced-test.csv')

    def classify_digits(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
        clf = KNeighborsClassifier()  # DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)

        count = 0
        for i in range(len(X_test)):
            if p[i] == y_test[i]:
                count += 1

        print('Dokładność modelu: ', count/len(X_test))
        return clf

    return classify_digits()


def learn_letters():

    def load_data(path, size=784):

        letters_id = range(10, 47)
        X = []
        y = []
        f = pd.read_csv(path, sep=',')
        data = f.values.tolist()

        for row in data:
            id = int(row.pop(0))
            if id in letters_id:
                row = np.array(row)
                X.append(row)
                y.append(id)

        return X, y

    X_train, y_train = load_data('letters/emnist-balanced-train.csv')
    X_test, y_test = load_data('letters/emnist-balanced-test.csv')

    def classify_letters(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)

        count = 0
        for i in range(len(X_test)):
            if p[i] == y_test[i]:
                count += 1

        print('Dokładność modelu: ', count/len(X_test))
        return clf

    return classify_letters()


def load_image(name, s):
    # print(s)
    img = Image.open(name).convert('L')
    img = img.resize((s, s))
    img = img.transpose(Image.ROTATE_270)
    img = np.asarray(img)
    img = img.reshape(-1, s*s)
    # print(s, img.size)
    return img


def id_toletter(id):
    letters = ['', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z',
               'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
    letters_id = range(10, 47)
    output_letters = []
    for i in id:
        for j in range(0, len(letters_id)):
            if i == letters_id[j]:
                output_letters.append(letters[j])
    return output_letters


class Digit_prediction():
    def __init__(self, model):
        self.clf = model

    def predict(self, img):
        return self.clf.predict(img)


class Letter_prediction():
    def __init__(self, model):
        self.clf = model

    def predict(self, img):
        return self.clf.predict(img)


if __name__ == '__main__':
    digit_predict = Digit_prediction(model=learn_digits())
    img = load_image('img3.png', 28)
    print(digit_predict.predict(img))

    # letter_predict = Letter_prediction(model=learn_letters())
    # img = load_image('img4.png', 28)
    # prediction = letter_predict.predict(img)
    # print(prediction)
    # print(id_toletter(prediction))
