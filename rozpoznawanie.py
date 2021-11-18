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


def learn_digits():
    def load_data(s=8):
        X = np.ones((10160, s*s))
        y = np.zeros(10160)
        for d in digits:
            filenames = os.listdir('digits/' + str(d))
            L = len(filenames)
            for i in range(L):
                name = filenames[i]
                y[i+d*1016] = d
                # This returns an image object
                img = Image.open("digits/" + str(d) + "/" + name)
                img = img.resize((s, s))
                img = np.asarray(img)  # convert it to ndarray
                img = img.reshape(-1, img.size)
                X[i+d*1016, :] = img
        return X, y

    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    data = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        data[0], data[1], test_size=0.33, random_state=42)

    def classify_digits(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)

        count = 0
        for i in range(len(X_test)):
            if p[i] == y_test[i]:
                count += 1
        print('Dokładność modelu: ', count/len(X_test))
        return clf

    return classify_digits()


def load_image(name, s):
    img = Image.open(name)
    img = img.resize((s, s))
    img = np.asarray(img)
    img = img.reshape(-1, img.size)
    return img


class Digit_prediction():
    def __init__(self, model):
        self.clf = model

    def predict(self, img):
        return self.clf.predict(img)


if __name__ == '__main__':
    digit_predict = Digit_prediction(model=learn_digits())
    img = load_image('img.png', 8)

    print(digit_predict.predict(img))
