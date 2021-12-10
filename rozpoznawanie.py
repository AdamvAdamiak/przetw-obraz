from time import time
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
        X = np.array(X)
        y = np.array(y)
        return X, y

    # X_train, y_train = load_data('letters/emnist-balanced-train.csv')
    # np.save('X_train', X_train)
    # np.save('y_train', y_train)
    # X_test, y_test = load_data('letters/emnist-balanced-test.csv')
    # np.save('X_test', X_test)
    # np.save('y_test', y_test)

    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    def classify_digits(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
        clf = KNeighborsClassifier(n_neighbors=3)  # DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)

        count = 0
        for i in range(len(X_test)):
            if p[i] == y_test[i]:
                count += 1

        print('Dokładność modelu: ', count/len(X_test))
        return clf

    clf = classify_digits()
    # analyse(clf, X_test, X_train, y_test, y_train)
    return clf


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

    # X_train, y_train = load_data('letters/emnist-balanced-train.csv')
    # np.save('X_trainl', X_train)
    # np.save('y_trainl', y_train)
    # X_test, y_test = load_data('letters/emnist-balanced-test.csv')
    # np.save('X_testl', X_test)
    # np.save('y_testl', y_test)

    X_train = np.load('X_trainl.npy')
    y_train = np.load('y_trainl.npy')
    X_test = np.load('X_testl.npy')
    y_test = np.load('y_testl.npy')

    def classify_letters(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
        clf = MLPClassifier()
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)

        count = 0
        for i in range(len(X_test)):
            if p[i] == y_test[i]:
                count += 1

        print('Dokładność modelu: ', count/len(X_test))
        return clf

    clf = classify_letters()
    # analyse(clf,X_test,X_train,y_test,y_train)
    print('x')
    return clf


def load_image(name, s):
    # print(s)
    img = Image.open(name).convert('L')
    img = img.resize((s, s))
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.transpose(Image.ROTATE_270)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = np.asarray(img)
    img = img.reshape(-1, s*s)
    return img


def grayscale_inversion(img): #get img as np.array with shape (1,784)
    img_new = []
    for i in range(784):
        img_new.append(255 - img[0,i])
    img_new = np.array(img_new)
    img_new = img_new.reshape(-1, 784)
    return img_new


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


def analyse(model, X_test, X_train, y_test, y_train):
    plot_confusion_matrix(model, X=X_test, y_true=y_test)
    plt.show()


if __name__ == '__main__':
    start = time()
    digit_predict = Digit_prediction(model=learn_digits())
    img = load_image('test_digit/img_test.png', 28)
    img = grayscale_inversion(img)
    print(digit_predict.predict(img))

    
    # for i in range(10):
    #     path = f'test_digit/img{i}.png'
    #     print(path)
    #     digit_predict = Digit_prediction(model=learn_digits())
    #     img = load_image(path, 28)
    #     print(digit_predict.predict(img))


    # X_test = np.load('X_test.npy')
    # y_test = np.load('y_test.npy')

    # for n in range(100):
    #     img = X_test[n]
    #     print(y_test[n])
    #     img = img.reshape((28,28))
    #     plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    #     plt.show()
    # letter_predict = Letter_prediction(model=learn_letters())
    # img = load_image('img3.png', 28)
    # prediction = letter_predict.predict(img)
    # print(id_toletter(prediction))

    # X_test = np.load('X_test.npy')
    # y_test = np.load('y_test.npy')
    # n=9
    # img = X_test[n]
    # print(y_test[n])
    # img = img.reshape((28,28))
    # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    print(round(time()-start, 2), ' s')