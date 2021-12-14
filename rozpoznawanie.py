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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn import utils
from matplotlib.pyplot import imread
from PIL import Image
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessClassifier



def learn_digits(hyp_param):

    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    def create_valid(X, y, size=100):

        X_valid = []
        y_valid = []
        indices_to_delete = []
        num_list = [size]*10

        for i in range(y.size):
            digit = y[i]
            if num_list[digit] != 0:
                X_valid.append(X[i])
                y_valid.append(y[i])
                num_list[digit] -= 1
                indices_to_delete.append(i)

            if num_list == [0]*10:
                break

        X_valid = np.array(X_valid)
        y_valid = np.array(y_valid)

        di = 0
        for i in indices_to_delete:
            i -= di
            X = np.delete(X, i, 0)
            y = np.delete(y, i, 0)
            di += 1

        return X_valid, y_valid, X, y

    X_valid, y_valid, X_train, y_train = create_valid(X_train, y_train)

    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_valid.npy', X_valid)
    np.save('y_valid.npy', y_valid)

    # X_valid = np.load('X_valid.npy')
    # y_valid = np.load('y_valid.npy')

    def classify_digits(X_train=X_train, X_test=X_valid, y_train=y_train, y_test=y_valid):
    #def classify_digits(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
        clf = MLPClassifier(hidden_layer_sizes=(hyp_param))  #KNeighborsClassifier(n_neighbors=5)   # #DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)

        count = 0
        for i in range(len(X_test)):
            if p[i] == y_test[i]:
                count += 1

        print('Dokładność modelu: ', round(count/len(X_test), 2))
        return clf

    clf = classify_digits()
    # analyse(clf, X_test, X_train, y_test, y_train)
    return clf


def learn_letters(hyp_param):

    X_train = np.load('X_trainl.npy')
    y_train = np.load('y_trainl.npy')
    X_test = np.load('X_testl.npy')
    y_test = np.load('y_testl.npy')

    def create_valid(X, y, size=100):

        X_valid = []
        y_valid = []
        indices_to_delete = []
        num_list = [size] * 26

        for i in range(y.size):
            index = y[i] - 10
            if num_list[index] != 0:
                X_valid.append(X[i])
                y_valid.append(y[i])
                num_list[index] -= 1
                indices_to_delete.append(i)

            if num_list == [0] * 10:
                break

        X_valid = np.array(X_valid)
        y_valid = np.array(y_valid)

        di = 0
        for i in indices_to_delete:
            i -= di
            X = np.delete(X, i, 0)
            y = np.delete(y, i, 0)
            di += 1

        return X_valid, y_valid, X, y

    X_valid, y_valid, X_train, y_train =create_valid(X_train, y_train)

    np.save('X_trainl.npy', X_train)
    np.save('y_trainl.npy', y_train)
    np.save('X_validl.npy', X_valid)
    np.save('y_validl.npy', y_valid)

    # X_valid = np.load('X_validl.npy')
    # y_valid = np.load('y_validl.npy')

    def classify_letters(X_train=X_train, X_test=X_valid, y_train=y_train, y_test=y_valid):
    #def classify_letters(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
        clf = MLPClassifier(hidden_layer_sizes=hyp_param)  #KNeighborsClassifier(n_neighbors=8), alpha=1, max_iter=1000)
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)

        count = 0
        for i in range(len(X_test)):
            if p[i] == y_test[i]:
                count += 1

        print('Dokładność modelu: ', round(count/len(X_test), 2))
        return clf

    clf = classify_letters()
    # analyse(clf,X_test,X_train,y_test,y_train)
    return clf


def load_image(name, s):
    img = Image.open(name).convert('L')
    img = img.resize((s, s))
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.transpose(Image.ROTATE_270)
    img = np.asarray(img)
    img = img.reshape(-1, s*s)
    img = grayscale_inversion(img)
    return img


def grayscale_inversion(img):  # get img as np.array with shape (1,784)
    img_new = []
    for i in range(784):
        img_new.append(255 - img[0, i])
    img_new = np.array(img_new)
    img_new = img_new.reshape(-1, 784)
    return img_new


def id_toletter(id):
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z']
    letters_id = range(10, 47)

    for i in id:
        for j in range(0, len(letters_id)):
            if i == letters_id[j]:
                output_letter = letters[j]
    return output_letter


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


def test_classifiers():
    start = time()

    digit_predict = Digit_prediction(model=learn_digits((100,)))
    # for i in range(10):
    #     path = f'test_digit/img{i}.png'
    #     print(path)
    #     img = load_image(path, 28)
    #     print(digit_predict.predict(img))

    letter_predict = Letter_prediction(model=learn_letters((260,)))
    # for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
    #     img = load_image('test_letters/img{}.png'.format(letter), 28)
    #     prediction = letter_predict.predict(img)
    #     if letter == id_toletter(prediction):
    #         print(letter, ' success')
    #     else:
    #         print(letter, ' classified as ', id_toletter(prediction))

    print(round(time()-start, 2), ' s')



if __name__ == '__main__':
    test_classifiers()