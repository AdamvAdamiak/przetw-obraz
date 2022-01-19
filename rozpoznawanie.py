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
import pickle


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

    X_valid = np.load('X_valid.npy')
    y_valid = np.load('y_valid.npy')

    def classify_digits(X_train=X_train, X_test=X_valid, y_train=y_train, y_test=y_valid):
        clf = MLPClassifier(hidden_layer_sizes=(hyp_param))
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)

        count = 0
        for i in range(len(X_test)):
            if p[i] == y_test[i]:
                count += 1

        print('Dokładność modelu: ', round(count/len(X_test), 2))
        return clf

    clf = classify_digits()
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

    X_valid = np.load('X_validl.npy')
    y_valid = np.load('y_validl.npy')

    def classify_letters(X_train=X_train, X_test=X_valid, y_train=y_train, y_test=y_valid):
        clf = MLPClassifier(hidden_layer_sizes=hyp_param)
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)

        count = 0
        for i in range(len(X_test)):
            if p[i] == y_test[i]:
                count += 1

        print('Dokładność modelu: ', round(count/len(X_test), 2))
        return clf

    clf = classify_letters()
    return clf


def load_image(name, s=28):
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


def learn_objects(hyp_param=(100,)):

    X_train = np.load('X_traino.npy')
    y_train = np.load('y_traino.npy')
    X_test = np.load('X_testo.npy')
    y_test = np.load('y_testo.npy')

    def classify_objects(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
        clf = MLPClassifier(hidden_layer_sizes=(hyp_param))
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)

        count = 0
        for i in range(len(X_test)):
            if p[i] == y_test[i]:
                count += 1

        print('Dokładność modelu: ', round(count/len(X_test), 2))
        return clf

    clf = classify_objects()
    return clf


class Digit_prediction():

    def predict(self, img):
        return self.clf.predict(img)[0], max(self.clf.predict_proba(img)[0])

    def save_model(self):
        pickle.dump(self.clf, open('digit_classifier.sav', 'wb'))

    def load_model(self):
        self.clf = pickle.load(open('digit_classifier.sav', 'rb'))


class Letter_prediction():

    def predict(self, img):
        return self.clf.predict(img)[0], max(self.clf.predict_proba(img)[0])

    def save_model(self):
        pickle.dump(self.clf, open('letter_classifier.sav', 'wb'))

    def load_model(self):
        self.clf = pickle.load(open('letter_classifier.sav', 'rb'))


class Object_prediction():

    def __init__(self, model):
        self.clf = model

    def predict(self, img):
        return self.clf.predict(img)[0], max(self.clf.predict_proba(img)[0])

    def save_model(self):
        pickle.dump(self.clf, open('object_classifier.sav', 'wb'))

    def load_model(self):
        self.clf = pickle.load(open('object_classifier.sav', 'rb'))


def Digit_predict(img):
    D_classifier = Digit_prediction()
    D_classifier.load_model()

    return D_classifier.predict(img)


def Letter_predict(img):
    L_classifier = Letter_prediction()
    L_classifier.load_model()

    return L_classifier.predict(img)


def D_predict(img):
    object_predict = Object_prediction('')
    object_predict.load_model()
    prediction = object_predict.predict(img)
    if prediction[0] <=9:
        return prediction[0]
    else:
        return id_toletter([prediction[0]])


    # D_classifier = Digit_prediction()
    # D_classifier.load_model()
    # L_classifier = Letter_prediction()
    # L_classifier.load_model()
    #
    # D_prediction = D_classifier.predict(img)
    # L_prediction = L_classifier.predict(img)
    #
    # if D_prediction[1] > L_prediction[1]:
    #     return D_prediction[0]
    # else:
    #     return id_toletter([L_prediction[0]])


if __name__ == '__main__':
    # img = load_image("test_digit/img4.png")
    # print(Digit_predict(img))

    img = load_image('test_digit/img8.png')
    # img = load_image('test_letters/imgD.png')

    print(D_predict(img))

    # O_predict = Object_prediction(model=learn_objects((512,128)))
    # O_predict.save_model()

    # X_valid = np.load('X_train.npy')
    # y_valid = np.load('y_train.npy')
    #
    # X_validl = np.load('X_trainl.npy')
    # y_validl = np.load('y_trainl.npy')
    #
    #
    # X_object = np.concatenate((X_valid, X_validl))
    # y_object = np.concatenate((y_valid, y_validl))
    # X_traino, X_testo, y_traino, y_testo = train_test_split(X_object, y_object, test_size=0.2)
    #
    # np.save('X_traino.npy', X_traino)
    # np.save('X_testo.npy', X_testo)
    # np.save('y_traino.npy', y_traino)
    # np.save('y_testo.npy', y_testo)

