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

import string





def learn_digits():
    def load_data():
        pass
        # X = np.ones((10160,s*s))
        # y = np.zeros(10160)
        # for digit in digits:
        #     filenames = os.listdir('digits/' + str(digit))
        #     L = len(filenames)
        #     for i in range(L):
        #         name = filenames[i]
        #         y[i+d*1016] = digit
        #         img = Image.open("digits/" + str(digit) + "/" + name) # This returns an image object
        #         img = img.resize((s,s))
        #         img = np.asarray(img) # convert it to ndarray
        #         img = img.reshape(-1, img.size)
        #         X[i+digit*1016,:] = img
        # return X, y

    classes_balanced = ['0','1','2','3','4','5','6','7','8','9']
    print(data)

    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.3, random_state=42)

    def classify_digits(X_train = X_train, X_test= X_test, y_train = y_train, y_test = y_test):
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)

        count = 0
        for i in range(len(X_test)):
            if p[i] == y_test[i]:
                count += 1
        print(count/len(X_test))

    classify_digits()

def learn_letters():
    letters_up = list(string.ascii_uppercase)
    letters_low = list(string.ascii_lowercase)

    for letter in letters_up:
        filenames = os.listdir('letters/' + str(letter))
        L = len(filenames)
        print(filenames)
if __name__ == '__main__':
    data = pd.read_csv("EMNIST/emnist-balanced-train.csv")
    for i in range(0, 10):
        digit = list(data.loc[i])
        print(digit[0])
        img = np.array(digit[1:])
        img.resize((28,28))
        img = Image.fromarray(img)
        img.show()