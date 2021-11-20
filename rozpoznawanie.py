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
from time import time




def learn_digits():
    def load_data(path): #returns data as np.array
        data = pd.read_csv(path)
        data = np.array(data)
        return data

    train = load_data("EMNIST/emnist-byclass-train.csv")
    X_train = train[:, 1:]
    y_train = train[:, 0]
    test = load_data("EMNIST/emnist-byclass-test.csv")
    X_test = test[:, 1:]
    y_test = test[:, 0]

    #classes_balanced = ['0','1','2','3','4','5','6','7','8','9']


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


if __name__ == '__main__':
    start_time = time()
    learn_digits()
    print("--- %s seconds ---" % (time() - start_time))