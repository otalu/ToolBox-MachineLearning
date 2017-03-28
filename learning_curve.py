"""
Exploring learning curves for classification of handwritten digits

Author: Onur Talu
"""

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


def display_digits():
    digits = load_digits()
    print(digits.DESCR)
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i+1)
        subplot.matshow(numpy.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()


def train_model(num_trials, LR):
    data = load_digits()
    train_percentages = range(5, 95, 5)
    test_accuracies = numpy.zeros(len(train_percentages))
    for i in range(num_trials):
        index = 0
        for perc in train_percentages:
            X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=perc/100)
            model = LogisticRegression(C=LR)
            model.fit(X_train, y_train)
            test_accuracies[index] += model.score(X_train, y_train)
            index += 1
    test_accuracies = test_accuracies/num_trials
    print("Train accuracy %f" %model.score(X_train, y_train))
    print("Test accuracy %f"%model.score(X_test, y_test))
    fig = plt.figure()
    plt.plot(train_percentages, test_accuracies)
    plt.xlabel('Percentage of Data Used for Training')
    plt.ylabel('Accuracy on Test Set')
    print(type(num_trials))
    # plt.savefig('Trial%d.png' % num_trials)
    plt.savefig('LR_Trials1.png')
    plt.show()


if __name__ == "__main__":
    # don't forget to change the name of saved image in line 46
    train_model(10, 10**-1)
