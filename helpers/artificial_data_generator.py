import sklearn.datasets
import numpy as np
import random
from sklearn import preprocessing

def to_one_hot(array, nbr_classes):
    nbr_sample = array.shape[0]
    to_return = np.zeros(nbr_sample*nbr_classes)
    indexes = np.arange(0,nbr_sample)*nbr_classes
    to_return[indexes + array] = 1
    return to_return.reshape((nbr_sample, nbr_classes))

def generate_classification_nl(NBR_SAMPLES = 5000, REAL_FEATURES = 5, COMBINATION_FEATURES = 15, TOTAL_FEATURES = 500, n_classes =2):

    datas, answers = sklearn.datasets.make_classification(n_samples = NBR_SAMPLES, n_features = TOTAL_FEATURES,
                                                          n_informative = REAL_FEATURES, n_redundant = COMBINATION_FEATURES,
                                                          shuffle = False, n_classes=n_classes)

    # Shuffle the samples because it has not been done
    indexes = np.arange(0,datas.shape[0])
    random.shuffle(indexes)
    datas = datas[indexes]

    answers = answers[indexes]

    datas = preprocessing.scale(datas)

    # Transform answers into one hot vector
    nbr_classes = np.unique(answers).shape[0]
    answers = to_one_hot(answers, nbr_classes)

    return datas, answers

def generate_regression_nl(NBR_SAMPLES = 2000, REAL_FEATURES = 5, COMBINATION_FEATURES = 15, TOTAL_FEATURES = 500):
    datas, answers = sklearn.datasets.make_friedman1(n_samples = NBR_SAMPLES, n_features = TOTAL_FEATURES, noise = 1.0)
    datas = preprocessing.scale(datas)
    answers = preprocessing.scale(answers)

    answers = np.reshape(answers, (-1,1))
    return datas, answers

def generate_regression_l(NBR_SAMPLES = 5000, REAL_FEATURES = 5, COMBINATION_FEATURES = 15, TOTAL_FEATURES = 500):

    datas, answers = sklearn.datasets.make_regression(n_samples = NBR_SAMPLES, n_features = TOTAL_FEATURES,
                                                      n_informative = REAL_FEATURES, shuffle = False)

    # Shuffle the samples because it has not been done
    indexes = np.arange(0,datas.shape[0])
    random.shuffle(indexes)
    datas = datas[indexes]

    answers = answers[indexes]

    datas = preprocessing.scale(datas)
    answers = preprocessing.scale(answers)

    answers = np.reshape(answers, (-1,1))
    return datas, answers

def generate_classification_l(NBR_SAMPLES = 5000, REAL_FEATURES = 5, COMBINATION_FEATURES = 15, TOTAL_FEATURES = 500):
    datas, answers = sklearn.datasets.make_regression(n_samples = NBR_SAMPLES, n_features = TOTAL_FEATURES,
                                                      n_informative = REAL_FEATURES, shuffle = False)
    esperance = np.median(answers)
    classes = [[1, 0] if x > esperance else [0,1] for x in answers]

    datas = preprocessing.scale(datas)
    return datas, np.array(classes)