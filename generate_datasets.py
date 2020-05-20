import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
root = os.getcwd()
sys.path.append(os.getcwd())

from helpers.artificial_data_generator import generate_regression_nl, generate_regression_l, generate_classification_l, generate_classification_nl
from helpers.aupr import get_aupr
import pickle
import numpy as np

for n_features in [5000]:
    for nbr in range(0,5):
        for t in ["regression", "classification"]:
            for t_ in ["l", "nl"]:

                # GENERATE DATASETS
                if t == "classification":
                    if t_ == "l":
                        datas, answers = generate_classification_l(NBR_SAMPLES=10000, REAL_FEATURES=25,
                                                                   COMBINATION_FEATURES=0,
                                                                   TOTAL_FEATURES=n_features)
                        useful_features = np.zeros(n_features, dtype=np.float32)
                        useful_features[:25] = 1.0
                    elif t_ == "nl":
                        datas, answers = generate_classification_nl(NBR_SAMPLES=10000, REAL_FEATURES=25,
                                                                    COMBINATION_FEATURES=0,
                                                                    TOTAL_FEATURES=n_features)
                        useful_features = np.zeros(n_features, dtype=np.float32)
                        useful_features[:25] = 1.0
                elif t == "regression":
                    if t_ == "l":
                        datas, answers = generate_regression_l(NBR_SAMPLES=10000, REAL_FEATURES=25,
                                                               COMBINATION_FEATURES=0,
                                                               TOTAL_FEATURES=n_features)
                        useful_features = np.zeros(n_features, dtype=np.float32)
                        useful_features[:25] = 1.0
                    elif t_ == "nl":
                        datas, answers = generate_regression_nl(NBR_SAMPLES=10000, REAL_FEATURES=5,
                                                                COMBINATION_FEATURES=15,
                                                                TOTAL_FEATURES=n_features)
                        useful_features = np.zeros(n_features, dtype=np.float32)
                        useful_features[:5] = 1.0
                print ("Datasets/" + str(t) + "_" + str(t_) + "_" + str(nbr))
                d = {'answers':answers, 'datas':datas, 'useful_features':useful_features}
                pickle.dump(d, open("Datasets/" + str(t) + "_" + str(t_) + "_" + str(n_features) + "_" + str(nbr), "wb"))
