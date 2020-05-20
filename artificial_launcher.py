#!/usr//bin/python3
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
root = os.getcwd()
sys.path.append(os.getcwd())

from helpers.artificial_data_generator import generate_regression_nl, generate_regression_l, generate_classification_l, generate_classification_nl
from helpers.aupr import get_aupr
import numpy as np
from FSNET import FSNET
import argparse
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import pickle
#os.chdir("../")
import tensorflow as tf

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


x = argparse.ArgumentParser()

args = x.parse_args()
load =False
n_features = 5000
train_samples = 2000


layers_sizes = [300, 150, 75]
for t in ["regression", "classification"]:
    if t == "regression":
        outdim = 1
    else:
        outdim = 2
    for t_ in ["nl", "l"]:
        for nbr in range(5):
            d = pickle.load(open("Datasets/" + str(t) + "_" + str(t_) + "_" + str(n_features) + "_" + str(nbr),"rb"))
            datas, answers, useful_features = d['datas'], d['answers'], d['useful_features']
            train_dataset = {'inputs': [np.reshape(el, [1, -1]) for el in datas[:train_samples]],
                             'outputs': [np.reshape(el, [1, -1]) for el in answers[:train_samples]]}

            test_dataset = {'inputs': [np.reshape(el, [1, -1]) for el in datas[train_samples:]],
                            'outputs': [np.reshape(el, [1, -1]) for el in answers[train_samples:]]}
            for build_selec in [False, True]:
                if build_selec:
                    regul_possible = ["l1l2"]
                else:
                    regul_possible = ["none"]
                for regul_type in regul_possible:
                    alphas = [0.0, 10.0, 100.0, 1000.0]
                    alphas_2 = [0.0, 10000.0, 100000.0, 1000000.0]
                    if regul_type == "l2" or regul_type == "none":
                        alphas = [0.0]
                    if regul_type == "l1" or regul_type == "none":
                        alphas_2 = [0.0]
                    for a in alphas:
                        for a2 in alphas_2:
                            current_net_name = t + "_" + t_ + "_" + str(build_selec) + "_" + \
                                               regul_type + "_" + str(nbr) + "_" + str(a) + "_" + str(a2)
                            r = {}
                            fsnet = FSNET(params = {'outdim':outdim,
                                                    'hidden':layers_sizes,
                                                    'build_selec':build_selec,
                                                    'lr':1e-3},
                                          save_path = "saved_results/", name = current_net_name)

                            if t == "classification":
                                score_loss = fsnet.cross_entropy
                            else:
                                score_loss = fsnet.mse

                            if regul_type == "l1":
                                regul_loss = lambda x : fsnet.l1(x, a, a2)
                            elif regul_type == "l2":
                                regul_loss = lambda x : fsnet.l2(x, a, a2)
                            elif regul_type == "l1l2":
                                regul_loss = lambda x : fsnet.l1l2(x, a, a2)
                            else:
                                regul_loss = lambda x : fsnet.noregul(x, a, a2)

                            train_loss = lambda x, y : fsnet.train_loss(x, y, 0.5, score_loss, regul_loss)
                            val_loss = lambda x, y : fsnet.train_loss(x, y, 0.0, score_loss, regul_loss)
                            importances_func = lambda x, y : fsnet.get_importances(x, y, np.ones((outdim), dtype=np.float32))

                            train_dataset_s = fsnet.standardize(train_dataset)
                            test_dataset_s = fsnet.standardize(test_dataset)

                            fsnet.train(train_dataset_s, steps=30000, batch_size=50, val_data=test_dataset_s, compiled = True, checkpoint_every=10000,
                                        display_init=1000, complete_unroll=True, func= train_loss, val_func=val_loss)

                            results = fsnet.evaluate(test_dataset_s, val_loss, compiled=True)
                            print (results)
                            r['results'] = results

                            imps = fsnet(func = importances_func, input_data = test_dataset_s, compiled = True)
                            if t == "classification":
                                score = np.mean(1 - fsnet.accuracy(datas[train_samples:].astype(np.float32),
                                                                   answers[train_samples:].astype(np.float32), 0.0))
                            else:
                                score = np.mean(fsnet.mse(datas[train_samples:].astype(np.float32),
                                                          answers[train_samples:].astype(np.float32), 0.0))
                            r['score'] = score


                            # COMPUTE AUPR
                            r['aupr'] = {}
                            imp_d = np.abs(np.array(imps['gradient']))
                            tmp_d_b = imp_d[:,0,:]
                            imp_d = np.mean(imp_d, axis = 0)
                            grad_aupr = get_aupr(imp_d, useful_features)
                            print (grad_aupr)
                            r['aupr']['gradient'] = grad_aupr

                            imp_a = np.abs(np.array(imps['lrp']))
                            tmp_a_b = imp_a[:, 0, :]
                            imp_a = np.mean(imp_a, axis=0)
                            activ_aupr = get_aupr(imp_a, useful_features)
                            print(activ_aupr)
                            r['aupr']['lrp'] = activ_aupr

                            if build_selec:
                                imp_s = np.abs(np.array(imps['selec']))
                                tmp_s_b = imp_s[:, 0, :]
                                imp_s = np.mean(imp_s, axis=0)
                                selec_aupr = get_aupr(imp_s, useful_features)
                                print(selec_aupr)
                                r['aupr']['selec'] = selec_aupr

                            pickle.dump(r, open("ArtificialResults/" + str(current_net_name), "wb"))

                            del(fsnet)

            for K in ['sqrt', 'log2', int(n_features / 3), int(n_features / 2)]:
                current_rf_name = t + "_" + t_ + "_RF_" + str(K) + "_" + str(nbr)
                r_f = {}

                if t == 'classification':
                    forest = RandomForestClassifier(n_estimators=1000, max_depth=20, max_features=K, n_jobs=40)
                    forest.fit(datas[:train_samples], answers[:train_samples])
                    score = forest.score(datas[train_samples:], answers[train_samples:])
                else:
                    forest = RandomForestRegressor(n_estimators=1000, max_depth=20, max_features=K, n_jobs=40)
                    forest.fit(datas[:train_samples], answers[:train_samples])
                    score = mean_squared_error(answers[train_samples:], forest.predict(datas[train_samples:]))

                forest_imp = forest.feature_importances_
                aupr_forest = get_aupr(forest_imp, useful_features)
                r_f['results'] = score
                r_f['imps'] = forest_imp
                r_f['aupr'] = aupr_forest
                pickle.dump(r_f, open("ArtificialResults/" + current_rf_name, "wb"))
                print (r_f['aupr'])

