import pandas as pd
import numpy as np
import random
import os
import json
from colorama import init, Fore

# Some basic functions performing basic operations:
# - Shuffling
# - Batch splitting
# - Grid Search with Cross Validation
# - Cross Validation

def rotate(list_, shift):
    return list_[shift:] + list_[:shift]

def shuffle(X, y):
    n_samples = X.shape[0]
    ind_list = [i for i in range(n_samples)]

    random.shuffle(ind_list) # Don't remove the prefix "random."(!)

    return X[ind_list, :], y[ind_list,]

def batchify(n_samples, batch_len):
    first = 0
    last = batch_len
    while n_samples // last:
        yield slice(first, last)
        first = last
        last = first + batch_len
         
    if first < n_samples:
        yield slice(first, n_samples)

def cross_validation(MLPConstructor, params, X, y, k_fold):
    n_samples, _ = X.shape

    batch_len = n_samples // k_fold
    folders = [bs for bs in batchify(n_samples, batch_len)]
    scores = []

    for _ in range(k_fold):
        model = MLPConstructor(**params)
        test_slice = folders[-1] 

        X_train = []
        y_train = []

        for slice_ in folders[:-1]: 
            X_train.extend(X[slice_].tolist()) 
            y_train.extend(y[slice_].tolist())

        X_test = X[test_slice]
        y_test = y[test_slice]
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test  = np.array(X_test)
        y_test  = np.array(y_test)
        
        model.fit(X_train, y_train, X_test, y_test)
        
        score = model.score(X_test, y_test)
        scores.append((model, score, model.score(X_train, y_train)))
        folders = rotate(folders, 1)

    scores.sort(key=lambda x: x[1])

    evaluations = np.array([pair[1] for pair in scores]) # Getting the list containing scores
    tr_evaluations = np.array([pair[2] for pair in scores]) # Getting the list containing scores
    best_model = scores[-1][0]

    return best_model, evaluations.mean(), evaluations.std(), tr_evaluations.mean(), tr_evaluations.std()

def grid_search_cv(MLPConstructor, params, classifier, X, y, k_fold=5):
    
    comb = [(n_hidden, rate, momentum, reg, epoch, tol, pat, func, batch_len, n_iter)
            for n_hidden in params["n_hidden"]
            for rate in params["learning_rate"]
            for momentum in params["momentum"]
            for reg in params["regularization"]
            for epoch in params["max_epoch"]
            for tol in params["tolerance"]
            for pat in params["patience"]
            for func in params["activation"]
            for batch_len in params["batch_len"]
            for n_iter in params["n_iter_no_early_stop"]
            ]
    
    printProgressBar(0, len(comb), decimals=1, length=50)
    classification = []
    
    for n_h, rate, mom, reg, epoch, tol, pat, func, b_len, n_iter in comb:
        best = None
        model_params = {
            "n_hidden"      : n_h,
            "learning_rate" : rate,
            "momentum"      : mom,
            "regularization": reg,
            "max_epoch"     : epoch,
            "tolerance"     : tol,
            "patience"      : pat,
            "activation"    : func,
            "batch_len"     : b_len,
            "n_iter_no_early_stop": n_iter,
            "is_classifier" : classifier                
        }

        model, score_mean, score_std, tr_score_mean, tr_score_std = cross_validation(MLPConstructor, model_params, X, y, k_fold)
        classification.append((model, score_mean, score_std, tr_score_mean, tr_score_std))

        printProgressBar(len(classification), len(comb), decimals=1, length=50)

    classification.sort(key=lambda x: x[1])
    
    return classification         

def save_mlp(mlp, path):
    model_to_save = mlp.get_params()
    model_to_save["batch_len"] = int(model_to_save["batch_len"])
    model_to_save["weights_"]  = [w.tolist() for w in mlp._best_weights]
    model_to_save["bias_"]     = [w.tolist() for w in mlp._best_bias]

    json_to_save = {}
    json_to_save["model"] = model_to_save
    json_to_save["loss"]  = mlp.loss_curve
    json_to_save["validation_loss"] = mlp.validation_loss
    json_to_save["score"] = mlp.scores
    json_to_save["tr_score"] = mlp.tr_scores
    
    with open(path, 'w') as outfile:  
        json.dump(json_to_save, outfile, indent=4)

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    os.system("cls" if os.name == "nt" else "clear")
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def print_params(params):
    print(Fore.YELLOW + "{")
    print("    " + Fore.YELLOW + "n_hidden: "+ Fore.CYAN + str(params["n_hidden"]))
    print("    " + Fore.YELLOW + "learning_rate: "+ Fore.CYAN + str(params["learning_rate"]))
    print("    " + Fore.YELLOW + "momentum: "+ Fore.CYAN + str(params["momentum"]))
    print("    " + Fore.YELLOW + "regularization: "+ Fore.CYAN + str(params["regularization"]))
    print("    " + Fore.YELLOW + "max_epoch: "+ Fore.CYAN + str(params["max_epoch"]))
    print("    " + Fore.YELLOW + "tolerance: "+ Fore.CYAN + str(params["tolerance"]))
    print("    " + Fore.YELLOW + "patience: "+ Fore.CYAN + str(params["patience"]))
    print("    " + Fore.YELLOW + "activation: "+ Fore.CYAN + str(params["activation"]))
    print("    " + Fore.YELLOW + "batch_len: "+ Fore.CYAN + str(params["batch_len"]))
    print("    " + Fore.YELLOW + "n_iter_no_early_stop: "+ Fore.CYAN + str(params["n_iter_no_early_stop"]))
    print(Fore.YELLOW + "}")
    print(Fore.WHITE)

# Set of functions returning datasets useful for both training and testing
# for the implemented MLP.
databases = "../databases/"

# Returns the well-known MONK training set #1
def load_monks_1_tr():
    d = pd.read_csv(databases + "monks/monks-1.train",
                    sep=' ',
                    header=-1,
                    skipinitialspace=True)

    return np.array(d[[1,2,3,4,5,6]].values), np.array(d[[0]].values)

# Returns the well-known MONK test set #1
def load_monks_1_ts():
    d = pd.read_csv(databases+"monks/monks-1.test",
                    sep=' ',
                    header=-1,
                    skipinitialspace=True)

    return np.array(d[[1,2,3,4,5,6]].values), np.array(d[[0]].values)


# Returns the well-known MONK training set #2
def load_monks_2_tr():
    d = pd.read_csv(databases + "monks/monks-2.train",
                    sep=' ',
                    header=-1,
                    skipinitialspace=True)

    return np.array(d[[1,2,3,4,5,6]].values), np.array(d[[0]].values)

# Returns the well-known MONK test set #2
def load_monks_2_ts():
    d = pd.read_csv(databases + "monks/monks-2.test",
                    sep=' ',
                    header=-1,
                    skipinitialspace=True)

    return np.array(d[[1,2,3,4,5,6]].values), np.array(d[[0]].values)

# Returns the well-known MONK training set #3
def load_monks_3_tr():
    d = pd.read_csv(databases + "monks/monks-3.train",
                    sep=' ',
                    header=-1,
                    skipinitialspace=True)

    return np.array(d[[1,2,3,4,5,6]].values), np.array(d[[0]].values)

# Returns the well-known MONK test set #2
def load_monks_3_ts():
    d = pd.read_csv(databases + "monks/monks-3.test",
                    sep=' ',
                    header=-1,
                    skipinitialspace=True)

    return np.array(d[[1,2,3,4,5,6]].values), np.array(d[[0]].values)
