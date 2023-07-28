# %%
"""
Author: Mélina Verger

Compute model absolute density distance (MADD).
"""

# To exit script
from sys import exit

# To load the trained models
import pickle

# For data manipulation
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# To print with tabular format
from tabulate import tabulate

# Plotting module
import matplotlib.pyplot as plt

# %% [markdown]
# ## Loading

# %%
DATA = pickle.load(open("../data/DATA", "rb"))
SPLIT = pickle.load(open("../data/SPLIT", "rb"))
SFEATURES = pickle.load(open("../data/SFEATURES", "rb"))

print(DATA)
print(SPLIT)
print(SFEATURES)

# %%
# Load test sets

X_test = pd.read_csv("../data/X_test" + "_" + DATA + "_" + SPLIT + ".csv")
y_test = pd.read_csv("../data/y_test" + "_" + DATA + "_" + SPLIT + ".csv")

# %%
# Load trained models

MODELS = pickle.load(open("../data/MODELS", "rb"))  # dict with names and trained models
models = MODELS

# %%
if "clf_svc" in models:
    del models["clf_svc"]  # except svc model because no probability outputs 

# %% [markdown]
# ## Separate data sets by (un-)protected groups

# %% [markdown]
# /!\ The following only works for binary sensitive features.

# %%
dict_subsets_test = dict()

for sensfeat in SFEATURES:
    # X_test_{sensitive feature and group 1/0}
    dict_subsets_test["X"+ "_test_" + sensfeat + "_"+ "1"] = X_test[X_test[sensfeat] == 1]
    dict_subsets_test["X"+ "_test_" + sensfeat + "_"+ "0"] = X_test[X_test[sensfeat] == 0]
    # y_test_{sensitive feature and group 1/0}
    dict_subsets_test["y"+ "_test_" + sensfeat + "_"+ "1"] = y_test.loc[dict_subsets_test["X" + "_test_" + sensfeat + "_" + "1"].index]
    dict_subsets_test["y"+ "_test_" + sensfeat + "_"+ "0"] = y_test.loc[dict_subsets_test["X" + "_test_" + sensfeat + "_" + "0"].index]

# %% [markdown]
# ## Prediction probabilities

# %%
dict_subsets_PP = dict()

for mod_names in models:

    if mod_names == "clf_lr":
        modl = "lr"
    elif mod_names == "clf_kn":
        modl = "kn"
    elif mod_names == "clf_dt":
        modl = "dt" 
    elif mod_names == "clf_rf":
        modl = "rf"
    elif mod_names == "clf_cnb":
        modl = "cnb"
    elif mod_names == "clf_mnb":
        modl = "mnb"
    else:
        print("Invalid model.")
        exit()

    # y_PP for X_test_{sensitive feature and group 1/0}
    for sensfeat in SFEATURES:
        dict_subsets_PP["y" + "_PP_" + modl + "_" + sensfeat + "_" + "1"] = models[mod_names].predict_proba(dict_subsets_test["X"+ "_test_" + sensfeat + "_"+ "1"])[:, 1]  # [:, 1] because propa of being in the class 1
        dict_subsets_PP["y" + "_PP_" + modl + "_" + sensfeat + "_" + "0"] = models[mod_names].predict_proba(dict_subsets_test["X"+ "_test_" + sensfeat + "_"+ "0"])[:, 0]  # [:, 0] because propa of being in the class 0

# %% [markdown]
# ## Model Absolute Density Distance (MADD)

# %% [markdown]
# /!\ The initial data vectors do not have the same length but it is handled by the density vectors transformation.

# %%
# Probability sampling step

e = 0.1

if e == 0.1:
    nb_decimals = 1
    nb_components = 11

if e == 0.01:
    nb_decimals = 2
    nb_components = 101

if e == 0.001:
    nb_decimals = 3
    nb_components = 1001

if e == 0.0001:
    nb_decimals = 4
    nb_components = 10001

NB_COMPO = nb_components
pickle.dump(NB_COMPO, open("../data/NB_COMPO", "wb"))

# %%
def normalized_density_vector(pred_proba_array):

    PP_rounded = np.around(pred_proba_array, decimals=nb_decimals)

    density_vector = np.zeros(nb_components)  # empty
    proba_values = np.linspace(0, 1, nb_components)  # 101 increasing components

    for i in range(len(proba_values)):
        compar = proba_values[i]
        count = 0
        for x in PP_rounded:
            if x == compar:
                count = count + 1
        density_vector[i] = count
    
    normalized_density_vec = density_vector / np.sum(density_vector)

    return normalized_density_vec

# %%
def MADD(norm_densvect_1, norm_densvect_0):
    return np.absolute(norm_densvect_1 - norm_densvect_0).sum()

# %%
res = list()

for sensfeat in SFEATURES:

    for mod_name in models:

        if mod_name == "clf_lr":
            modl = "lr"
        elif mod_name == "clf_kn":
            modl = "kn"  # model that generates FutureWarning
        elif mod_name == "clf_dt":
            modl = "dt" 
        elif mod_name == "clf_rf":
            modl = "rf"
        elif mod_name == "clf_cnb":
            modl = "cnb"
        elif mod_name == "clf_mnb":
            modl = "mnb"
        else:
            print("Invalid model.")
            exit()
    
        subres = list()
        subres.append(sensfeat)
        subres.append(modl)

        norm_densvect1 = normalized_density_vector(dict_subsets_PP["y" + "_PP_" + modl + "_" + sensfeat + "_" + "1"])
        norm_densvect0 = normalized_density_vector(dict_subsets_PP["y" + "_PP_" + modl + "_" + sensfeat + "_" + "0"])
        MADDvalue = round(MADD(norm_densvect1, norm_densvect0), nb_decimals)

        subres.append(MADDvalue)

        res.append(subres)

print(tabulate(res, headers=["Sensitive feature", "Model", "  MADD     "]))


