# %%
"""
Author: Mélina Verger

Visual prediction probabilities distribution inspections.
"""

# To exit script
from sys import exit

# To load the trained models
import pickle

# For data manipulation
import pandas as pd
import numpy as np

# Plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# ## Load data sets and trained models

# %%
DATA = pickle.load(open("../data/DATA", "rb"))
SPLIT = pickle.load(open("../data/SPLIT", "rb"))
SFEATURES = pickle.load(open("../data/SFEATURES", "rb"))
NB_COMPO = pickle.load(open("../data/NB_COMPO", "rb"))

print(DATA)
print(SPLIT)
print(SFEATURES)
print(NB_COMPO)

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
    del models["clf_svc"]  # except svc 

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
# ## Plotting prediction probability distributions

# %%
for sensfeat in SFEATURES:

    if sensfeat == "gender":
        color_gp1 = "mediumaquamarine"
        color_gp0 = "lightcoral"
    elif sensfeat == "imd_band" or sensfeat == "poverty":
        color_gp1 = "gold"
        color_gp0 = "dimgray"
    elif sensfeat == "disability":
        color_gp1 = "mediumpurple"
        color_gp0 = "lightskyblue"
    elif sensfeat == "age_band" or sensfeat == "age":
        color_gp1 = "salmon"
        color_gp0 = "seagreen"
    else:
        # random colors
        color_gp1 = (np.random.random(), np.random.random(), np.random.random())
        color_gp0 = (np.random.random(), np.random.random(), np.random.random())

    

    for mod_name in models:  # except svc model because no probability outputs 

        if mod_name == "clf_lr":
            model_name = "LR"
        elif mod_name == "clf_kn":
            model_name = "KN"
        elif mod_name == "clf_dt":
            model_name = "DT"
        elif mod_name == "clf_rf":
            model_name = "RF"
        elif mod_name == "clf_cnb":
            model_name = "NB"
        elif mod_name == "clf_mnb":
            model_name = "NB"
        else:
            print("Invalid model.")
            exit()
        
        fig, axes = plt.subplots(1, 3, figsize=(10, 2.5), constrained_layout=True)  # figsize=(12, 4) for better visualization
        fig.supxlabel("Predicted probabilities  [0 ; 1]", fontsize=16, fontweight='bold')
        # plot the 2 y_pred_{sensitive feature for group 0/1} separately
        ax0 = sns.histplot(ax=axes[0], data=models[mod_name].predict_proba(dict_subsets_test["X"+ "_test_" + sensfeat + "_"+ "1"])[:, 1], kde=False, stat="proportion", color=color_gp1, bins=np.linspace(0,1,NB_COMPO))
        #ax0.set_ylim(0, 0.3)
        ax0.set_xlim(0, 1)
        ax0.set_ylabel("Density", fontsize=16, fontweight='bold')
        ax1 = sns.histplot(ax=axes[1], data=models[mod_name].predict_proba(dict_subsets_test["X"+ "_test_" + sensfeat + "_"+ "0"])[:, 0], kde=False, stat="proportion", color=color_gp0, bins=np.linspace(0,1,NB_COMPO))
        #ax1.set_ylim(0, 0.3)
        ax1.set_xlim(0, 1)
        ax1.set_yticklabels([]) # turn off y ticks labels
        ax1.yaxis.set_visible(False)
        # plot the 2 DDPs on the same graph
        ax2 = sns.kdeplot(ax=axes[2], data=models[mod_name].predict_proba(dict_subsets_test["X"+ "_test_" + sensfeat + "_"+ "1"])[:, 1], color=color_gp1, label=sensfeat + ": 1")
        ax2 = sns.kdeplot(ax=axes[2], data=models[mod_name].predict_proba(dict_subsets_test["X"+ "_test_" + sensfeat + "_"+ "0"])[:, 0], color=color_gp0, label=sensfeat + ": 0")
        ax2.set_ylabel("Density", fontsize=16, fontweight='bold')
        ax2.set_xlim(0, 1)
        
        #plt.legend(loc="upper left")
        plt.legend(bbox_to_anchor = (1.65, 0.5), loc='center right', prop={'weight':'bold'})
        ax1.set_title(f"{model_name}", loc="center", fontsize=16, fontweight='bold')


