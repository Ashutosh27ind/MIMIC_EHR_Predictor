# python run_shallow_explain.py --outcome=mortality --T=48.0 --dt=1.0 --model_type=RF

import sys, os, time, pickle, random
import pandas as pd
import numpy as np
import joblib

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import yaml

# Ashutosh added new imports for explainataion:
import matplotlib.pyplot as plt
import plotly
import shap
#import lime
#import lime.lime_tabular
import json

with open('config.yaml') as f:
    config = yaml.safe_load(f)

data_path = config['data_path']
search_budget = config['train']['budget']


# Ashutosh added below :
os.environ['NUMEXPR_MAX_THREADS'] = '50'

#figure_path = '/mnt/disks/user/project/FIDDLE_experiments-master/mimic3_experiments/figures/' # Nibmlebox
figure_path = '/home/azureuser/cloudfiles/code/Users/ashutoshind2017/FIDDLE_experiments-master/mimic3_experiments/figures'  # Azure 

# Ashutosh changing the jobs from 50 to 22 :
n_jobs = 20
#n_jobs = -2

"""
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--outcome', type=str, required=True)
parser.add_argument('--T', type=float, required=True)
parser.add_argument('--dt', type=float, required=True)
parser.add_argument('--model_type', type=str, required=True)
parser.add_argument('--cuda', type=int, default=7)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
"""

task = 'mortality'
model_type = 'RF'
model_name = model_type

T = 48.0
dt = 1.0

if model_type == 'CNN':
    assert False
elif model_type == 'RNN':
    assert False
elif model_type == 'LR':
    pass
elif model_type == 'RF':
    pass
else:
    assert False


print('EXPERIMENT:', 'model={},outcome={},T={},dt={}'.format(model_name, task, T, dt))

# Create checkpoint directories
import pathlib
pathlib.Path("./checkpoint/model={},outcome={},T={},dt={}/".format(model_name, task, T, dt)).mkdir(parents=True, exist_ok=True)

######
# Data
import lib.data as data
if task == 'mortality':
    tr_loader, va_loader, te_loader = data.get_benchmark_splits(fuse=True)
else:
    tr_loader, va_loader, te_loader = data.get_train_val_test(task, duration=T, timestep=dt, fuse=True)

# Reshape feature vectors
X_tr, s_tr, y_tr = tr_loader.dataset.X, tr_loader.dataset.s, tr_loader.dataset.y
X_va, s_va, y_va = va_loader.dataset.X, va_loader.dataset.s, va_loader.dataset.y
X_te, s_te, y_te = te_loader.dataset.X, te_loader.dataset.s, te_loader.dataset.y

X_tr.shape, s_tr.shape, y_tr.shape, X_va.shape, s_va.shape, y_va.shape, X_te.shape, s_te.shape, y_te.shape, 

# Concatenate tr+va to create large training set (used for cross-validation)
Xtr = np.concatenate([X_tr, X_va])
ytr = np.concatenate([y_tr, y_va]).ravel()
Str = np.concatenate([s_tr, s_va])

Xte = X_te
yte = y_te.ravel()
Ste = s_te

# Flatten time series features
Xtr = Xtr.reshape(Xtr.shape[0], -1)
Xte = Xte.reshape(Xte.shape[0], -1)

# Combine time-invariant and time series
Xtr = np.concatenate([Xtr, Str], axis=1)
Xte = np.concatenate([Xte, Ste], axis=1)

print("Printing information of Xtr , ytr etc now............")
print(Xtr.shape, ytr.shape, Xte.shape, yte.shape)
print("Printed .............................................")

######
# Train model with CV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn import metrics, feature_selection, utils
import scipy.stats
from joblib import Parallel, delayed
from tqdm import tqdm_notebook as tqdm

"""
if args.seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
"""

# Ashutosh adding the explaination module : 
if model_type == 'LR':
    #clf.fit(Xtr, ytr)
    print("LR..............")
else:
    #clf.fit(Xtr, ytr)
    print("RF.............")
    
    """
    
    # SHAP Dependence plots for different features :
    #fig3 = shap.dependence_plot("alcohol", shap_values, Xtr)
    #fig3.savefig(figure_path + 'SHAP_dependence_plot.png', dpi=600)
    
    # SHAP Interaction plots : 
    # Taking more time than SHAP values to compute, since this is just an example we only explain, so considering 1000 values only : 
    shap_interaction_values = shap.TreeExplainer(clf.best_estimator_).shap_interaction_values(Xtr.iloc[:1000,:])
    fig4 = shap.summary_plot(shap_interaction_values, Xtr.iloc[:1000,:])
    fig4.savefig(figure_path + 'SHAP_Interaction_Plots.png', dpi=600)
    
    # SHAP Waterfall plots :
    # visualize the first prediction's explanation
    fig5 = shap.plots.waterfall(shap_values[0])
    fig5.savefig(figure_path + 'SHAP_WaterFall_Plots.png', dpi=600)
    
    # SHAP Force Plot : 
    # visualize the first prediction's explanation with a force plot
    fig6 = shap.plots.force(shap_values[0])
    fig6.savefig(figure_path + 'SHAP_Force_Plot.png', dpi=600)
    
    # Visualize all the training set predictions
    fig7 = shap.plots.force(shap_values)
    fig7.savefig(figure_path + 'SHAP_Training_All.png', dpi=600)
    
    # Summarize the effects of all the features
    fig8 = shap.plots.beeswarm(shap_values)
    fig8.savefig(figure_path + 'SHAP_all_features.png', dpi=600)
    
    # Create a dependence scatter plot to show the effect of a single feature across the whole dataset
    # fig 9 = shap.plots.scatter(shap_values[:,"RM"], color=shap_values)
    #fig9.savefig(figure_path + 'SHAP_single_feature_effect.png', dpi=600)
    
    # HeatMap plot : (100 values)
    fig10 = shap.plots.heatmap(shap_values[200:300])
    fig10.savefig(figure_path + 'SHAP_heatmap.png', dpi=600)
    """
    # Creating a sub-set of dataset for testing :
    #Xtr_new = Xtr[1, 0:10]
    #print("Xtr_new value is.........")
    #print(Xtr_new)
    
    #Xte_new = Xte[:, 0:10]
    #print("Xte_new values is..........")
    #print(Xte_new)
    
    # Load json files :
    json_path = '/home/azureuser/cloudfiles/code/Users/ashutoshind2017/FIDDLE_experiments-master/mimic3_experiments/data/processed/output/'
    #json_path = '/mnt/disks/user/project/FIDDLE_experiments-master/mimic3_experiments/data/processed/output/'

    f1 = open (json_path + 'S.feature_names.json', "r")
    J1 = json.loads(f1.read())
    #print(len(J1))
    #print(J1)
    f2 = open (json_path + 'X.feature_names.json', "r")
    J2 = json.loads(f2.read())
    #print(len(J2))
    
    print("JSON read done !!!!")
    feature_names = [feature+'_'+str(i) for feature in J2 for i in range(48)] + J1
    print(len(feature_names))
    
    df_Xtr = pd.DataFrame(Xtr, columns = feature_names)
    df_Xte = pd.DataFrame(Xte, columns = feature_names)
    
   
    # RFClassifier based on best params above from CV Results  :
    rf = RandomForestClassifier(criterion ='entropy', n_estimators=413, max_depth= None, max_features=89, min_samples_split = 4, min_samples_leaf = 1, random_state = 0,
                                class_weight='balanced' )
    
    # Fit :                       
    #rf.fit(Xtr, ytr)
    rf.fit(df_Xtr, ytr)
    
    """
    # Make predictions for the test set
    y_pred_test = rf.predict(df_Xte)
    
    # Print accuracy for the training and test sets: 
    print("Accuracy on training set: {:.3f}".format(rf.score(df_Xtr, ytr)))
    print("Accuracy on test set: {:.3f}".format(rf.score(df_Xte, yte)))
    """
   
    ################################ Added SHAP explainataion ###############################
    print("Initialising the SHAP Tree explainer for RF... ")
    # Defining training sample :

    #sample_train = df_Xtr.iloc[0:1000, :]
    sample_train = df_Xtr
    #samples = df_Xte.iloc[0:100,:]  # Working 
    samples_test = df_Xte
    #samples = Xte[0:10, : ]
    
    print(sample_train.shape)
    print(samples_test.shape)
    
    print(type(sample_train))
    print(type(samples_test))
    
    #print(samples)
    
    print("Start explainer now......")
    
    import shap
    shap_values = shap.TreeExplainer(rf).shap_values(sample_train, approximate=True, check_additivity=False)
    
    #explainer = shap.TreeExplainer(rf)
    #shap_values = explainer.shap_values(samples, approximate=False, check_additivity=False)
    #shap_values = explainer.shap_values(samples, check_additivity=False)
    
    print("Shap values were calculated......")
    
    ##### SHAP Summary Plots :
    
    shap.initjs()
    f = plt.figure(figsize =(30,20))
    shap.summary_plot(shap_values, samples_test)
    #Save figures  :
    f.tight_layout()
    f.savefig(figure_path + 'Summary_Plot_SHAP_ipynb.png', dpi=600)
    
    ##### SHAP Dependence Plot with Mortality :

    # Feature1 : 
    f1 = plt.figure(figsize =(15,15))
    plt.title("Dependence-Plot Mediastinal Drainage with Mortality Outcome")
    shap.dependence_plot("226592_value_(-0.001, 20.0]_10", shap_values[1], sample_train)
    plt.tight_layout()
    f1.savefig(figure_path + 'SHAP_Dependenceplot_F1_ipynb.png', dpi=600)
    
    # Feature2 : 
    f2 = plt.figure(figsize =(15,15))
    plt.title("Dependence-Plot Granular Casts in Urine with Mortality Outcome")
    shap.dependence_plot("51479_value_(0.999, 3.0]_10", shap_values[1], sample_train)
    plt.tight_layout()
    f2.savefig(figure_path + 'SHAP_Dependenceplot_F2_ipynb.png', dpi=600)
    
    # Feature3 : 
    f3 = plt.figure(figsize =(15,15))
    plt.title("Dependence-Plot Amylase Body-Fluid with Mortality Outcome")
    shap.dependence_plot("51026_value_(17.8, 23.0]_25", shap_values[1], sample_train)
    plt.tight_layout()
    f3.savefig(figure_path + 'SHAP_Dependenceplot_F3_ipynb.png', dpi=600)
    
    # Feature4 : 
    f4 = plt.figure(figsize =(15,15))
    plt.title("Dependence-Plot Caspofungin Antibiotic Used with Mortality Outcome")
    shap.dependence_plot("225848_Dose_value_1.0_46", shap_values[1], sample_train)
    plt.tight_layout()
    f4.savefig(figure_path + 'SHAP_Dependenceplot_F4_ipynb.png', dpi=600)
    
    # Feature5 : 
    f5 = plt.figure(figsize =(15,15))
    plt.title("Dependence-Plot ICP Bolt Inserted for Monitoring with Mortality Outcome")
    shap.dependence_plot("226474_value_1.0_31", shap_values[1], sample_train)
    plt.tight_layout()
    f5.savefig(figure_path + 'SHAP_Dependenceplot_F5_ipynb.png', dpi=600)
    
    # Feature6 : 
    f6 = plt.figure(figsize =(15,15))
    f6.title("Dependence-Plot PBP(Prefilter) Replacement Rate(1800-2000ml/hr) with Mortality Outcome")
    shap.dependence_plot("228005_value_(1800.0, 2000.0]_30", shap_values[1], sample_train)
    f6.tight_layout()
    f6.savefig(figure_path + 'SHAP_Dependenceplot_F6_ipynb.png', dpi=600)
    
    # Feature7 : 
    f7 = plt.figure(figsize =(15,15))
    plt.title("Dependence-Plot Heparin Sodium Medication(1300-2000 Units) with Mortality Outcome")
    shap.dependence_plot("225152_Amount_value_(1300.0, 2000.0]_4", shap_values[1], sample_train)
    plt.tight_layout()
    f7.savefig(figure_path + 'SHAP_Dependenceplot_F7_ipynb.png', dpi=600)
    
    ##### SHAP Force Plot :
    shap.initjs()   
    
    def shap_plot(j):
        explainerModel = shap.TreeExplainer(rf)
        shap_values_Model = explainerModel.shap_values(sample_train, approximate=True, check_additivity=False)
        p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], sample_train.iloc[j,:])
        return(p)
        
    f8 = plt.figure(figsize =(15,4))
    plt.title("SHAP Force Plot for ICU Admission 1")
    shap_plot(0)
    plt.tight_layout()
    f8.savefig(figure_path + 'SHAP_Force_Plot_F8_ipynb.png', dpi=600)
    
    f9 = plt.figure(figsize =(15,4))
    plt.title("SHAP Force Plot for ICU Admission 100")
    shap_plot(100)
    plt.tight_layout()
    f9.savefig(figure_path + 'SHAP_Force_Plot_F9_ipynb.png', dpi=600)
    
    f10 = plt.figure(figsize =(15,4))
    plt.title("SHAP Force Plot for ICU Admission 1000")
    shap_plot(1000)
    plt.tight_layout()
    f10.savefig(figure_path + 'SHAP_Force_Plot_F10_ipynb.png', dpi=600)

    # Waterfall Plot :
    
    def make_shap_waterfall_plot(shap_values, features, num_display=20):
        column_list = features.columns
        feature_ratio = (np.abs(shap_values).sum(0) / np.abs(shap_values).sum()) * 100
        column_list = column_list[np.argsort(feature_ratio)[::-1]]
        feature_ratio_order = np.sort(feature_ratio)[::-1]
        cum_sum = np.cumsum(feature_ratio_order)
        column_list = column_list[:num_display]
        feature_ratio_order = feature_ratio_order[:num_display]
        cum_sum = cum_sum[:num_display]
    
        num_height = 0
        if (num_display >= 20) & (len(column_list) >= 20):
            num_height = (len(column_list) - 20) * 0.4
        
        fig, ax1 = plt.subplots(figsize=(8, 8 + num_height))
        ax1.plot(cum_sum[::-1], column_list[::-1], c='blue', marker='o')
        ax2 = ax1.twiny()
        ax2.barh(column_list[::-1], feature_ratio_order[::-1], alpha=0.6)
    
        ax1.grid(True)
        ax2.grid(False)
        ax1.set_xticks(np.arange(0, round(cum_sum.max(), -1)+1, 10))
        ax2.set_xticks(np.arange(0, round(feature_ratio_order.max(), -1)+1, 10))
        ax1.set_xlabel('Cumulative Ratio')
        ax2.set_xlabel('Composition Ratio')
        ax1.tick_params(axis="y", labelsize=13)
        plt.ylim(-1, len(column_list))
        plt.savefig(figure_path + 'SHAP_waterfall_plot.png', dpi=600)
    
    # SHAP waterfall plot
    make_shap_waterfall_plot(shap_values, samples_test)

    
    """
    
    # Make force_plot for test data : 
    print("Printing test data selected for the force plot.......... ")
    choosen_instance = Xte[700, :]
    print(choosen_instance)
    
    print(" Print the SHAP for choosen instance ............")
    shap_values = explainer.shap_values(choosen_instance, check_additivity=False)
    
    # 
    shap.initjs()

    f = plt.figure()
    shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)
    f.savefig(figure_path + 'SHAP_Force_Plot.png', dpi=600, show = False)
    
    #shap.summary_plot(shap_values[1], samples)
    print("Trying to print the summary plot now......")
    
    # SHAP Summary Plots :
    #fig1 = shap.summary_plot(shap_values, Xtr, plot_type="bar", show = False)
    #fig2 = shap.summary_plot(shap_values, Xtr , show = False)
    fig2 = shap.summary_plot(shap_values[1], samples, show = False)
    
    #Save figures  :
    fig2.save_to_file(figure_path + 'Fig2.html')
    #fig2.save_to_file(figure_path + 'Fig2.html')
    
    #Save figures  :
    #fig1.savefig(figure_path + 'SHAP_summary_plot1.png',dpi=600)
    #fig2.savefig(figure_path + 'SHAP_summary_plot2.png', bbox_inches='tight', dpi=600)
    
    #shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:],show=False,matplotlib=True).savefig('scratch.png')
     
    """
