import sys, os, time, pickle, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
with open('config.yaml') as f:
    config = yaml.load(f)
    
    
import json


# Ashutosh added new imports for explainataion:
import matplotlib.pyplot as plt
import plotly
import shap
import lime
import lime.lime_tabular

import torch
import json
########
## Constants

# Ashutosh added below "data_path":
data_path = config['data_path']

figure_path = '/home/azureuser/cloudfiles/code/Users/ashutoshind2017/FIDDLE_experiments-master/mimic3_experiments/figures'  # Azure 

# Ashutosh updated to use only Deep Learning models :
#model_names = config['model_names']

model_names = {
    'CNN': 'CNN_V3',
    'RNN': 'RNN_V2',
    #'LR': 'LR',
    #'RF': 'RF',
}

training_params = {'batch_size', 'lr'}

# Feature dimensions
dimensions = config['feature_dimension']

########

def main(task, T, dt, model_type):
    
    #print("Inside the main now... ")
    
    L_in = int(np.floor(T / dt))
    in_channels = dimensions[task][T]

    import lib.models as models
    model_name = model_names[model_type]
    ModelClass = getattr(models, model_name)
    df_search = pd.read_csv('./log/df_search.model={}.outcome={}.T={}.dt={}.csv'.format(model_name, task, T, dt))
    
    import lib.evaluate as evaluate
    best_model_info = evaluate.get_best_model_info(df_search)
    checkpoint, model = evaluate.load_best_model(best_model_info, ModelClass, in_channels, L_in, training_params)

    import lib.data as data
    if task == 'mortality':
        te_loader = data.get_benchmark_test(fuse=True)
    else:
        te_loader = data.get_test(task, duration=T, timestep=dt, fuse=True)
    
    y_true, y_score = evaluate.get_test_predictions(model, te_loader, '{}_T={}_dt={}'.format(task, T, dt), model_name)
    evaluate.save_test_predictions(y_true, y_score, task, T, dt, model_name)

    from sklearn import metrics, utils
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    fig = plt.figure(figsize=(5,5))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot([0,1], [0,1], ':')
    plt.plot(fpr, tpr, color='darkorange')
    plt.show()

    ## Bootstrapped 95% Confidence Interval
    # try:
    #     yte_pred = clf.decision_function(Xte)
    # except AttributeError:
    #     yte_pred = clf.predict_proba(Xte)[:,1]

    # Ashutosh corrected as per the update from author for using Parallel and delayed :
    #from sklearn.externals.joblib import Parallel, delayed
    from joblib import Parallel, delayed

    from tqdm import tqdm_notebook as tqdm
    def func(i):
        yte_true_b, yte_pred_b = utils.resample(y_true, y_score, replace=True, random_state=i)
        return metrics.roc_auc_score(yte_true_b, yte_pred_b)

    test_scores = Parallel(n_jobs=16)(delayed(func)(i) for i in tqdm(range(1000), leave=False))
    print('Test AUC: {:.3f} ({:.3f}, {:.3f})'.format(np.median(test_scores), np.percentile(test_scores, 2.5), np.percentile(test_scores, 97.5)))

    # idx = (np.abs(tpr - 0.5)).argmin()
    # y_pred = (y_score > thresholds[idx])
    # metrics.roc_auc_score(y_true, y_score)

    precision, recall, thresholds_ = metrics.precision_recall_curve(y_true, y_score)
    fig = plt.figure(figsize=(5,5))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(recall, precision, color='darkorange')
    plt.show()

    # target TPR = 50%
    idx = (np.abs(tpr - 0.5)).argmin()
    y_pred = (y_score > thresholds[idx])
    metrics.roc_auc_score(y_true, y_score)

    pd.DataFrame([{
        'tpr': tpr[idx],
        'fpr': fpr[idx],
        'ppv': metrics.precision_score(y_true, y_pred),
    }])
    
    # Ashutosh adding the explaination module here :
    
    print("Trying to interpret NOW... ")
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

    Xte1 = Xte
    Xtr1 = Xtr

    # Flatten time series features
    Xtr = Xtr.reshape(Xtr.shape[0], -1)
    Xte = Xte.reshape(Xte.shape[0], -1)

    # Combine time-invariant and time series
    Xtr = np.concatenate([Xtr, Str], axis=1)
    Xte = np.concatenate([Xte, Ste], axis=1)

    print("Printing information of Xtr , ytr etc now............")
    print(Xtr.shape, ytr.shape, Xte.shape, yte.shape)
    print("Printed .............................................")
    
    # Load json files :
    json_path = '/home/azureuser/cloudfiles/code/Users/ashutoshind2017/FIDDLE_experiments-master/mimic3_experiments/data/processed/output/'

    f1 = open (json_path + 'S.feature_names.json', "r")
    J1 = json.loads(f1.read())
    print(len(J1))
    print(J1)
    f2 = open (json_path + 'X.feature_names.json', "r")
    J2 = json.loads(f2.read())
    print(len(J2))
    
    print("JSON read done !!!!")

    feature_names = [feature+'_'+str(i) for feature in J2 for i in range(48)] + J1
    print(len(feature_names))
    
    # Print:
    print(feature_names[-100 : ])

    df_Xtr = pd.DataFrame(Xtr, columns = feature_names)
    df_Xre = pd.DataFrame(Xte, columns = feature_names)
    
    print(df_Xtr.shape)
    print("Print 2nd shape........")
    print(df_Xre.shape)
    
    # Creating a sub-set of dataset for testing :
    Xtr_new = df_Xtr[0:500]  # Only 1000 for testing now..

    #Xtr_new = df_Xtr

    #Xtr_new = Xtr[1, 0:10]
    print("Xtr_new value is.........")
    #print(Xtr_new)
    
    #Xte_new = Xte[:, 0:10]
    Xte_new = df_Xre
    print("Xte_new values is..........")
    #print(Xte_new)
    X_featurenames = feature_names
    
    Xtr1 = Xtr1[0:500, : , :]
    
    print("size of Xte1 and Xtr1 are .........................")
    print(Xte1.shape)
    print(Xtr1.shape)
    
    
    print("Initialising the LIME Explainer....")
    ## LIME Explaination Module :
    explainer = lime.lime_tabular.RecurrentTabularExplainer(Xtr1,
                    feature_names=X_featurenames, 
                    class_names=['0', '1'], 
                    verbose=True, 
                    mode='classification',
                    discretize_continuous=True)
    
    i = np.random.randint(0,Xte1.shape[0]) # what you want to explain 
    #j = np.random.randint(0,Xte1.shape[1])
    
    t = Xte1[i,:,:]
    print(t)
    print(t.shape)
    print("LIME in action now..........")

    # Convert N-D Array to pytorch tensor :
    Xte_new_tensor = torch.from_numpy(np.array(Xte_new))
    
    print("Checking transponse.....")
    Xte1 = Xte1.transpose(1,2)
    #x = x.transpose(1,2)

    #exp = explainer.explain_instance(Xte_new.iloc[i],model.predict,num_features=Xtr_new.shape[1],top_labels=None)
    exp = explainer.explain_instance(t ,model(Xte1), num_features=10)
    #exp = explainer.explain_instance(Xte_new.iloc[i] ,model(), num_features=10)
    
    print("Running the lime display.....")
    # Display :
    exp.show_in_notebook()
    exp.as_pyplot_figure()

    print("Saving the plots........")
    # Save :
    exp.save_to_file(figure_path + 'LIME.html')
    
    

    
     
# Ashutosh added function call to main () as the same was missing :
# Param : add CNN or RNN as applicable 

if __name__ == "__main__": 
    print("Main function is called ")
    # Ashutosh change the parameter as : CNN or RNN as needed :
    main('mortality', 48.0, 1.0, 'CNN')
else: 
    print("Main not called ")