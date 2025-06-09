# %% Train catboost
import os
curr_dir = '/home/padela/Scrivania/LLMs/gui_xrecs_presc_analytics'
os.chdir(curr_dir)

import catboost
import pandas as pd
import numpy as np
import json
import tqdm
import sys
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import pm4py

import utils.log_parsing as log_parsing

kpi = 'outcome_pred' #Can be either 'lead_time' or 'outcome_pred'
cb_loss = 'CrossEntropy' #  CrossEntropy' 
case_studies = ['bpi17']
samples = ['max']


def suppress_print():
    
    class SuppressPrint:
        def __enter__(self):
            self.original_stdout = sys.stdout  # Salva il riferimento originale di stdout
            sys.stdout = open(os.devnull, 'w')  # Reindirizza stdout a /dev/null
        
        def __exit__(self, exc_type, exc_value, traceback):
            sys.stdout.close()
            sys.stdout = self.original_stdout  # Ripristina stdout originale
    
    return SuppressPrint()

def fit_model(train_df, y, test_df, test_y):

        with suppress_print():
            categorical_features = train_df.select_dtypes(exclude=np.number).columns
            train_df[categorical_features] = train_df[categorical_features].astype(str)
            column_types = train_df.dtypes.astype(str).to_dict()
            
            params = {
                'depth': 8,
                'learning_rate': 0.01,
                'iterations': 5500,
                'early_stopping_rounds': 100,
                'thread_count': 4,
                'logging_level': 'Verbose',
                'task_type': "CPU",  # "GPU" if int(os.environ["USE_GPU"]) else "CPU"
                'loss_function': cb_loss
            }

            # print('Starting training...')
            # params["loss_function"] = "MAE"
            # print('The train and test shapes are ', train_df.shape, test_df.shape)
            train_data = Pool(train_df, y, cat_features=categorical_features.values)
            test_data = Pool(test_df, test_y, cat_features=categorical_features.values)

            if kpi == 'lead_time':
                model = CatBoostRegressor(**params)

            elif kpi == 'outcome_pred':
                model = CatBoostClassifier(**params)

            model.fit(train_data, verbose=False, plot=False, eval_set=(test_data), use_best_model=True)

        return model

remove_outliers = False
cf_preprocessing = 'aggr_hist' # hparams['cf_preprocessing']
if cf_preprocessing == 'sequential':
    raise ValueError('Catboost does not support sequential encoding')

for exp_name in case_studies:
    
    with open(f'hparams/{exp_name}.json') as f:
        hparams = json.load(f)
    df_train = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_train_{kpi}.csv')
    df_test = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_test_{kpi}.csv')

    if remove_outliers:
        df_train = log_parsing.remove_outliers_iqr(df_train, 'lead_time')
        df_test = log_parsing.remove_outliers_iqr(df_test, 'lead_time')
        # print('Outliers removed')

    for n_samples in samples:
        print('\n'*2)
        print('Case study is', exp_name, 'with samples', n_samples)
        lmae, lmape, lrmae, lcvae = [], [], [], []
        f_scores, precisions, recalls = [], [], []

        if n_samples == 'max':
            n_simulations = 1
        else:
            n_simulations = 10

        for seed in tqdm.tqdm(range(n_simulations)):
            try:
                
                # Import again the train
                df_train = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_train_{kpi}.csv')
                
                rseed = int(1618 + seed)
                try:    
                    df_train = df_train.sample(frac=1, random_state=rseed).reset_index(drop=True).iloc[:n_samples] 
                except: 
                    print('Sampling not done')
                    None        

                #Set the y
                if kpi == 'lead_time':
                    # COl to remove
                    act_to_encode = hparams["acts_not_freq"][0]

                    y_train = df_train['lead_time']
                    y_test = df_test['lead_time']

                    # #Remove from train 
                    X_train = df_train.drop(['lead_time'], axis=1)
                    X_test = df_test.drop(['lead_time'], axis=1)

                elif kpi == 'outcome_pred':
                    act_to_encode = hparams["acts_not_freq"][0]

                    y_train = df_train[f'occ_{act_to_encode}']
                    y_test = df_test[f'occ_{act_to_encode}']

                    # #Remove from train 
                    X_train = df_train.drop([f'occ_{act_to_encode}'], axis=1)
                    X_test = df_test.drop([f'occ_{act_to_encode}'], axis=1)                
                    X_train = X_train[X_test.columns]
                    X_test = X_test[X_train.columns]

                model = fit_model(X_train, y_train, X_test, y_test)
                y_pred = model.predict(X_test)

                #Evaluate MAE using sklearn
                from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error, mean_absolute_error, f1_score, precision_recall_fscore_support
                from utils.plot_stats import relative_mae

                # y_pred = [y_test.median() for i in range(len(y_test))]

                if kpi== 'lead_time':
                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    # print('The MAPE is ', round(mse, 2)) 

                    mae = mean_absolute_error(y_test, y_pred)
                    # print('The MAE is ', round(mae, 2))

                    #Same with median 
                    rmae = relative_mae(y_test, y_pred)
                    # print('The rMAE is ', round(rmae, 2))

                    errors = (y_pred - y_test)
                    cvae = np.std(errors)#/np.mean(y_test)

                    lmae.append(mae)
                    lmape.append(mape)
                    lrmae.append(rmae)
                    lcvae.append(cvae)

                elif kpi == 'outcome_pred':

                    precision, recall, f_score = precision_recall_fscore_support(y_test, y_pred, average=None)[:3]
                    print(f"Quello che ti serve ora è {f_score} - {precision} - {recall}")
                    f_scores.append(f_score)
                    precisions.append(precision)
                    recalls.append(recall)

            except:
                print('Error in the simulation\'s sampling due to similarity')

        #Print an empty line for separation
        # print('\n'*2)

        # Print the mean value of y_pred
        print(f'Using {n_samples} samples for the log {exp_name} and {n_simulations} simulations')
        print('The mean of y_pred is ', round(np.mean(y_pred), 2))
        # print(f' The median of y_pred is {round(np.median(y_pred), 2)}')
        # print(f' Test lenght is {len(df_test)}')
        # print(f' Train lenght is {len(df_train)}')
        # print(f' The log has {len(df_train["case:concept:name"].unique())} traces')
        # print('\n'*8)

        y_mean = np.full(len(y_test), np.mean(y_test))
        y_median = np.full(len(y_test), np.median(y_test))

        if kpi == 'lead_time':
            print('The means are ', round(np.mean(lmae), 2),'-',
                round(np.mean(lmape), 2),'-', round(np.mean(lrmae), 2))
            print('The stds are ', round(np.std(lmae), 2),'-',
                    round(np.std(lmape), 2),'-', round(np.std(lrmae), 2))

        if kpi == 'outcome_pred':
            print('For the case study ', exp_name, 'with samples ', n_samples, 'F1 is ', round(np.mean(f_scores), 2), '± ', round(np.std(f_scores), 2),
                'Precision is ', round(np.mean(precisions), 2), '± ', round(np.std(precisions), 2),
                'Recall is ', round(np.mean(recalls), 2), '± ', round(np.std(recalls), 2))

# %%
