# %% Train KNN
import os
import sys
curr_dir = '/home/padela/Scrivania/LLMs/gui_xrecs_presc_analytics' # '/home/padela/Desktop/LLMs_PM'
os.chdir(curr_dir)

import pandas as pd
import numpy as np
import json
import tqdm
import sys
import pm4py
import random
from utils.select_columns import select_columns as sc
from utils.path_predictor import filter_and_compute_mean

kpi = 'lead_time' #Can be either 'lead_time' or 'outcome_pred'
case_studies = ['bpi12']
samples = [10]
random.seed(1618)  
cf_preprocessing = 'aggr_hist' 

for exp_name in case_studies:

    with open(f'hparams/{exp_name}.json') as f:
        hparams = json.load(f)
    df_test = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_test_{kpi}.csv')

    for n_samples in samples:
        print('\n'*2)
        print('Case study is', exp_name, 'with samples', n_samples)
        lmae = []
        no_rows_counts = []
        f_scores, precisions, recalls = [], [], []

        if n_samples == 'max':
            n_simulations = 1
        else:
            n_simulations = 50

        for seed in tqdm.tqdm(range(n_simulations)):
            try:
                rseed = int(1618 + seed)
                try:    
                    df_train = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_train_{kpi}.csv')
                    df_train = df_train.sample(frac=1, random_state=rseed).reset_index(drop=True).iloc[:n_samples] 
                except: 
                    print('Sampling not done')

                # Select the columns plus the target column, that is the last one
                selected_columns = sc(hparams, df_train) + [df_train.columns[-1]]
                df_train = df_train[selected_columns]
                df_test = df_test[selected_columns]
                
                #Set the y
                if kpi == 'lead_time':

                    y_train = df_train['lead_time']
                    y_test = df_test['lead_time']

                    # #Remove from train 
                    X_train = df_train.drop(['lead_time'], axis=1)
                    X_test = df_test.drop(['lead_time'], axis=1)

                elif kpi == 'outcome_pred':
                    act_to_encode = hparams["acts_not_freq"][0]
                    print(f'The target activity is {act_to_encode}')

                    y_train = df_train[f'occ_{act_to_encode}']
                    y_test = df_test[f'occ_{act_to_encode}']

                    # #Remove from train 
                    X_train = df_train.drop([f'occ_{act_to_encode}'], axis=1)
                    X_test = df_test.drop([f'occ_{act_to_encode}'], axis=1)                
                    X_train = X_train[X_test.columns]
                    X_test = X_test[X_train.columns]

                # For each row in X_test, provide a prediction using the filter_and_compute_mean function, store it in a y_pred list
                y_pred = []
                no_rows_count = 0
                for _, row in tqdm.tqdm(X_test.iterrows()):
                    # Create a row that includes the target column for the function
                    if kpi == 'lead_time':
                        # For lead_time, we need to add the target column back to the row
                        row_with_target = row.copy()
                        prediction, no_rows_flag = filter_and_compute_mean(df_train, row_with_target, mode='median')
                        # print(f'The prediction is {prediction} and the no_rows_flag is {no_rows_flag}')
                    elif kpi == 'outcome_pred':
                        # For outcome_pred, we need to add the target column back to the row
                        row_with_target = row.copy()
                        # We'll use the mean of the training target as a placeholder since we don't have the actual target
                        row_with_target[f'occ_{act_to_encode}'] = y_train.mean()
                        prediction, no_rows_flag = filter_and_compute_mean(df_train, row_with_target)
                    
                    y_pred.append(prediction)
                    no_rows_count += no_rows_flag
                
                #Evaluate MAE using sklearn
                from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error, mean_absolute_error, f1_score, precision_recall_fscore_support
                from utils.plot_stats import relative_mae

                # y_pred = [y_test.median() for i in range(len(y_test))]

                if kpi== 'lead_time':
                    # mape = mean_absolute_percentage_error(y_test, y_pred)
                    # print('The MAPE is ', round(mse, 2)) 

                    mae = mean_absolute_error(y_test, y_pred)
                    # print('The MAE is ', round(mae, 2))

                    #Same with median 
                    # rmae = relative_mae(y_test, y_pred)
                    # print('The rMAE is ', round(rmae, 2))

                    errors = (y_pred - y_test)
                    cvae = np.std(errors)#/np.mean(y_test)

                    lmae.append(mae)
                    print('The mean of y_pred is ', round(np.mean(y_pred), 2))
                    # lmape.append(mape)
                    # lrmae.append(rmae)
                    # lcvae.append(cvae)

                elif kpi == 'outcome_pred':

                    precision, recall, f_score = precision_recall_fscore_support(y_test, y_pred, average='macro')[:3]
                    print(f"Quello che ti serve ora è {f_score} - {precision} - {recall}")
                    precision, recall, f_score = precision_recall_fscore_support(y_test, y_pred, average='macro')[:3]
                    print(f"Quello che ti serve ora è {f_score} - {precision} - {recall}")
                    f_scores.append(f_score)
                    precisions.append(precision)
                    recalls.append(recall)

                no_rows_counts.append(no_rows_count)
            except:
                print('Error in the simulation\'s sampling due to similarity')

        #Print an empty line for separation
        # print('\n'*2)

        # Print the mean value of y_pred
        print(f'Using {n_samples} samples for the log {exp_name} and {n_simulations} simulations')
        
        print('the mean of y_test is ', round(np.mean(y_test), 2))
        print(f'The mean of y_train is {round(np.mean(y_train), 2)}')
        # print(f' The median of y_pred is {round(np.median(y_pred), 2)}')
        # print(f' Test lenght is {len(df_test)}')
        # print(f' Train lenght is {len(df_train)}')
        # print(f' The log has {len(df_train["case:concept:name"].unique())} traces')
        # print('\n'*8)

        y_mean = np.full(len(y_test), np.mean(y_test))
        y_median = np.full(len(y_test), np.median(y_test))

        if kpi == 'lead_time':
            print('The mean mae for case study:', exp_name, 'with samples:', n_samples, 'is:', round(np.mean(lmae), 2))
            print('With std ', round(np.std(lmae), 2))
            print('The number of rows that had no rows meet the criteria is ', np.mean(no_rows_counts), 'with std ', round(np.std(no_rows_counts), 2))

        if kpi == 'outcome_pred':
            print('For the case study ', exp_name, 'with samples ', n_samples, 'F1 is ', round(np.mean(f_scores), 2), '± ', round(np.std(f_scores), 2),
                'Precision is ', round(np.mean(precisions), 2), '± ', round(np.std(precisions), 2),
                'Recall is ', round(np.mean(recalls), 2), '± ', round(np.std(recalls), 2))
            print('The number of rows that had no rows meet the criteria is ', np.mean(no_rows_counts), 'with std ', round(np.std(no_rows_counts), 2))

print('train shape was ', X_train.shape)
print('test shape was ', X_test.shape)
# %%
