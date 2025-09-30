# %% Train Linear Regression
import os
import sys
curr_dir = '/home/padela/Scrivania/LLMs/gui_xrecs_presc_analytics' # '/home/padela/Desktop/LLMs_PM'
os.chdir(curr_dir)

# Add the current directory to Python path
sys.path.append(curr_dir)

from utils.linear_regression_predictor import LinearRegressionPredictor

import pandas as pd
import numpy as np
import json
import tqdm
import sys
import pm4py
import random
from utils.select_columns import select_columns as sc

kpi = 'lead_time' #Can be either 'lead_time' or 'outcome_pred'
case_studies = ['bpi12']
samples = [100]
random.seed(1618)  #
cf_preprocessing = 'aggr_hist' # hz

def fit_model(train_df, y, hparams):
    """
    Fit Linear Regression model using the LinearRegressionPredictor class.
    """
    # Determine task type based on kpi
    task_type = 'regression' if kpi == 'lead_time' else 'classification'

    # Create Linear Regression predictor
    lr_predictor = LinearRegressionPredictor(task_type=task_type, random_state=42)

    # Prepare training dataframe with target column
    train_df_with_target = train_df.copy()
    train_df_with_target['target'] = y
    
    # Get feature columns (all columns except target)
    feature_columns = [col for col in train_df.columns]

    # Train the model
    results = lr_predictor.train(
        df=train_df_with_target,
        feature_columns=feature_columns,
        target_column='target',
        cv_folds=5
    )

    return lr_predictor


for exp_name in case_studies:
    
    with open(f'hparams/{exp_name}.json') as f:
        hparams = json.load(f)
    df_test = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_test_{kpi}.csv')

    # Select the columns plus the target column, that is the last one
    selected_columns = sc(hparams, df_test) + [df_test.columns[-1]]
    df_test = df_test[selected_columns]

    for n_samples in samples:
        print('\n'*2)
        print('Case study is', exp_name, 'with samples', n_samples)
        lmae, lmae_mean_baseline, lmae_median_baseline = [], [], []
        f_scores, precisions, recalls = [], [], []
        
        # Initialize variables to avoid NameError
        y_pred = []
        y_test = []
        y_train = []

        if n_samples == 'max':
            n_simulations = 1
        else:
            n_simulations = 200

        for seed in tqdm.tqdm(range(n_simulations)):
            try:                
                df_train = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_train_{kpi}.csv')
                df_train = df_train[selected_columns]
                rseed = int(1618 + seed)
                try:    
                    if n_samples != 'max':
                        df_train = df_train.sample(frac=1, random_state=rseed).reset_index(drop=True).iloc[:n_samples] 
                    
                except: 
                    print('Sampling not done')

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

                model = fit_model(X_train, y_train, hparams)
                y_pred = model.predict(X_test)
                
                # Create baseline predictions using mean and median of training set
                y_pred_mean_baseline = np.full(len(y_test), y_train.mean())
                y_pred_median_baseline = np.full(len(y_test), y_train.median())
                
                # print(f'y_pred is {y_pred}')
                # print(f'y_true is {y_test}')
                #Evaluate MAE using sklearn
                from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error, mean_absolute_error, f1_score, precision_recall_fscore_support
                from utils.plot_stats import relative_mae

                # y_pred = [y_test.median() for i in range(len(y_test))]

                if kpi== 'lead_time':
                    # mape = mean_absolute_percentage_error(y_test, y_pred)
                    # print('The MAPE is ', round(mse, 2)) 

                    # Linear Regression MAE
                    mae = mean_absolute_error(y_test, y_pred)
                    # print('The MAE is ', round(mae, 2))

                    # Baseline MAEs
                    mae_mean_baseline = mean_absolute_error(y_test, y_pred_mean_baseline)
                    mae_median_baseline = mean_absolute_error(y_test, y_pred_median_baseline)

                    #Same with median 
                    # rmae = relative_mae(y_test, y_pred)
                    # print('The rMAE is ', round(rmae, 2))

                    errors = (y_pred - y_test)
                    cvae = np.std(errors)#/np.mean(y_test)

                    lmae.append(mae)
                    lmae_mean_baseline.append(mae_mean_baseline)
                    lmae_median_baseline.append(mae_median_baseline)
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

            except:
                print('Error in the simulation\'s sampling due to similarity')

        #Print an empty line for separation
        # print('\n'*2)

        # Print the mean value of y_pred
        print(f'Using {n_samples} samples for the log {exp_name} and {n_simulations} simulations')
        if len(y_pred) > 0:
            print('The mean of y_pred is ', round(np.mean(y_pred), 2), 'and the median is ', round(np.median(y_pred), 2))
            print('the mean of y_test is ', round(np.mean(y_test), 2), 'and the median is ', round(np.median(y_test), 2))
            print(f'The mean of y_train is {round(np.mean(y_train), 2)} and the median is {round(np.median(y_train), 2)}')
        else:
            print('No successful simulations completed')
        # print(f' The median of y_pred is {round(np.median(y_pred), 2)}')
        # print(f' Test lenght is {len(df_test)}')
        # print(f' Train lenght is {len(df_train)}')
        # print(f' The log has {len(df_train["case:concept:name"].unique())} traces')
        # print('\n'*8)

        if kpi == 'lead_time' and len(lmae) > 0:
            print('The mean mae for case study:', exp_name, 'with samples:', n_samples, 'is:', round(np.mean(lmae), 2))
            print('With std ', round(np.std(lmae), 2))
            
            # Report baseline MAEs
            print('Baseline MAE (using mean of train set):', round(np.mean(lmae_mean_baseline), 2), '±', round(np.std(lmae_mean_baseline), 2))
            print('Baseline MAE (using median of train set):', round(np.mean(lmae_median_baseline), 2), '±', round(np.std(lmae_median_baseline), 2))
            
            # Calculate average of both baselines
            avg_baseline_mae = (np.mean(lmae_mean_baseline) + np.mean(lmae_median_baseline)) / 2
            print('Average baseline MAE (mean + median):', round(avg_baseline_mae, 2))

        if kpi == 'outcome_pred' and len(f_scores) > 0:
            print('For the case study ', exp_name, 'with samples ', n_samples, 'F1 is ', round(np.mean(f_scores), 2), '± ', round(np.std(f_scores), 2),
                'Precision is ', round(np.mean(precisions), 2), '± ', round(np.std(precisions), 2),
                'Recall is ', round(np.mean(recalls), 2), '± ', round(np.std(recalls), 2))

# Print shapes if variables are defined
try:
    print('train shape was ', X_train.shape)
    print('test shape was ', X_test.shape)
    del model
    print('model deleted')
except NameError:
    print('Variables not defined due to simulation errors')
# %%
