# %% Train TabPFN
import os
curr_dir = '/home/padela/Scrivania/LLMs/gui_xrecs_presc_analytics' # '/home/padela/Desktop/LLMs_PM'
os.chdir(curr_dir)

from perpetual import PerpetualBooster
import pandas as pd
import numpy as np
import json
import tqdm
import sys
import pm4py
import random

kpi = 'lead_time' #Can be either 'lead_time' or 'outcome_pred'
case_studies = ['bpi12']
samples = ['max']
random.seed(1618)  # Set a random seed for reproducibility

def fit_model(train_df, y):
    """
    Fit PerpetualBooster model with validation-based budget selection.
    """
    
    # Split training data into train and validation sets (85-15 split)
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        train_df, y, test_size=0.15, random_state=42, stratify=None
    )
    
    # Budget range from 0.1 to 2.0 with step 0.1
    budget_range = [round(0.1 + i * 0.1, 1) for i in range(20)]  # [0.1, 0.2, ..., 2.0]
    best_budget = 1.0
    best_score = float('inf') if kpi == 'lead_time' else 0.0
    
    print(f"Testing {len(budget_range)} budget values from {min(budget_range)} to {max(budget_range)}")
    
    # Test different budgets on validation set
    for budget in budget_range:
        try:
            # Create PerpetualBooster model with current budget
            if kpi == 'lead_time':
                model = PerpetualBooster(objective="SquaredLoss", num_threads=4, budget=budget)
            elif kpi == 'outcome_pred':
                raise NotImplementedError('Outcome prediction not implemented for PerpetualBooster')
            
            # Fit the model on training split with current budget
            model.fit(X_train_split, y_train_split, budget=budget)
            
            # Predict on validation set
            y_val_pred = model.predict(X_val_split)
            
            # Calculate validation score
            if kpi == 'lead_time':
                from sklearn.metrics import mean_absolute_error
                val_score = mean_absolute_error(y_val_split, y_val_pred)
                if val_score < best_score:
                    best_score = val_score
                    best_budget = budget
                    print(f"  Budget {budget}: MAE = {val_score:.4f} (NEW BEST)")
                else:
                    print(f"  Budget {budget}: MAE = {val_score:.4f}")
                    
            elif kpi == 'outcome_pred':
                from sklearn.metrics import f1_score
                val_score = f1_score(y_val_split, y_val_pred, average='macro')
                if val_score > best_score:
                    best_score = val_score
                    best_budget = budget
                    print(f"  Budget {budget}: F1 = {val_score:.4f} (NEW BEST)")
                else:
                    print(f"  Budget {budget}: F1 = {val_score:.4f}")
                    
        except Exception as e:
            print(f"  Budget {budget}: Error - {str(e)}")
            continue
    
    print(f"Best budget selected: {best_budget} with validation score: {best_score:.4f}")
    
    # Train final model with best budget on full training set
    if kpi == 'lead_time':
        final_model = PerpetualBooster(objective="SquaredLoss", num_threads=4)
    elif kpi == 'outcome_pred':
        final_model = PerpetualBooster(objective="LogLoss", num_threads=4)
    
    # Fit the final model on full training data with best budget
    final_model.fit(train_df, y, budget=best_budget)
    
    # Store the best budget in the model for tracking
    final_model.best_budget = best_budget
    final_model.best_validation_score = best_score
    
    return final_model

remove_outliers = False
cf_preprocessing = 'aggr_hist' # hparams['cf_preprocessing']
if cf_preprocessing == 'sequential':
    raise ValueError('TabPFN does not support sequential encoding')

for exp_name in case_studies:
    
    with open(f'hparams/{exp_name}.json') as f:
        hparams = json.load(f)
    df_train = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_train_{kpi}.csv')
    df_test = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_test_{kpi}.csv')

    for n_samples in samples:
        print('\n'*2)
        print('Case study is', exp_name, 'with samples', n_samples)
        lmae, lmape, lrmae, lcvae = [], [], [], []
        f_scores, precisions, recalls = [], [], []
        best_budgets = []  # Track selected budgets

        if n_samples == 'max':
            n_simulations = 1
        else:
            n_simulations = 1

        for seed in tqdm.tqdm(range(n_simulations)):
            
                
            # Import again the train
            df_train = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_train_{kpi}.csv')
            
            # Only select the columns that are numerical
            selected_columns = [col for col in df_train.columns if pd.api.types.is_numeric_dtype(df_train[col])]
            df_test = df_test[selected_columns]
            df_train = df_train[selected_columns]
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
                print(f'The target activity is {act_to_encode}')

                y_train = df_train[f'occ_{act_to_encode}']
                y_test = df_test[f'occ_{act_to_encode}']

                # #Remove from train 
                X_train = df_train.drop([f'occ_{act_to_encode}'], axis=1)
                X_test = df_test.drop([f'occ_{act_to_encode}'], axis=1)                
                X_train = X_train[X_test.columns]
                X_test = X_test[X_train.columns]

            model = fit_model(X_train, y_train)
            
            # Track the best budget used
            best_budgets.append(model.best_budget)
            
            y_pred = model.predict(X_test)
            # print(f'y_pred is {y_pred}')
            # print(f'y_true is {y_test}')
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



        #Print an empty line for separation
        # print('\n'*2)

        # Print the mean value of y_pred
        print(f'Using {n_samples} samples for the log {exp_name} and {n_simulations} simulations')
        print('The mean of y_pred is ', round(np.mean(y_pred), 2))
        print('the mean of y_test is ', round(np.mean(y_test), 2))
        print(f'The mean of y_train is {round(np.mean(y_train), 2)}')

        # y_mean = np.full(len(y_test), np.mean(y_test))
        # y_median = np.full(len(y_test), np.median(y_test))

        if kpi == 'lead_time':
            print('The means are ', round(np.mean(lmae), 2))#,'-',
                # round(np.mean(lmape), 2),'-', round(np.mean(lrmae), 2))
            print('The stds are ', round(np.std(lmae), 2))#,'-',
                    # round(np.std(lmape), 2),'-', round(np.std(lrmae), 2))

        if kpi == 'outcome_pred':
            print('For the case study ', exp_name, 'with samples ', n_samples, 'F1 is ', round(np.mean(f_scores), 2), '± ', round(np.std(f_scores), 2),
                'Precision is ', round(np.mean(precisions), 2), '± ', round(np.std(precisions), 2),
                'Recall is ', round(np.mean(recalls), 2), '± ', round(np.std(recalls), 2))

# %%
