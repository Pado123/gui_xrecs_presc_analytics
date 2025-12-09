# %% Train KNN
import os
import sys
curr_dir = '/home/padela/Desktop/LLMs_PM' # '/home/padela/Scrivania/LLMs/gui_xrecs_presc_analytics' # '/home/padela/Desktop/LLMs_PM'
os.chdir(curr_dir)

# Add the current directory to Python path
sys.path.append(curr_dir)

from utils.knn_predictors import KNNPredictor

import pandas as pd
import numpy as np
import json
import tqdm
import sys
import pm4py
import random
from utils.select_columns import select_columns as sc

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

kpi = 'lead_time' #Can be either 'lead_time' or 'outcome_pred'
case_studies = ['bac', 'hospital']
samples = ['max']
random.seed(1618)  #
cf_preprocessing = 'aggr_hist' # hz

def fit_model(train_df, y, hparams):
    """
    Fit KNN model using the KNNPredictor class.
    Supports mixed-type inputs (categorical and numerical).
    """
    # Determine task type based on kpi
    task_type = 'regression' if kpi == 'lead_time' else 'classification'

    # Create KNN predictor with support for mixed types
    # The predictor will automatically detect categorical vs numerical columns
    knn_predictor = KNNPredictor(
        task_type=task_type, 
        random_state=42,
        categorical_encoding='onehot',  # Use one-hot encoding for categorical variables
        max_categories=20  # Columns with <=20 unique values might be treated as categorical
    )

    # Prepare training dataframe with target column
    train_df_with_target = train_df.copy()
    train_df_with_target['target'] = y
    
    # Get feature columns (all columns except target)
    feature_columns = [col for col in train_df.columns]
    
    # Ensure proper data types for mixed-type handling
    # Convert object columns to string to avoid issues
    for col in train_df_with_target[feature_columns].select_dtypes(include=['object']).columns:
        train_df_with_target[col] = train_df_with_target[col].astype(str)

    # Train the model
    results = knn_predictor.train(
        df=train_df_with_target,
        feature_columns=feature_columns,
        target_column='target',
        cv_folds=5,
        k_values= [1, 3, 5, 7, 9, 11, 15, 19, 25]  # KNN specific parameters [1]
    )
    
    # # Log detected column types (optional, for debugging)
    # if hasattr(knn_predictor, 'detected_categorical_columns') and knn_predictor.detected_categorical_columns:
    #     print(f"Detected {len(knn_predictor.detected_categorical_columns)} categorical columns")
    # if hasattr(knn_predictor, 'detected_numerical_columns') and knn_predictor.detected_numerical_columns:
    #     print(f"Detected {len(knn_predictor.detected_numerical_columns)} numerical columns")

    return knn_predictor


for exp_name in case_studies:
    
    with open(f'hparams/{exp_name}.json') as f:
        hparams = json.load(f)
    df_test = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_test_{kpi}.csv')

    # Select the columns plus the target column, that is the last one
    selected_cols = sc(hparams, df_test)
    if selected_cols is None:
        # If None, use all columns (select_columns returns None for "pre" == "all")
        selected_columns = list(df_test.columns)
    else:
        # Add target column if not already in the list
        target_col_name = df_test.columns[-1]
        if target_col_name not in selected_cols:
            selected_columns = selected_cols + [target_col_name]
        else:
            selected_columns = selected_cols
    
    # Apply column selection to df_test
    df_test = df_test[selected_columns]
    
    # Remove 'case:concept:name' if it exists (not needed for training)
    if 'case:concept:name' in df_test.columns:
        df_test = df_test.drop(['case:concept:name'], axis=1)
        selected_columns = [col for col in selected_columns if col != 'case:concept:name']

    for n_samples in samples:
        print('\n'*2)
        print('Case study is', exp_name, 'with samples', n_samples)
        lmae = []
        f_scores, precisions, recalls = [], [], []

        if n_samples == 'max':
            n_simulations = 1
        else:
            n_simulations = 40

        # Initialize variables for final reporting
        y_pred = None
        y_test = None
        y_train = None
        X_train = None
        X_test = None
        model = None

        for seed in tqdm.tqdm(range(n_simulations)):

            df_train = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_train_{kpi}.csv')
            df_train = df_train[selected_columns]
            # Remove 'case:concept:name' if it exists (not needed for training)
            if 'case:concept:name' in df_train.columns:
                df_train = df_train.drop(['case:concept:name'], axis=1)
            rseed = int(1618 + seed)
            try:    
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
                # X_test = X_test[X_train.columns]
        
                # Ensure y_train and y_test are Series (not DataFrame)
                if isinstance(y_train, pd.DataFrame):
                    y_train = y_train.iloc[:, 0]
                if isinstance(y_test, pd.DataFrame):
                    y_test = y_test.iloc[:, 0]

            # Ensure column alignment between train and test
            common_cols = [col for col in X_train.columns if col in X_test.columns]
            X_train = X_train[common_cols]
            X_test = X_test[common_cols]
            
            # Ensure X_test has the same column order as X_train
            X_test = X_test[X_train.columns]
            
            # Handle NaN values for mixed-type data before training
            # Fill numerical columns with median, categorical with mode
            for col in X_train.columns:
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    # Numerical column: fill with median
                    median_val = X_train[col].median()
                    if pd.isna(median_val):
                        # If all values are NaN, fill with 0
                        X_train[col] = X_train[col].fillna(0)
                        X_test[col] = X_test[col].fillna(0)
                    else:
                        X_train[col] = X_train[col].fillna(median_val)
                        X_test[col] = X_test[col].fillna(median_val)
                else:
                    # Categorical/object column: fill with mode or 'missing'
                    mode_val = X_train[col].mode()
                    if len(mode_val) > 0:
                        fill_val = mode_val[0]
                    else:
                        fill_val = 'missing'
                    X_train[col] = X_train[col].fillna(fill_val)
                    X_test[col] = X_test[col].fillna(fill_val)
            
            # Convert object columns to string for consistent handling
            for col in X_train.select_dtypes(include=['object']).columns:
                X_train[col] = X_train[col].astype(str)
                X_test[col] = X_test[col].astype(str)

            # print('the columns of X_train are ', X_train.columns.tolist())
            # print('the columns of X_test are ', X_test.columns.tolist())
            # print(f'X_train dtypes: {X_train.dtypes.value_counts().to_dict()}')
            # print(f'X_test dtypes: {X_test.dtypes.value_counts().to_dict()}')

            model = fit_model(X_train, y_train, hparams)
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
                f_scores.append(f_score)
                precisions.append(precision)
                recalls.append(recall)

        #Print an empty line for separation
        # print('\n'*2)

        # Print the mean value of y_pred
        print(f'Using {n_samples} samples for the log {exp_name} and {n_simulations} simulations')
        if y_pred is not None:
            print('The mean of y_pred is ', round(np.mean(y_pred), 2), 'and the median is ', round(np.median(y_pred), 2))
        if y_test is not None:
            print('the mean of y_test is ', round(np.mean(y_test), 2), 'and the median is ', round(np.median(y_test), 2))
        if y_train is not None:
            print(f'The mean of y_train is {round(np.mean(y_train), 2)} and the median is {round(np.median(y_train), 2)}')
        # print(f' The median of y_pred is {round(np.median(y_pred), 2)}')
        # print(f' Test lenght is {len(df_test)}')
        # print(f' Train lenght is {len(df_train)}')
        # print(f' The log has {len(df_train["case:concept:name"].unique())} traces')
        # print('\n'*8)

        if kpi == 'lead_time':
            print('The mean mae for case study:', exp_name, 'with samples:', n_samples, 'is:', round(np.mean(lmae), 2), '±', round(np.std(lmae)))

        if kpi == 'outcome_pred':
            print('For the case study ', exp_name, 'with samples ', n_samples, 'F1 is ', round(np.mean(f_scores), 2), '± ', round(np.std(f_scores)),
                'Precision is ', round(np.mean(precisions), 2), '± ', round(np.std(precisions), 2),
                'Recall is ', round(np.mean(recalls), 2), '± ', round(np.std(recalls), 2))

# print('train shape was ', X_train.shape, 'train columns were ', X_train.columns.tolist(),'and they were equal to the test columns? ', (X_train.columns==X_test.columns).all())

if model is not None:
    del model
    print('model deleted')
    print('the pre was', hparams['pre'])
# %%
s