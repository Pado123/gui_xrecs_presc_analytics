# %% Train catboost
import os
import catboost
import pandas as pd
import numpy as np
import json
from catboost import CatBoostRegressor, Pool
import pm4py
from tabpfn import TabPFNRegressor

os.chdir('/home/padela/Scrivania/LLMs/gui_xrecs_presc_analytics')
# import ipdb; ipdb.set_trace()
import utils.log_parsing as log_parsing


# Set the exp_name 
exp_name = 'bpi12'
print(f"Experiment name set to: {exp_name}")

print(f"Loading hyperparameters from 'hparams/{exp_name}.json'")
with open(f'hparams/{exp_name}.json') as f:
    hparams = json.load(f)
print("Hyperparameters loaded successfully.")

remove_outliers = False
n_samples = 50
n_simulations = 10
cf_preprocessing = hparams['cf_preprocessing']
if cf_preprocessing == 'sequential':
    raise ValueError('Catboost does not support sequential encoding')

df_train = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_train.csv')
df_test = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_test.csv')

if remove_outliers:
    df_train = log_parsing.remove_outliers_iqr(df_train, 'lead_time')
    df_test = log_parsing.remove_outliers_iqr(df_test, 'lead_time')
    print('Outliers removed')

if n_samples == 500:
    print(f'The ratio of 500/len is {500/len(df_train)}')
lmae, lmape, lrmae, lcvae = [], [], [], []

def fit_model(train_df, y):

    reg = TabPFNRegressor(random_state=1618)
    reg.fit(train_df, y, )
    return reg

for seed in range(n_simulations):

    # Import again the train
    df_train = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_train.csv')
    
    rseed = int(1618 + seed)
    try:    
        df_train = df_train.sample(frac=1, random_state=rseed).reset_index(drop=True).iloc[:n_samples] 
    except: 
        print('Sampling not done')
        None        

    # #Set y as "lead_time"
    y_train = df_train['lead_time']
    y_test = df_test['lead_time']

    # #Remove from train 
    X_train = df_train.drop(['lead_time'], axis=1)
    X_test = df_test.drop(['lead_time'], axis=1)

    model = fit_model(df_train.drop(['lead_time'], axis=1), 
                df_train['lead_time'])

    # Predict the value of y n_simulations
    # print('Predict Catboost')
    y_pred = model.predict(X_test)

    # Print the mean for the y_pred
    # print('The mean of y_true is ', round(np.mean(y_test), 2))

    #Evaluate MAE using sklearn
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.metrics import median_absolute_error
    from sklearn.metrics import mean_absolute_error
    from utils.plot_stats import relative_mae

    # y_pred = [y_test.median() for i in range(len(y_test))]

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
    print(f'{mae} - {mape} - {rmae}')
    print(f'Iteration {seed+1} completed')

#Print an empty line for 8 lines
print('\n'*8)

# Print the mean value of y_pred
print(f'Using {n_samples} samples')
print('The mean of y_pred is ', round(np.mean(y_pred), 2))
print(f' The median of y_pred is {round(np.median(y_pred), 2)}')
print(f' Test lenght is {len(df_test)}')
print(f' Train lenght is {len(df_train)}')
print('\n'*8)

y_mean = np.full(len(y_test), np.mean(y_test))
y_median = np.full(len(y_test), np.median(y_test))
print(f'The benchmark value for mean are','-', round(mean_absolute_percentage_error(y_test, y_mean), 2),
      '-', round(mean_absolute_error(y_test, y_mean), 2), '-', round(relative_mae(y_test, y_mean), 2))

print(f'The benchmark value for median are','-', round(mean_absolute_percentage_error(y_test, y_median), 2),
        '-', round(mean_absolute_error(y_test, y_median), 2), '-', round(relative_mae(y_test, y_median), 2))



print('\n'*8)

print('The means are ', round(np.mean(lmae), 2),'-',
       round(np.mean(lmape), 2),'-', round(np.mean(lrmae), 2))
print('The stds are ', round(np.std(lmae), 2),'-',
         round(np.std(lmape), 2),'-', round(np.std(lrmae), 2))




# # #Plot the errors
# import matplotlib.pyplot as plt

# #Plot the mean and the median of the errors as line
# errors = (y_pred - y_test)/y_test.mean()
# plt.plot(y_test, errors, 'o')
# plt.hlines(np.mean(errors), xmax=np.max(y_test), xmin=0, colors='r', label='Mean')
# plt.hlines(np.median(errors), xmax=np.max(y_test), xmin=0, colors='b', label='Median')
# plt.xlabel('True values')
# plt.ylabel('Predicted values')
# plt.title(f'Predicted vs True for {n_samples} examples')

# # Plot the y-distribution
# plt.figure()
# plt.hist(y_test, bins=50, alpha=0.5, label='Test', density=True)
# plt.hist(y_train, bins=50, alpha=0.5, label='Train', density=True)
# plt.hist(y_pred, bins=50, alpha=0.5, label='Predicted', density=True)
# plt.legend()
# plt.title(f'True vs Predicted for {n_samples} examples {"" if remove_outliers else "without"} outliers')




# %%


