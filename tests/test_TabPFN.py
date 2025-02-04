# %% Train TabPFN
import os
os.chdir('..')

import pandas as pd
import numpy as np
import json
from tabpfn import TabPFNRegressor
import utils.log_parsing as log_parsing
import numpy as np

# Set the exp_name 
exp_name = 'bpi12'
print(f"Experiment name set to: {exp_name}")

print(f"Loading hyperparameters from 'hparams/{exp_name}.json'")
with open(f'hparams/{exp_name}.json') as f:
    hparams = json.load(f)
print("Hyperparameters loaded successfully.")

remove_outliers = False
n_samples = 'max'
cf_preprocessing = hparams['cf_preprocessing']
if cf_preprocessing == 'sequential':
    raise ValueError('Catboost does not support sequential encoding')

df_train = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_train.csv')
df_test = pd.read_csv(f'experiments/{exp_name}/preprocessed_log_{cf_preprocessing}_test.csv')

if remove_outliers:
    df_train = log_parsing.remove_outliers_iqr(df_train, 'lead_time')
    df_test = log_parsing.remove_outliers_iqr(df_test, 'lead_time')
    print('Outliers removed')


try:    
    df_train = df_train.sample(frac=1, random_state=1625).reset_index(drop=True).iloc[:n_samples] 
    print(f'Using {n_samples} samples')
except: None

# test = df_test.sample(frac=1, random_state=1618).reset_index(drop=True).iloc[20:40]

for col in df_train.select_dtypes(['object', 'category']).columns:
    df_train[col] = pd.factorize(df_train[col])[0]

for col in df_test.select_dtypes(['object', 'category']).columns:
    df_test[col] = pd.factorize(df_test[col])[0]

def fit_model(train_df, y):

    reg = TabPFNRegressor(random_state=1618)
    reg.fit(train_df, y)
    return reg

#Set y as "lead_time"
y_train = df_train['lead_time']
y_test = df_test['lead_time']

# #Remove from train 
X_train = df_train.drop(['lead_time'], axis=1)
X_test = df_test.drop(['lead_time'], axis=1)

reg = fit_model(X_train, y_train)

# Predict the value of y
print('Predict Catboost')
y_pred = reg.predict(X_test)

# Print the mean for the y_pred
print('The mean of y_true is ', round(np.mean(y_test), 2))

#Evaluate MAE using sklearn
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error
from utils.plot_stats import relative_mae

# y_pred = [y_test.median() for i in range(len(y_test))]

# Print the mean value of y_pred
print('The mean of y_pred is ', round(np.mean(y_pred), 2))

mse = mean_absolute_percentage_error(y_test, y_pred)
print('The MAPE is ', round(mse, 2)) 

mae = mean_absolute_error(y_test, y_pred)
print('The MAE is ', round(mae, 2))

#Same with median 
rmae = relative_mae(y_test, y_pred)
print('The rMAE is ', round(rmae, 2))

print('MAE, MAPE and rMAE calculated, they are ', round(mae, 2),'-', 
      round(mse, 2),'-', round(rmae, 2))

# Print df_train lenght
print(f' Train lenght is {len(df_train)}')

#Plot the errors
import matplotlib.pyplot as plt
plt.plot(y_pred - y_test, 'o')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title(f'Predicted vs True for {n_samples} examples')

# Plot the y-distribution
plt.figure()
plt.hist(y_test, bins=50, alpha=0.5, label='Test', density=True)
plt.hist(y_train, bins=50, alpha=0.5, label='Train', density=True)
plt.hist(y_pred, bins=50, alpha=0.5, label='Predicted', density=True)
plt.legend()
plt.title(f'True vs Predicted for {n_samples} examples {"" if remove_outliers else "without"} outliers')

# %%
