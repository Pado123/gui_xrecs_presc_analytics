# %% Train catboost
import os
os.chdir('..')

from tabpfn import TabPFNRegressor
import pandas as pd
import numpy as np
import json
import pm4py
import utils.log_parsing as log_parsing

n_examples = 'max'
remove_outliers = False

df_train = pd.read_csv('experiments/hospital/train_hospital_processed.csv')#.iloc[:n_examples]
df_test = pd.read_csv('experiments/hospital/test_hospital_processed.csv')

# Convert all 'object' and 'category' columns to numeric
for col in df_train.select_dtypes(['object', 'category']).columns:
    df_train[col] = pd.factorize(df_train[col])[0]  # Factorize encodes as integers
    df_test[col] = pd.factorize(df_test[col])[0]  # Can happen an error here, if the test set has new values or the train has values that are not in the test set

try: df_train = df_train.sample(frac=1, random_state=1618).reset_index(drop=True).iloc[:n_examples]
except: None

if remove_outliers:
    df_train = log_parsing.remove_outliers_iqr(df_train, 'lead_time')


def fit_model():
    try: del model
    except: None
    reg = TabPFNRegressor(random_state=42)
    reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)



# Remove train outliers in the y

model = fit_model(df_train.drop(['lead_time'], axis=1), df_train['lead_time'], df_test.drop(['lead_time'], axis=1), df_test['lead_time'])

# #Set y as "lead_time"
y_train = df_train['lead_time']
y_test = df_test['lead_time']

# #Remove from train 
X_train = df_train.drop(['lead_time'], axis=1)
X_test = df_test.drop(['lead_time'], axis=1)

# Predict the value of y
print('Predict Catboost')
y_pred = model.predict(X_test)

# Print the mean for the y_pred
print('The mean of y_true is ', np.mean(y_test))

#Evaluate MAE using sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
mae = mean_absolute_error(y_test, y_pred)
# print('The MAE for mean is ', mae) 

#Same with median 
mape = mean_absolute_percentage_error(y_test, y_pred)
# print('The MAE fom median is ', mse)
rmae = mae/np.mean(y_test)
print('The relative mae is', rmae)

print(f'The MAE, MAPE and RMAE are {round(mae, 2)} - {round(mape, 2)} - {round(rmae, 2)} respectively')

print(f' Train lenght is {len(df_train)}')

#Plot the errors
import matplotlib.pyplot as plt
plt.plot(y_pred - y_test, 'o')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title(f'Predicted vs True for {n_examples} examples')

# Plot the y-distribution
plt.figure()
plt.hist(y_test, bins=20, alpha=0.5, label='Test', density=True)
plt.hist(y_train, bins=20, alpha=0.5, label='Train', density=True)
plt.hist(y_pred, bins=20, alpha=0.5, label='Predicted', density=True)
plt.legend()
plt.title(f'True vs Predicted for {n_examples} examples {"" if remove_outliers else "without"} outliers')

# %%
