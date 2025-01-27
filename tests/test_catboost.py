# %% Train catboost
import os
os.chdir('..')

import catboost
import pandas as pd
import numpy as np
import json
from catboost import CatBoostRegressor, Pool
import pm4py
import utils.log_parsing as log_parsing

n_examples = 'max'
remove_outliers = True

df_train = pd.read_csv('experiments/hospital/train_hospital_processed.csv')#.iloc[:n_examples]
df_test = pd.read_csv('experiments/hospital/test_hospital_processed.csv').iloc[:60]

if remove_outliers:
    df_train = log_parsing.remove_outliers_iqr(df_train, 'lead_time')


def fit_model(train_df, y, test_df, test_y):

    categorical_features = train_df.select_dtypes(exclude=np.number).columns
    train_df[categorical_features] = train_df[categorical_features].astype(str)
    column_types = train_df.dtypes.astype(str).to_dict()
    
    params = {
        'depth': 10,
        'learning_rate': 0.2,
        'iterations': 700,
        'early_stopping_rounds': 80,
        'thread_count': 4,
        'logging_level': 'Verbose',
        'task_type': "CPU"  # "GPU" if int(os.environ["USE_GPU"]) else "CPU"
    }

    print('Starting training...')
    params["loss_function"] = "MAE"
    train_data = Pool(train_df, y, cat_features=categorical_features.values)
    test_data = Pool(test_df, test_y, cat_features=categorical_features.values)
    model = CatBoostRegressor(**params)
    model.fit(train_data, verbose=True, plot=False, eval_set=(test_data))
    return model


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
from sklearn.metrics import median_absolute_error
mse = mean_absolute_error(y_test, y_pred)
print('The MAE for mean is ', mse) 

#Same with median 
mse = median_absolute_error(y_test, y_pred)
print('The MAE fom median is ', mse)

import pickle as pkl
pkl.dump(model, open('model.pkl', 'wb'))

# Pkl dump the y_pred
pkl.dump(y_pred, open(f'y_pred_{n_examples}.pkl', 'wb'))

# Print df_train lenght
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
plt.legend()
plt.title(f'True vs Predicted for {n_examples} examples {"" if remove_outliers else "without"} outliers')

# %%
