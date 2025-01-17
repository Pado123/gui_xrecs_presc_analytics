# # %% Train Perpetual
# from perpetual import PerpetualBooster
# import pandas as pd
# import catboost

# df_train = pd.read_csv('logs/BPI2017_before_dfTrain.csv')
# df_test = pd.read_csv('logs/BPI2017_before_dfTest.csv')

# # Delete a list of columns
# # df_train = df_train.drop(['exp_vec_len', 't0', 't1', 't2', 't3', 'time_after_1', 'time_after_5', 'time_after_20'], axis=1)
# # df_test = df_test.drop(['exp_vec_len', 't0', 't1', 't2', 't3', 'time_after_1', 'time_after_5', 'time_after_20'], axis=1)

# # Find the categorical columns
# categorical_columns = df_train.select_dtypes(include=['object']).columns

# # Convert the categorical columns to pandas categorical
# df_train[categorical_columns] = df_train[categorical_columns].astype('category')
# df_test[categorical_columns] = df_test[categorical_columns].astype('category')

# #Set y as "lead_time"
# y_train = df_train['lead_time']
# y_test = df_test['lead_time']

# # #Remove from train 
# X_train = df_train.drop(['lead_time', 'case:concept:name'], axis=1)
# X_test = df_test.drop(['lead_time','case:concept:name'], axis=1)

# print('Train Perpetual')
# model = PerpetualBooster(objective="SquaredLoss", num_threads=8)
# for budget in range(1, 20):
#     model.fit(X_train, y_train, budget=0.1*budget)


#     # Predict the value of y
#     print('Predict Perpetual')
#     y_pred = model.predict(X_test)

#     #Evaluate MSE using sklearn
#     from sklearn.metrics import mean_absolute_error
#     mse = mean_absolute_error(y_test, y_pred)
#     print('The MAE from perpetual is ', mse, 'with budget', budget)


# The mse is 743902.451 for budget 1.0
# The mse is 737207 for budget 2.1
# The mse is 3283986 for budget 4.1
# The mse is 739799 for budget 1.5
# budget 0 sballa
# Best mae is 517

# The best mse for catboost is 3469715
# The best mae for catboost is 763


# %% Train catboost
import catboost
import pandas as pd
import numpy as np
import json
from catboost import CatBoostRegressor, Pool
import pm4py

# n_examples = 777

df_train = pd.read_csv('/home/padela/Scaricati/consulta_1.0_0.007_train.csv')#.iloc[:n_examples]
df_test = pd.read_csv('/home/padela/Scaricati/consulta_1.0_0.007_test.csv')
# print(df_train.columns)


# # Delete a list of columns
# df_train = df_train.drop(['exp_vec_len', 't0', 't1', 't2', 't3', 'time_after_1', 'time_after_5', 'time_after_20'], axis=1)
# df_test = df_test.drop(['exp_vec_len', 't0', 't1', 't2', 't3', 'time_after_1', 'time_after_5', 'time_after_20'], axis=1)

def fit_model(train_df, y, test_df, test_y):

    categorical_features = train_df.select_dtypes(exclude=np.number).columns
    train_df[categorical_features] = train_df[categorical_features].astype(str)
    column_types = train_df.dtypes.astype(str).to_dict()
    
    params = {
        'depth': 10,
        'learning_rate': 0.2,
        'iterations': 600,
        'early_stopping_rounds': 80,
        'thread_count': 4,
        'logging_level': 'Verbose',
        'task_type': "GPU"  # "GPU" if int(os.environ["USE_GPU"]) else "CPU"
    }

    print('Starting training...')
    params["loss_function"] = "MAE"
    train_data = Pool(train_df, y, cat_features=categorical_features.values)
    test_data = Pool(test_df, test_y, cat_features=categorical_features.values)
    model = CatBoostRegressor(**params)
    model.fit(train_data, verbose=True, plot=False, eval_set=(test_data))
    return model


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
# %%
