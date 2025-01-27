# %% Train Perpetual
from perpetual import PerpetualBooster
import pandas as pd
import catboost

df_train = pd.read_csv('logs/BPI2017_before_dfTrain.csv')
df_test = pd.read_csv('logs/BPI2017_before_dfTest.csv')

# Delete a list of columns
# df_train = df_train.drop(['exp_vec_len', 't0', 't1', 't2', 't3', 'time_after_1', 'time_after_5', 'time_after_20'], axis=1)
# df_test = df_test.drop(['exp_vec_len', 't0', 't1', 't2', 't3', 'time_after_1', 'time_after_5', 'time_after_20'], axis=1)

# Find the categorical columns
categorical_columns = df_train.select_dtypes(include=['object']).columns

# # Convert the categorical columns to pandas categorical
df_train[categorical_columns] = df_train[categorical_columns].astype('category')
df_test[categorical_columns] = df_test[categorical_columns].astype('category')

#Set y as "lead_time"
y_train = df_train['lead_time']
y_test = df_test['lead_time']

# #Remove from train 
X_train = df_train.drop(['lead_time', 'case:concept:name'], axis=1)
X_test = df_test.drop(['lead_time','case:concept:name'], axis=1)

print('Train Perpetual')
model = PerpetualBooster(objective="SquaredLoss", num_threads=8)
for budget in range(1, 20):
    model.fit(X_train, y_train, budget=0.1*budget)

    # Predict the value of y
    print('Predict Perpetual')
    y_pred = model.predict(X_test)

    #Evaluate MSE using sklearn
    from sklearn.metrics import mean_absolute_error
    mse = mean_absolute_error(y_test, y_pred)
    print('The MAE from perpetual is ', mse, 'with budget', budget)


