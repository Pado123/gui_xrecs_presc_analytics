# %% Import libraries
import pandas as pd
import pm4py 
import json
import os

# Set the exp_name 
exp_name = 'bac'
print(f"Experiment name set to: {exp_name}")

print(f"Loading hyperparameters from 'hparams/{exp_name}.json'")
with open(f'hparams/{exp_name}.json') as f:
    hparams = json.load(f)
print("Hyperparameters loaded successfully.")

from utils import preprocessing_acts as pr_act
from utils import preprocessing_times as pr_time
from utils import log_parsing
from utils import IO

# From the hparams file, derive variables
log_path = hparams['log_path']
date_format = hparams['date_format']
cf_preprocessing = hparams['cf_preprocessing']
case_id_name, activity_column_name, trace_attr = log_parsing.parse_cf_caseid_traceatt(hparams)

#The first date is the start date of the log, the second date is the end date of the log
start_date = hparams['start_date']
end_date = hparams['end_date']
parse_dates = [start_date, end_date]
print(f"Log path: {log_path}, Date format: {date_format}, Parse dates: {parse_dates}")

log = pr_act.encode_log(log_path, case_id_name=case_id_name, parse_dates=parse_dates,
                         activity_column_name=activity_column_name, 
                         encoding=cf_preprocessing, last_act_num=3)
print("Activity history features added.")

log = pr_time.add_time_features(log)
print("Time-based features added.")

log = pr_time.add_daily_features(log)
print("Daily features added.")

log = log_parsing.reorder_cols(log)
print("Columns reordered.")

# Create the experiment folder if it doesn't exist
experiment_folder = f'experiments/{exp_name}'
if not os.path.exists(experiment_folder):
    print(f"Creating experiment folder at '{experiment_folder}'")
    os.makedirs(experiment_folder)
    print("Experiment folder created.")

IO.save_log(experiment_folder, log)
print("Log saved.")

# Split the log into train and test
train, test = pr_time.train_test_split(log, test_size=0.2, 
                                       random_state=1618, temporal=True, 
                                       encoding=cf_preprocessing,
                                       trace_attr=trace_attr)

IO.save_log(experiment_folder, train, type='train')
IO.save_log(experiment_folder, test, type='test')
print("Preprocessing Procedure completed.")

# %%
# Check phase
train = pd.read_csv(f'{experiment_folder}/train.csv')
test = pd.read_csv(f'{experiment_folder}/test.csv')

# Make a vector with the maximum timestamp of each trace
max_timestamps = train.groupby('case:concept:name')['time:timestamp'].max()
max_timestamps = max_timestamps.reset_index()
max_timestamps.columns = ['case:concept:name', 'max_timestamp']

# same for the test
max_timestamps_test = test.groupby('case:concept:name')['time:timestamp'].max()
max_timestamps_test = max_timestamps_test.reset_index()
max_timestamps_test.columns = ['case:concept:name', 'max_timestamp']

# Check if the maximum of the train is less than the minimum of the test
if max(max_timestamps['max_timestamp']) < min(max_timestamps_test['max_timestamp']):
    print("The train and test sets are correctly split.")

# If not, print the percentage of wrong values
else:
    wrong_values = sum(max_timestamps['max_timestamp'] > min(max_timestamps_test['max_timestamp'])) / len(max_timestamps)
    print(f"Percentage of overlapping events: {round(wrong_values*100, 2)}%")

# Check if they share some cases
shared_cases = set(max_timestamps['case:concept:name']).intersection(set(max_timestamps_test['case:concept:name']))
print(f"Number of shared cases between test and train: {len(shared_cases)}")

# Controlla se la durata media e mediana della lunghezza di ogni traccia di train e test è simile
train['lead_time'].describe()
test['lead_time'].describe()


# %%
