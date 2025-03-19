# %% Import libraries
import pandas as pd
import pm4py 
import json
import os

# Set the exp_name 
exp_name = 'hospital'
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

log, attr_trace_dict = pr_act.encode_log(log_path, case_id_name=case_id_name, parse_dates=parse_dates,
                         activity_column_name=activity_column_name, 
                         encoding=cf_preprocessing, last_act_num=3, trace_attr=trace_attr)

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

log = log_parsing.drop_0s(log) 
log = log_parsing.add_attr(log, attr_trace_dict, cf_preprocessing)

# IO.save_log(experiment_folder, log=log,
#             encoding_cf=cf_preprocessing, type=None)
print("Log saved.")

# Split the log into train and test
train, test = pr_time.train_test_split(log, test_size=0.2, 
                                       random_state=1618, temporal=True, 
                                       encoding=cf_preprocessing,
                                       trace_attr=trace_attr,
                                       attr_trace_dict=attr_trace_dict)

IO.save_log(experiment_folder=experiment_folder, log=train,
            encoding_cf=cf_preprocessing, type='train')
IO.save_log(experiment_folder=experiment_folder, log=test,
            encoding_cf=cf_preprocessing, type='test')
print("Preprocessing Procedure completed.")

