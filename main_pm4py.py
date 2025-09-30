# %% Import libraries
import pandas as pd
import json
import os
import random

# Resolve repository base directory from this file location to avoid chdir side effects
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the exp_name and KPI
random.seed(1618)  # Set a random seed for reproducibility
exp_name = 'bpi12'
kpi = 'lead_time' #Can be either 'lead_time' or 'outcome_pred'
print(f"Experiment name set to: {exp_name}, KPI is set to: {kpi}")

print(f"Loading hyperparameters from 'hparams/{exp_name}.json'")
with open(os.path.join(BASE_DIR, 'hparams', f'{exp_name}.json')) as f:
    hparams = json.load(f)
print("Hyperparameters loaded successfully.")

from utils import add_activity_outcome
from utils import preprocessing_acts as pr_act
from utils import preprocessing_times as pr_time
from utils import log_parsing
from utils import IO

# From the hparams file, derive variables
log_path = hparams['log_path']
date_format = hparams['date_format']
cf_preprocessing = hparams['cf_preprocessing']
hashed = hparams['hashed']
case_id_name, activity_column_name, trace_attr = log_parsing.parse_cf_caseid_traceatt(hparams)

#The first date is the start date of the log, the second date is the end date of the log
start_date = hparams['start_date']
end_date = hparams['end_date']
parse_dates = [start_date, end_date]
print(f"Log path: {log_path}, Date format: {date_format}, Parse dates: {parse_dates}")

log, attr_trace_dict = pr_act.encode_log(log_path, case_id_name=case_id_name, parse_dates=parse_dates,
                        activity_column_name=activity_column_name, 
                        encoding=cf_preprocessing, last_act_num=3, trace_attr=trace_attr)

# print(f" Save the hasing_dict to 'hparams/{exp_name}_hasing_dict.json'")
print("Activity history features added.")

if not hparams['rem_time']:
    log = pr_time.add_time_features(log)
    print("Time-based features added.")

else:
    log = pr_time.add_remaining_time_features(log)
    # Rename the column "remaining_time" to "lead_time"
    log = log.rename(columns={'remaining_time': 'lead_time'})
    print("Remaining time features added.")
    print(log['lead_time'].mean(), 'AAAAAAAAAAAAAAAAAAAA')

log = pr_time.add_daily_features(log)
print("Daily features added.")

log = log_parsing.reorder_cols(log)
print("Columns reordered.")

# Create the experiment folder if it doesn't exist
experiment_folder = os.path.join(BASE_DIR, 'experiments', exp_name)
if not os.path.exists(experiment_folder):
    print(f"Creating experiment folder at '{experiment_folder}'")
    os.makedirs(experiment_folder)
    print("Experiment folder created.")

log = log_parsing.drop_0s(log) 
log = log_parsing.add_attr(log, attr_trace_dict, cf_preprocessing)


if kpi == 'outcome_pred':
    log = add_activity_outcome.add_activity_outcome(log, act_to_encode=hparams["acts_not_freq"][0], 
                                                    occurs_in_remaining=False)
    log.pop('lead_time', None)
    print("Activity outcome added.")
    
if hashed:
    print("Hashing the log...")
    log, hashing_dict_act, hashing_dict_attr, hash_trace_attr_names = pr_act.hash_log(log, activity_column_name=activity_column_name, 
                          kpi=kpi, trace_attr=trace_attr)
    print("Log hashed.")

# Split the log into train and test
train, test = pr_time.train_test_split(log, test_size=0.2, 
                                       random_state=1618, temporal=True, 
                                       encoding=cf_preprocessing,
                                       trace_attr=trace_attr,
                                       attr_trace_dict=attr_trace_dict,
                                       kpi=kpi, hash_trace_attr_names=hash_trace_attr_names if hashed else None)

IO.save_log(experiment_folder=experiment_folder, log=train,
            encoding_cf=cf_preprocessing, type='train', kpi=kpi, hashed=hashed)
IO.save_log(experiment_folder=experiment_folder, log=test,
            encoding_cf=cf_preprocessing, type='test', kpi=kpi, hashed=hashed)
print("Train and test logs saved.")

if hashed:
    with open(os.path.join(experiment_folder, 'hashing_dict.json'), 'w') as f:
        json.dump(hashing_dict_act, f)
    with open(os.path.join(experiment_folder, 'hash_trace_attr_names.json'), 'w') as f:
        json.dump(hash_trace_attr_names, f)

    print("Hashing dictionary saved successfully.")
print("Preprocessing Procedure completed.")


# %%
