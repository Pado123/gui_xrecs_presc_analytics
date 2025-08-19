# %%
import os 
#Go to the main dir
# working_directory = '/home/padela/Scrivania/LLMs/gui_xrecs_presc_analytics'
# os.chdir(working_directory)

import pandas as pd
import json
from utils.log_parsing import read_data

def activity_frequency_in_traces(event_log):
    """
    Given an event log in .csv format, returns a dictionary where keys are activities
    and values are the number of traces in which the activity occurs.

    Args:
        event_log_path (str): Path to the event log file in .csv format.

    Returns:
        dict: A dictionary with activities as keys and their frequency in traces as values.
    """

    # Ensure the required columns exist
    if 'case:concept:name' not in event_log.columns or 'concept:name' not in event_log.columns:
        raise ValueError("The event log must contain 'trace_id' and 'activity' columns.")

    # Group by activity and count unique trace IDs
    activity_frequency = event_log.groupby('concept:name')['case:concept:name'].nunique().to_dict()

    return activity_frequency

# Set the exp_name 
def apply_activity_frequency_in_traces():
    for exp_name in ['bac', 'bpi12', 'hospital']:
        print(f"Log is: {exp_name}")

        # print(f"Experiment name set to: {exp_name}")

        # print(f"Loading hyperparameters from 'hparams/{exp_name}.json'")
        with open(f'hparams/{exp_name}.json') as f:
            hparams = json.load(f)
        # print("Hyperparameters loaded successfully.")

        # Read the log
        log_path = hparams['log_path']
        date_format = hparams['date_format']
        start_date = hparams['start_date']
        end_date = hparams['end_date']
        parse_dates = [start_date, end_date]

        event_log = read_data(log_path, start_col=parse_dates[0], date_format="%Y-%m-%d %H:%M:%S")
        # print("Log loaded successfully.")

        # Use the function defined above    
        traces_freq = activity_frequency_in_traces(event_log)

        acts_not_freq = []

        # For each key in the dictionary, print it with the value
        for key, value in traces_freq.items():
            freq = int(100*value/len(event_log['case:concept:name'].unique()))

            # If the frequence is between 5 and 20, print it
            if freq >= 5 and freq <= 70:
                acts_not_freq.append(key)
                print(f"Activity: {key}, Frequency: {freq} %")
        
        # Add act_not_freq into the hparams dict
        hparams['acts_not_freq'] = acts_not_freq

        # Save the hparams dict into a json file
        with open(f'hparams/{exp_name}.json', 'w') as f:
            json.dump(hparams, f, indent=4)
# %%
apply_activity_frequency_in_traces()
# %%
