# %%

def encode_trace_log(input_log: pd.DataFrame, trace_attributes: list):
    """
    Encodes an event log into a format where each trace is represented by a single row.
    
    Each row includes:
    - Trace attributes specified by the user.
    - A list of activities with their respective time_from_start values.

    Parameters:
        input_log (pd.DataFrame): The event log, containing 'case:concept:name', 'concept:name', and 'time_from_start'.
        trace_attributes (list): A list of column names to include as trace-level attributes.
    
    Returns:
        pd.DataFrame: A DataFrame where each row represents a single trace.
    """
    # Group events by trace
    grouped = input_log.groupby('case:concept:name')
    
    trace_dict = {}
    
    for trace_id, group in grouped:
        # Sort events in the trace by 'time_from_start'
        group = group.sort_values(by='time_from_start')
        
        # Create the list of activity-time tuples
        activity_time_seq = [
            (row['concept:name'], row['time_from_start']) 
            for _, row in group.iterrows()
        ]
        
        # Create a dictionary for the trace with its attributes and sequence
        trace_info = {attr: group.iloc[0][attr] for attr in trace_attributes}
        trace_info['ActTimeSeq'] = activity_time_seq
        
        # Add to the main dictionary with the trace ID as the key
        trace_dict[trace_id] = trace_info
    
    return trace_dict

# %% 
import pandas as pd
import random

n_samples = 1
train = pd.read_csv('/home/padela/Scaricati/train.csv')
train = train.sample(frac=1, random_state=1618).reset_index(drop=True).iloc[:n_samples]

test = pd.read_csv('/home/padela/Scaricati/test.csv')
test = test.sample(frac=1, random_state=1618).reset_index(drop=True).iloc[:20]

import numpy as np

data = np.array([
    [9939.7, 346.34, 0.51],
    [10307.48, 352.24, 0.53],
    [16597.58, 586.63, 0.85],
    [9499.52, 357.2, 0.49],
    [9476.67, 288.03, 0.49],
    [8815.13, 192.22, 0.45],
    [11419.47, 94.57, 0.59],
    [10307.48, 352.24, 0.53],
    [16597.58, 586.63, 0.85]
])

column_means = np.mean(data, axis=0)
print(f"{column_means[0]:.2f} - {column_means[1]:.2f} - {column_means[2]:.2f}")

