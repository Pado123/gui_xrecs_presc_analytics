import pandas as pd
import pm4py
import json
import tqdm

def from_lifecycles_to_start_end(event_log):
    """
    Transforms the event log so that each activity in a trace is represented by a single row
    with "start_date" and "end_date" instead of separate rows for 'start' and 'complete' transitions.

    :param event_log: pandas DataFrame containing the event log
    :return: pandas DataFrame with combined transitions
    """
    # Filter start and complete events
    start_events = event_log[event_log['lifecycle:transition'] == 'start'].copy()
    complete_events = event_log[event_log['lifecycle:transition'] == 'complete'].copy()

    # Assign unique identifiers to match start and complete events correctly within each trace
    start_events['activity_instance'] = start_events.groupby(
        ['case:concept:name', 'concept:name']
    ).cumcount()

    complete_events['activity_instance'] = complete_events.groupby(
        ['case:concept:name', 'concept:name']
    ).cumcount()

    # Merge start and complete events on case ID, activity name, and activity instance
    combined_events = pd.merge(
        start_events, 
        complete_events, 
        on=['case:concept:name', 'concept:name', 'activity_instance'],
        suffixes=('_start', '_complete')
    )

    # Create the result with 'start_date' and 'end_date'
    combined_events = combined_events[[
        'case:concept:name', 'concept:name', 'time:timestamp_start', 'time:timestamp_complete'
    ]].rename(columns={
        'time:timestamp_start': 'start:timestamp',
        'time:timestamp_complete': 'time:timestamp'
    })

    return combined_events

def dump_hashing_act(log, activity_column_name='concept:name'):

    # To each different activity, assign a progressive letter, then if they are more than 26, use two letters
    unique_activities = log[activity_column_name].unique()
    activity_hash = {}
    for i, activity in enumerate(unique_activities):
        if i < 26:
            activity_hash[activity] = chr(65 + i)  # A-Z
        else:
            first_letter = chr(65 + (i // 26) - 1)
            second_letter = chr(65 + (i % 26))
            activity_hash[activity] = first_letter + second_letter  # AA, AB, ..., AZ, BA, BB, ...

    # Return a dictionary mapping activities to their corresponding letters
    return activity_hash

def hash_log(log, activity_column_name='concept:name'):

    # Generate the hashing dictionary
    hasing_dict = dump_hashing_act(log, activity_column_name=activity_column_name)

    # Replace every activity in the log with its corresponding letter
    log[activity_column_name] = log[activity_column_name].map(hasing_dict)

    #return the log with hashed activities
    return log


def gen_attr_dict(df, trace_attr):

    """
    Generates a dictionary where each key is an attribute from trace_attr,
    and each value is a dictionary mapping 'case:concept:name' to the attribute value.
    
    :param df: Pandas DataFrame containing the event log.
    :param trace_attr: List of trace attributes to extract.
    :return: Nested dictionary {trace_attr: {case:concept:name: value}}
    """
    attr_trace_dict = {}
    
    for attr in trace_attr:
        attr_trace_dict[attr] = df.set_index('case:concept:name')[attr].to_dict()
    
    return attr_trace_dict


def encode_log(df, case_id_name='case:concept:name', parse_dates=['start:timestamp','time:timestamp'], 
               activity_column_name='concept:name', encoding='aggr_hist', last_act_num=3, trace_attr=None):
    """
    Adds historical information to a dataframe based on the specified encoding.

    Parameters:
    - df (pd.DataFrame or str): The input dataframe or path to the CSV/XES file.
    - case_id_name (str): The column representing the case identifier.
    - activity_column_name (str): The column representing activity names.
    - encoding (str): Encoding type ('aggr_hist', 'last_k', 'no_hist').
    - last_act_num (int): Number of last activities to consider for 'last_k' encoding.

    Returns:
    - pd.DataFrame: The transformed dataframe.
    """
    encoding_list = ['aggr_hist', 'last_k', 'no_hist', 'sequential']

    if isinstance(df, str):
        try:
            if df.endswith('.xes'):
                log = pm4py.read_xes(df)
                df = pm4py.convert_to_dataframe(log, parse_dates=parse_dates)
            else:
                df = pd.read_csv(df, parse_dates=parse_dates)
        except Exception as e:
            raise ValueError(f"Error loading dataframe from path: {e}")

    hasing_dict = dump_hashing_act(df, activity_column_name=activity_column_name)

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame or a string path to a CSV/XES file.")

    if encoding not in encoding_list:
        raise ValueError(f"Unknown encoding: {encoding}, possible values are: {encoding_list}")

    if encoding == 'last_k' and not isinstance(last_act_num, int):
        raise ValueError(f"last_act_num must be an integer, got {type(last_act_num).__name__}")

        # If lifecycles are present, convert them to start and end events

    attr_trace_dict = gen_attr_dict(df, trace_attr)
    if 'lifecycle:transition' in df.columns:

        # Remove the transitions that are not start or end, then print the number of rows removed
        print(f"Removing {len(df[~df['lifecycle:transition'].isin(['start', 'complete'])])} rows with transitions different from 'start' or 'complete'")
        df = df[df['lifecycle:transition'].isin(['start', 'complete'])]

        print("Lifecycles detected, converting")
        df = from_lifecycles_to_start_end(df)

    if encoding == 'aggr_hist':
        for activity in df[activity_column_name].unique():
            df[f"# {activity_column_name}={activity}"] = 0
            # First put 1 in correspondence to each activity
            df.loc[df[activity_column_name] == activity, f"# {activity_column_name}={activity}"] = 1
            # Sum the count from the previous events
            df[f"# {activity_column_name}={activity}"] = \
                df.groupby(case_id_name)[f"# {activity_column_name}={activity}"].cumsum()
        return df, attr_trace_dict, hasing_dict

    elif encoding == 'last_k':
        # Add columns for the last `last_act_num` activities
        for i in range(1, last_act_num + 1):
            df[f'last_{i}_activity'] = None

        # Iterate over each case to fill in the last `k` activities
        for case_id, group in df.groupby(case_id_name):
            history = []
            for idx, row in group.iterrows():
                # Fill the last `k` activities for the current row
                for i in range(1, last_act_num + 1):
                    if len(history) >= i:
                        df.loc[idx, f'last_{i}_activity'] = history[-i]
                    else:
                        df.loc[idx, f'last_{i}_activity'] = None
                # Update the history with the current activity
                history.append(row[activity_column_name])
        return df, attr_trace_dict, hasing_dict

    elif encoding == 'no_hist':  
        return df, None, hasing_dict

    elif encoding == 'sequential':
        return df, attr_trace_dict, hasing_dict
        