import pandas as pd
import random

def encode_trace_log(input_log: pd.DataFrame, trace_attributes: 
                     list, attr_trace_dict: dict, type: str, kpi:str) -> pd.DataFrame:
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
    
    attr_trace_dict = {}
    
    for trace_id, group in grouped:
        # Sort events in the trace by 'time_from_start'
        group = group.sort_values(by='time_from_start')
        
        # Create the list of activity-time tuples
        activity_time_seq = [
            (row['concept:name'], row['time_from_start']) 
            for _, row in group.iterrows()
        ]
        
        # Create a dictionary for the trace with its attributes and sequence
        trace_info = {}

        # if Trace_attributes is not none or an empty list, add the attributes to the trace_info dictionary
        if trace_attributes:
            for attr in trace_attributes:
                trace_info[attr] = group[attr].iloc[0]

        if type == 'test':
            if type == 'lead_time':
                activity_time_seq.append(['Running'])
                trace_info['ActTimeSeq'] = activity_time_seq
                attr_trace_dict[trace_id] = trace_info
            elif kpi == 'outcome_pred':
                attr_trace_dict[trace_id] = trace_info
                trace_info['ActTimeSeq'] = activity_time_seq
                if group[str(list(group.columns)[-1])].iloc[-1] == 0:
                    out = 0
                else:
                    out = 1
                trace_info[str(list(group.columns)[-1])[4:]] = out
        
        elif type == 'train':
            if kpi == 'outcome_pred':
                attr_trace_dict[trace_id] = trace_info
                trace_info['ActTimeSeq'] = activity_time_seq
                trace_info[str(list(group.columns)[-1])[4:]] = int(group[str(list(group.columns)[-1])].mean()>0)
                

            elif kpi == 'lead_time':
                attr_trace_dict[trace_id] = trace_info
                trace_info['lead_time'] = group['lead_time'].iloc[0]               
                trace_info['ActTimeSeq'] = activity_time_seq
    
    return attr_trace_dict

def add_time_features(log, start_col='start:timestamp', end_col='time:timestamp', date_format='%Y-%m-%d %H:%M:%S%z'):
    
    # If the start and end columns are not in datetime format, convert them
    if not pd.api.types.is_datetime64_any_dtype(log[start_col]):
        log[start_col] = pd.to_datetime(log[start_col], format='mixed', utc=True)
    if not pd.api.types.is_datetime64_any_dtype(log[end_col]):
        log[end_col] = pd.to_datetime(log[end_col], format='mixed', utc=True)

    # Cast the time columns to unix 
    log['activity_duration'] = (log[end_col] - log[start_col])
    log['activity_duration'] = (log['activity_duration'].dt.total_seconds() / 60).round(0).astype(int)

    # For each activity in each trace, evaluate the time from the start of the trace
    log['time_from_start'] = ((log[end_col] - log.groupby('case:concept:name')[start_col].transform('first')).dt.total_seconds() / 60).round(0).astype(int)
    
    #Group by trace and calculate the duration of the trace, put it in a column called "lead_time" that is the difference between the last time:timestamp and the first start:timestamp
    log['lead_time'] = log.groupby('case:concept:name')[end_col].transform('last') - log.groupby('case:concept:name')[start_col].transform('first') 
    log['lead_time'] = (log['lead_time'].dt.total_seconds() / 60).round(0).astype(int)

    return log

def add_daily_features(log, start_col='start:timestamp', end_col='time:timestamp'):

    # Extract the day of the week and the hour of the day from the start:timestamp
    log['day_of_week'] = log[end_col].dt.dayofweek
    log['hour_of_day'] = log[end_col].dt.hour

    return log

def get_running(df, case_col="case:concept:name"):
    """
    Trunca ogni traccia nel DataFrame sostituendola con un suo prefisso di lunghezza casuale tra 2 e len(trace)-1.

    :param df: DataFrame contenente le tracce.
    :param case_col: Nome della colonna che identifica i casi (default: "case_id").
    :return: DataFrame con tracce troncate.
    """
    truncated_df_list = []

    # Raggruppa per case_id e applica la troncatura
    for case_id, group in df.groupby(case_col):
        if len(group) > 2:
            new_length = random.randint(2, len(group))
            truncated_group = group.iloc[:new_length]  # Prende solo il prefisso
        else:
            truncated_group = group  # Se ha solo 1-2 eventi, la lasciamo invariata

        truncated_df_list.append(truncated_group)

    # Ricostruzione del DataFrame finale
    truncated_df = pd.concat(truncated_df_list).reset_index(drop=True)
    return truncated_df


def train_test_split(log, test_size=0.2, random_state=1618, temporal=True,
                     encoding=None, trace_attr=None, attr_trace_dict=None, kpi=None):

    case_ids = log['case:concept:name'].unique()

    if temporal:
        log = log.sort_values(['case:concept:name','time:timestamp'], ascending=True)
    else:
        log = log.sample(frac=1, random_state=random_state)
    
    n_test = int(len(case_ids) * test_size)
    train_ids = case_ids[:-n_test]
    test_ids = case_ids[-n_test:]

    train = log[log['case:concept:name'].isin(train_ids)]
    test = log[log['case:concept:name'].isin(test_ids)]

    if encoding == 'sequential':
        train = train.sort_values(by=['case:concept:name', 'time:timestamp'])
        test = test.sort_values(by=['case:concept:name', 'time:timestamp'])
        test = get_running(test)
        train = encode_trace_log(train, trace_attr, attr_trace_dict, type='train', kpi=kpi)
        test = encode_trace_log(test, trace_attr, attr_trace_dict, type='test', kpi=kpi)
        
    return train, test