import pandas as pd
import random
random.seed(1618)  # Set a random seed for reproducibility

def encode_trace_log(input_log: pd.DataFrame, trace_attributes: 
                     list, attr_trace_dict: dict, type: str, kpi:str, encoding:str) -> pd.DataFrame:
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
    if encoding == 'sequential':
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
                if kpi == 'lead_time':
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
                    # This means that if there is at least one activity in the trace that has a value of 1, then the outcome is 1
                    # trace_info[str(list(group.columns)[-1])[4:]] = int(group[str(list(group.columns)[-1])].mean()>0)
                    

                elif kpi == 'lead_time':
                    attr_trace_dict[trace_id] = trace_info
                    trace_info['lead_time'] = group['lead_time'].iloc[0]               
                    trace_info['ActTimeSeq'] = activity_time_seq
        
        return attr_trace_dict

    elif encoding == 'aggr_hist':
        # Remove everything but: trace attributes, case_id, activity, time_from_start and every column starting with '#' and last column
        log_cols = ['case:concept:name', 'concept:name', 'time:timestamp', 'time_from_start'] + [col for col in input_log.columns if col.startswith('#')] + trace_attributes + [str(list(input_log.columns)[-1])]
        # log_cols = ['case:concept:name', 'concept:name'] + [str(list(input_log.columns)[-1])]
        input_log = input_log[log_cols]
        return input_log 


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
"""
def get_running(df, case_col="case:concept:name"):
    \"""
    Trunca ogni traccia nel DataFrame sostituendola con un suo prefisso di lunghezza casuale tra 2 e len(trace)-1.

    :param df: DataFrame contenente le tracce.
    :param case_col: Nome della colonna che identifica i casi (default: "case_id").
    :return: DataFrame con tracce troncate.
    \"""
    truncated_df_list = []
    df[case_col] = df[case_col].astype(str)  # Assicurati che case_col sia di tipo stringa

    # Raggruppa per case_id e applica la troncatura
    for case_id, group in df.groupby(case_col):
        if len(group) > 3:
            new_length = random.randint(3, len(group)-1)
            truncated_group = group.iloc[:new_length]  # Prende solo il prefisso
        else:
            continue  # Se ha solo 1-2 eventi, skip it
        truncated_df_list.append(truncated_group)

    # Ricostruzione del DataFrame finale
    truncated_df = pd.concat(truncated_df_list).reset_index(drop=True)
    return truncated_df
"""

def get_running(df, case_col="case:concept:name", encoding=None):
    """
    Truncate each trace to a random prefix of length between 3 and len(trace)-1.
    Traces of length <= 3 are discarded.
    """
    df = df.copy()
    df[case_col] = df[case_col].astype(str)
    random.seed(1618) 

    # Evaluate the median length of the traces
    median_length = df.groupby(case_col).size().median()
    print(f"Median length of traces: {median_length}")
    left_cutoff_length = max(2, int(median_length)*0.1)  # Ensure at least 3 events

    def truncate(group):
        if len(group) > 2*left_cutoff_length :
            new_length = random.randint(left_cutoff_length, len(group) - left_cutoff_length)
            if encoding == 'sequential':
                return group.iloc[:new_length]
            else:
                return group.iloc[new_length]
        else:
            return None  # drop short traces

    truncated_df = (
        df.groupby(case_col, sort=False, group_keys=False)
          .apply(truncate)
          .dropna(how="all")
          .reset_index(drop=True)
    )

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


    train = train.sort_values(by=['case:concept:name', 'time:timestamp'])
    test = test.sort_values(by=['case:concept:name', 'time:timestamp'])
    test = get_running(test, encoding=encoding, case_col='case:concept:name') 
    print(f'last column mean of test is {test[str(list(test.columns)[-1])].mean()}')
    print(f'last column mean of train is {train[str(list(train.columns)[-1])].mean()}')
    print(f'last column names is {str(list(train.columns)[-1])}')
    train = encode_trace_log(train, trace_attr, attr_trace_dict, type='train', kpi=kpi, encoding=encoding)
    test = encode_trace_log(test, trace_attr, attr_trace_dict, type='test', kpi=kpi, encoding=encoding)
    if encoding == 'aggr_hist':
        # drop time:timestamp column
        train = train.drop(columns=['time:timestamp'])
        test = test.drop(columns=['time:timestamp'])

    return train, test