
import pandas as pd
def add_time_features(log, start_col='start:timestamp', end_col='time:timestamp'):
    
    # Cast the time columns to unix 
    log['activity_duration'] = (log[end_col] - log[start_col])
    log['activity_duration'] = (log['activity_duration'].dt.total_seconds() / 60).round(0).astype(int)

    #Group by trace and calculate the duration of the trace, put it in a column called "trace_duration" that is the difference between the last time:timestamp and the first start:timestamp
    log['trace_duration'] = log.groupby('case:concept:name')[end_col].transform('last') - log.groupby('case:concept:name')[start_col].transform('first') 
    log['trace_duration'] = log['trace_duration'] = (log['trace_duration'].dt.total_seconds() / 60).round(0).astype(int)

    return log

def add_daily_features(log, start_col='start:timestamp', end_col='time:timestamp'):

    # Extract the day of the week and the hour of the day from the start:timestamp
    log['day_of_week'] = log[end_col].dt.dayofweek
    log['hour_of_day'] = log[end_col].dt.hour

    return log