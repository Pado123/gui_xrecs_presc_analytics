import pandas as pd
import numpy as np
import pm4py
import tqdm
from pm4py.objects.log.importer.xes import importer as xes_importer

def drop_0s(log):

    #Cut traces of lenght 2
    # log = log.groupby('case:concept:name').filter(lambda x: len(x) > 2)
    return log[log['lead_time'] > 0]

def parse_cf_caseid_traceatt(hparams):
    try:
        activity_column_name = hparams['activity_column_name']
    except:
        activity_column_name = 'concept:name'

    try:
        case_id_name = hparams['case_id_name']
    except:
        case_id_name = 'case:concept:name'

    try:
        trace_attr = hparams['trace_attr']
    except:
        trace_attr = []

    return case_id_name, activity_column_name, trace_attr

def read_data(filename, start_col='start:timestamp', date_format="%Y-%m-%d %H:%M:%S"):

    if '.csv' in filename:
        try:
            df = pd.read_csv(filename, header=0, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(filename, header=0, encoding="cp1252", low_memory=False)
    
    elif '.xes' in filename:
        log = xes_importer.apply(filename)
        df = pm4py.convert_to_dataframe(log)
    
    elif '.parquet' in filename:
        df = pd.read_parquet(filename, engine='pyarrow')
    

    return df

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def reorder_cols(df, encoding=None):

    if encoding == 'sequential':
        return df

    # Check if 'case:concept:name' exists and should be the first column
    case_concept_column = ['case:concept:name'] if 'case:concept:name' in df.columns else []
    
    # Columns starting with '#'
    hash_columns = [col for col in df.columns if col.startswith('#')]
    
    # Column 'lead_time'
    lead_time_column = ['lead_time'] if 'lead_time' in df.columns else []
    
    # Other columns (those that don't start with '#' or are 'lead_time' or 'case:concept:name')
    other_columns = [col for col in df.columns if not col.startswith('#') and col != 'lead_time' and col != 'case:concept:name']
    
    # Create the new column order: case_concept_column, then other_columns, then hash_columns, and finally lead_time
    new_column_order = case_concept_column + other_columns + hash_columns + lead_time_column
    
    # Reorder the DataFrame columns
    df = df[new_column_order]
    
    return df

def re_add_attributes(df, log_path, trace_attr, parse_dates):

    real_df = read_data(log_path, start_col=parse_dates[0], date_format="%Y-%m-%d %H:%M:%S")
    return df

def add_attr(log, trace_dict, cf_preprocessing):

    # Initialize a new column for each trace attribute
    for attr, mapping in trace_dict.items():
        log[attr] = log['case:concept:name'].map(mapping)
    
    # Put the columns you created after the first one
    log = reorder_cols(log, encoding=cf_preprocessing)

    return log

