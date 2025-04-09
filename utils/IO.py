import pandas as pd
import json 

def save_log(experiment_folder, log, encoding_cf, type=None, kpi=None):

    # Check the type of the log variable, if it is a pandas dataframe or a dict
    if isinstance(log, dict):
        output_path = f'{experiment_folder}/preprocessed_log_sequential_{type}_{kpi}.json'
    elif isinstance(log, pd.DataFrame):
        output_path = f'{experiment_folder}/preprocessed_log_{encoding_cf}_{type}_{kpi}.csv'

    # If the output path is a CSV file, save the log as a CSV
    if '.csv' in output_path:
        log = log.sort_values(by=['case:concept:name', 'time:timestamp'])
        print(f"Saving preprocessed log to '{output_path}'")
        log.to_csv(path_or_buf=output_path, index=False)
        print("Preprocessed log saved as csv")
    
    # if the file is a pickle file, save the log as a json file
    elif '.json' in output_path:
        print(f"Saving preprocessed log to '{output_path}'")
        dict_as_string = json.dumps(log, default=str)    
        with open(output_path, 'w') as file:
            file.write(dict_as_string)
        print("Preprocessed log saved as str")