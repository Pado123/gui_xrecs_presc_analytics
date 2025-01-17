import pandas as pd
import pm4py

def add_history(df, case_id_name='case:concept:name', activity_column_name='concept:name',
                 encoding='aggr_hist', last_act_num=3):
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
    encoding_list = ['aggr_hist', 'last_k', 'no_hist']

    if isinstance(df, str):
        try:
            if df.endswith('.xes'):
                log = pm4py.read_xes(df)
                df = pm4py.convert_to_dataframe(log)
            else:
                df = pd.read_csv(df)
        except Exception as e:
            raise ValueError(f"Error loading dataframe from path: {e}")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame or a string path to a CSV/XES file.")

    if encoding not in encoding_list:
        raise ValueError(f"Unknown encoding: {encoding}, possible values are: {encoding_list}")

    if encoding == 'last_k' and not isinstance(last_act_num, int):
        raise ValueError(f"last_act_num must be an integer, got {type(last_act_num).__name__}")

    if encoding == 'aggr_hist':
        for activity in df[activity_column_name].unique():
            df[f"# {activity_column_name}={activity}"] = 0
            # First put 1 in correspondence to each activity
            df.loc[df[activity_column_name] == activity, f"# {activity_column_name}={activity}"] = 1
            # Sum the count from the previous events
            df[f"# {activity_column_name}={activity}"] = \
                df.groupby(case_id_name)[f"# {activity_column_name}={activity}"].cumsum()
        return df

    if encoding == 'last_k':
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
        return df

    if encoding == 'no_hist':  
        return df
