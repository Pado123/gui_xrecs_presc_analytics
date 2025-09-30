import pandas as pd
import numpy as np

def filter_and_compute_mean(dataset, new_row, mode='mean'):
    """
    Filter dataset rows where values in all columns except the last one are greater 
    or equal to corresponding values in the new row, then return the mean of 
    the last column for filtered rows.
    
    Parameters:
    -----------
    dataset : pandas.DataFrame
        The main dataset to filter
    new_row : pandas.Series or dict or pandas.DataFrame (single row)
        The reference row with same columns as dataset
    mode : str
        The mode to use for the prediction
        'mean' : return the mean of the last column for rows that meet the filtering criteria
        'median' : return the median of the last column for rows that meet the filtering criteria
        
    Returns:
    --------
    float
        Mean of the last column for rows that meet the filtering criteria
        Returns NaN if no rows meet the criteria
    """
    # Convert new_row to pandas Series if it's not already
    if isinstance(new_row, dict):
        new_row = pd.Series(new_row)
    elif isinstance(new_row, pd.DataFrame):
        if len(new_row) != 1:
            raise ValueError("new_row DataFrame must contain exactly one row")
        new_row = new_row.iloc[0]
    
    # Get all columns except the last one (which is used for outcome)
    filter_columns = dataset.columns[:-1]
    
    if len(filter_columns) == 0:
        raise ValueError("Dataset must have at least 2 columns (one for filtering, one for outcome)")
    
    # Check that new_row has all the filter columns
    missing_cols = [col for col in filter_columns if col not in new_row.index]
    if missing_cols:
        raise ValueError(f"new_row is missing columns: {missing_cols}")
    
    # Create filter condition: all filter columns in dataset >= corresponding values in new_row
    filter_condition = pd.Series([True] * len(dataset))
    
    for col in filter_columns:
        filter_condition &= (dataset[col] >= new_row[col])
    
    # Apply filter
    filtered_dataset = dataset[filter_condition]
    
    if len(filtered_dataset) == 0:
        # print('No rows meet the criteria, returning the mean of the dataset')
        return np.mean(dataset[dataset.columns[-1]]), 1
    
    # Get the last column (outcome column)
    outcome_column = dataset.columns[-1]
    
    # Return mean of the outcome column for filtered rows
    if mode == 'mean':
        return filtered_dataset[outcome_column].mean(), 0
    elif mode == 'median':
        return filtered_dataset[outcome_column].median(), 0
