import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

def relative_mae_sklearn(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) / np.mean(y_true)

def plot_activity_duration(log):
    
    if 'activity_duration' not in log.columns:
        raise ValueError("The log does not contain the activity_duration column.")

    #PLot the hist of the activity duration with 100 bins
    log['activity_duration'].hist(bins=100)

    #PLot the 95th percentile of the activity duration
    plt.axvline(log['activity_duration'].quantile(0.95), color='r')

    #plot the trace duration with 100 bins
    log['trace_duration'].hist(bins=100)

    #PLot the 95th percentile of the trace duration
    plt.axvline(log['trace_duration'].quantile(0.99), color='r')

def relative_mae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mean_true = np.mean(y_true)
    return mae / mean_true

def returns_acts_freq(log_input, activity_name: str = 'concept:name', case_id_name: str = 'case:concept:name'):
    """
    Returns the frequency (as a percentage) of activities in the log, based on the number of cases (traces)
    in which each activity appears at least once.

    :param log_input: Path to the event log file (.xes, .csv, .parquet) or a pandas DataFrame.
    :param activity_name: Column name for activities.
    :param case_id_name: Column name for case IDs.
    :return: Dictionary {activity_name: frequency_percentage}
    """

    # Load the log based on input type
    if isinstance(log_input, str):
        if log_input.endswith('.csv'):
            try:
                log = pd.read_csv(log_input, header=0, low_memory=False)
            except UnicodeDecodeError:
                log = pd.read_csv(log_input, header=0, encoding="cp1252", low_memory=False)
        elif log_input.endswith('.xes'):
            log = xes_importer.apply(log_input)
            log = pm4py.convert_to_dataframe(log)
        elif log_input.endswith('.parquet'):
            log = pd.read_parquet(log_input, engine='pyarrow')
        else:
            raise ValueError("Unsupported file type. Please provide a .xes, .csv, or .parquet file.")
    elif isinstance(log_input, pd.DataFrame):
        log = log_input
    else:
        raise TypeError("Input must be a file path or a pandas DataFrame.")

    # Compute frequency of each activity based on unique traces
    activity_freq = log.groupby(activity_name)[case_id_name].nunique().to_dict()

    # Convert to percentages
    total_traces = log[case_id_name].nunique()
    for activity in activity_freq:
        activity_freq[activity] = (activity_freq[activity] / total_traces) * 100

    # Sort by frequency descending
    activity_freq = dict(sorted(activity_freq.items(), key=lambda item: item[1], reverse=True))

    # Print the results
    for activity, freq in activity_freq.items():
        print(f"Activity: {activity}, Frequency: {freq:.2f}%")

    return activity_freq

# returns_acts_freq("logs/bpi12w.xes")

# %%

def plot_last_column_histogram(file_path, bins=50, title=None, figsize=(10, 6)):
    """
    Plot histogram distribution of the last column in a dataset.
    
    :param file_path: Path to the dataset file (.csv, .xes, .parquet)
    :param bins: Number of bins for the histogram (default: 50)
    :param title: Custom title for the plot (optional)
    :param figsize: Figure size as tuple (width, height)
    :return: None (displays the plot)
    """
    
    # Load the dataset based on file extension
    if file_path.endswith('.csv'):
        try:
            data = pd.read_csv(file_path, header=0, low_memory=False)
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, header=0, encoding="cp1252", low_memory=False)
    elif file_path.endswith('.xes'):
        log = xes_importer.apply(file_path)
        data = pm4py.convert_to_dataframe(log)
    elif file_path.endswith('.parquet'):
        data = pd.read_parquet(file_path, engine='pyarrow')
    else:
        raise ValueError("Unsupported file type. Please provide a .xes, .csv, or .parquet file.")
    
    # Get the last column
    last_column = data.iloc[:, -1]
    column_name = data.columns[-1]
    
    # Create the histogram plot
    plt.figure(figsize=figsize)
    plt.hist(last_column.dropna(), bins=bins, alpha=0.7, edgecolor='black')
    
    # Set labels and title
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Histogram Distribution of {column_name}')
    
    # Add some statistics to the plot
    mean_val = last_column.mean()
    median_val = last_column.median()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print some basic statistics
    print(f"Statistics for column '{column_name}':")
    print(f"Count: {last_column.count()}")
    print(f"Mean: {mean_val:.4f}")
    print(f"Median: {median_val:.4f}")
    print(f"Std Dev: {last_column.std():.4f}")
    print(f"Min: {last_column.min():.4f}")
    print(f"Max: {last_column.max():.4f}")

# plot_last_column_histogram('/home/padela/Scrivania/LLMs/gui_xrecs_presc_analytics/experiments/bpi12/preprocessed_log_aggr_hist_train_lead_time.csv')
# plot_last_column_histogram('/home/padela/Scrivania/LLMs/gui_xrecs_presc_analytics/experiments/bpi12/preprocessed_log_aggr_hist_test_lead_time.csv')
# %%
