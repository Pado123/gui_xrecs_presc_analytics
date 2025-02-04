import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error


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