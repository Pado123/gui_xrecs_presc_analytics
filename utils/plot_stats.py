import matplotlib.pyplot as plt

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