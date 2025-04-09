def add_activity_outcome(log, act_to_encode, occurs_in_remaining=True):
    
    # Create a new column "occ" initialized to 0
    log[f'occ_{act_to_encode}'] = 0

    # Group the log by trace identifier
    grouped = log.groupby('case:concept:name')

    if not occurs_in_remaining:
    # Iterate over each group (trace)
        for trace_id, group in grouped:
            # Check if the activity occurs in the trace
            if act_to_encode in group['concept:name'].values:
                # Set "occ" to 1 for all rows in the trace
                log.loc[group.index, f'occ_{act_to_encode}'] = 1

    else:
        # Iterate over each group (trace)
        for trace_id, group in grouped:
            # Iterate over each event in the trace
            for i in range(len(group)):
                # Check if the activity occurs in the remaining part of the trace
                occurs_in_remaining = act_to_encode in group['concept:name'].iloc[i + 1:].values
                # Assign 1 or 0 to the current event based on the occurrence in the remaining part
                log.loc[group.index[i], f'occ_{act_to_encode}'] = int(occurs_in_remaining)

    return log