def add_activity_outcome(log, act_to_encode, occurs_in_remaining=True):
    
    # Create a new column "occ" initialized to 0
    log[f'occ_{act_to_encode}'] = 0

    # Group the log by trace identifier
    grouped = log.groupby('case:concept:name')

    if act_to_encode == 'RADIOLOGIA ESECUZIONE': 
        # If there is an activity in a trace, in which there is RADIOLOGIA ESECUZIONE, then set "occ" to 1 for all rows in that trace
        # Iterate over each group (trace)
        log[f'occ_{act_to_encode}'] = 0

        # Iterate over each group (trace)
        for trace_id, group in grouped:
            if group['concept:name'].str.contains('RADIOLOGIA ESECUZIONE', na=False).any():
                # Set "occ" to 1 for all rows in the trace
                log.loc[group.index, f'occ_{act_to_encode}'] = 1

            """        for trace_id, group in grouped:
                        # find the first index of the activity in this trace
                        mask = group['concept:name'].str.contains('RADIOLOGIA ESECUZIONE', na=False)
                        if mask.any():
                            first_idx = group.index[mask][0]  # get the first occurrence
                            # set 1 only for the rows BEFORE that event
                            before_idx = group.index[group.index < first_idx]
                            log.loc[before_idx, 'occ_RADIOLOGIA ESECUZIONE_OC'] = 1
            """

    elif not occurs_in_remaining:
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