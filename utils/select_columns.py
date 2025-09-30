def select_columns(hparams, df):

    if hparams['pre'] == None: 
        print('No hparams column selected, returning all the columns')
        return None
    
    elif hparams['pre'] == 'analogical_cf':
        # Return the 'concept:name' column and the ones that start with '#'
        return ['concept:name'] + [col for col in df.columns if col.startswith('#')]
    
    elif hparams['pre'] == 'analogical_att':
        # Return the columns in the hparams['trace_attr'] list
        return hparams['trace_attr']

    elif hparams['pre'] == 'path_pred':
        return [col for col in df.columns if col.startswith('#')]

    elif hparams['pre'] == 'similarity_times':
        return ['activity_duration', 'time_from_start']
    