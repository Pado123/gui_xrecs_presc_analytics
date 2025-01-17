# %%
import pandas as pd
import preprocessing_acts as pr_act
import preprocessing_times as pr_time

date_format = "%Y-%m-%d %H:%M:%S%z"
parse_dates = ['start:timestamp', 'time:timestamp']
log = pd.read_csv('../logs/hospital_log_CONFIDENTIAL.csv', 
                  date_format=date_format, parse_dates=parse_dates)

log = pr_time.add_time_features(log)
log = pr_time.add_daily_features(log)
log = pr_act.add_history(log, case_id_name='case:concept:name', 
                         activity_column_name='concept:name', 
                         encoding='last_k', last_act_num=3)

# %%
