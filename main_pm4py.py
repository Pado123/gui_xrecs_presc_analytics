# %% Import libraries
import pandas as pd
import pm4py 
import json


# Select the case study and read the hparams file in json
case_study = "bpi17" # Accepted values: bpi17
hparams = json.load(open(f"hparams/{case_study}.json", "r"))


# %% Import an event log
log = pm4py.read_xes(f"{hparams['log_path']}")

# %%
