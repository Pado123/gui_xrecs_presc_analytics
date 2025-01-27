### How to use it

1. Create an environment with py3.11 and the requirements in the requirements.txt file

2. Upload an event log in the "logs" folder

3. modify the parameter in the hparams folder (the hparams file has to have the same name of the log)

4. Pay attention to the "cf_preprocessing", it has to be one of the followings: ['aggr_hist', 'last_k', 'no_hist', 'sequential'] (different encodings are listed in the paper)