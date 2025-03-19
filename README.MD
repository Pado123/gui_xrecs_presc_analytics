# Using LLMs for enhancing Predictive Process Monitoring in Event Log Small-Scaled scenarios.
Code related to the implementation of the paper ''Enhancing Predictive Process Monitoring on Small-Scale Event Logs Using LLMs''

The code has been developed and tested on Both Windows 11 and Linux 22.04 LTS. With a Python 3.11 environment. The laptop running the code has 16GB of RAM and an Intel Core i7-10750H CPU @ 2.60GHz × 12. For storing all the Profiles generated, we used the SSD memory of the laptot, with 1TB of space. 

The code supports the use of 10 Gemini API keys, that can be retrieved at https://deepmind.google/technologies/gemini/flash-thinking/

# How to run the code

1. Create an environment with python3.11 and the requirements in the requirements.txt file

2. Upload an event log in the "logs" folder

3. modify the parameters in the hparams folder (the hparams file has to have the same name of the log)

4. Pay attention to the "cf_preprocessing", it has to be one of the followings: ['aggr_hist', 'no_hist', 'sequential'] (different encodings are listed in the paper). 

5. Encode the logs using the code main_pm4py.py, changinge the parameter exp_name in the script and runing it using "python3 main_pm4py.py", encoded logs will be shown in the "experiments" folder

6. PAOLO MISSING

7. For visualizing prompts and results PAOLO MISSING

8. For the Catboost benchmark, change the parameter exp_name in the script tests/test_catboost.py and run it using "python3 tests/test_catboost.py"

                  