#%%
import json
import os
import glob

def extract_clean_reasoning(raw_answer):
    try:
        # Step 1: Extract the reasoning block
        reasoning_section = raw_answer.split('[[ ## reasoning ## ]]')[1]
        reasoning_only = reasoning_section.split('[[ ## answer ## ]]')[0]

        # Step 2: Clean up escape characters and extra spaces
        cleaned = reasoning_only.replace('\\n', ' ').replace('\\t', ' ')
        # Step 2: Clean text
        cleaned = reasoning_only.replace('\\n', ' ') \
                                 .replace('\\t', ' ') \
                                 .replace('\n', ' ') \
                                 .replace('\t', ' ') \
                                 .replace("\\'", "'") \
                                 .replace('\'', '"')        
        cleaned = ' '.join(cleaned.split())  # Remove extra spaces

        return cleaned
    except Exception:
        return None  # Return None for malformed input

def extract_predictions_and_labels(data):
    results = []
    for test_seed, test_content in data["100"]["test_seeds"].items():
        for train_seed, train_content in test_content["train_seeds"].items():
            predictions = train_content["predictions"]
            outputs = train_content["output_text"]
            
            for i, output in enumerate(outputs):
                # Extract the model's prediction from the text
                try:
                    pred_text = output.split("[[ ## answer ## ]]")[1].split("[[ ## completed ## ]]")[0].strip()
                    predicted = int(pred_text)
                except (IndexError, ValueError):
                    predicted = None  # Handle malformed prediction

                # Get the actual label from the list
                actual = test_content['aggregated_metrics']['True_values'][i]

                results.append({
                    # "test_seed": test_seed,
                    # "train_seed": train_seed,
                    # "index": i,
                    "predicted_from_text": predictions[i],
                    "actual_label": actual,
                    "text": extract_clean_reasoning(output.strip())
                })
    return results

def read_and_process_case_studies(case_studies=['bac', 'hospital', 'bpi12']):

    for case_study in case_studies:

        print(f"Processing case study: {case_study}")
        base_folder = f'/home/padela/Desktop/LLMs_PM/llmp/output/pm/{case_study}'
        os.chdir(base_folder)
        json_files = glob.glob("*.json")

        # For it JSON file in the folder, read it, preprocess it and save the results
        for json_file in json_files:
            with open(f"{base_folder}/{json_file}", 'r') as f:
                data = json.load(f)

            results = extract_predictions_and_labels(data)

            # Save the results to a new JSON file in the outcome_results folder
            output_file = f"/home/padela/Desktop/LLMs_PM/outcome_results/{case_study}/{json_file}"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)

            print(f"Results saved to {output_file}")


read_and_process_case_studies()



# %%
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer


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

returns_acts_freq("/home/padela/Desktop/LLMs_PM/logs/bpi17_w.xes")
# %%
