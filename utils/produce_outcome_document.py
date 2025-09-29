import json
import os
from sklearn.metrics import f1_score
import glob

def extract_clean_reasoning(raw_answer):
    try:
        reasoning_section = raw_answer.split('[[ ## reasoning ## ]]')[1]
        reasoning_only = reasoning_section.split('[[ ## answer ## ]]')[0]
        cleaned = reasoning_only.replace('\\n', ' ') \
                                 .replace('\\t', ' ') \
                                 .replace('\n', ' ') \
                                 .replace('\t', ' ')
        cleaned = ' '.join(cleaned.split())
        return cleaned
    except Exception:
        return None

def extract_predictions_and_labels(data):
    results = []
    first_key = list(data.keys())[0]
    for test_seed, test_content in data[first_key]["test_seeds"].items():
        for train_seed, train_content in test_content["train_seeds"].items():
            predictions = train_content["predictions"]
            outputs = train_content["output_text"]
            
            for i, output in enumerate(outputs):
                # Extract the model's prediction from the text
                try:
                    pred_text = output.split("[[ ## answer ## ]]")[1].split("[[ ## completed ## ]]")[0].strip()
                    predicted_from_text = int(pred_text)
                except (IndexError, ValueError):
                    predicted_from_text = None

                # Get the actual label from the list
                actual = test_content['aggregated_metrics']['True_values'][i]

                results.append({
                    "predicted_from_logits": predictions[i],
                    "predicted_from_text": predicted_from_text,
                    "actual_label": actual,
                    "text": extract_clean_reasoning(output.strip())
                })
    # Add model configuration to results as first element of the list
    results.insert(0, data['model_config'])
    return results

def read_and_process_case_studies(case_studies=['bac', 'hospital', 'bpi12']):

    for case_study in case_studies:

        print(f"Processing case study: {case_study}")
        base_folder = f'/home/padela/Scrivania/LLMs/gui_xrecs_presc_analytics/llmp/output/pm/{case_study}'
        json_files = glob.glob("*.json")

        # For it JSON file in the folder, read it, preprocess it and save the results
        for json_file in json_files:
            with open(f"{base_folder}/{json_file}", 'r') as f:
                data = json.load(f)

            results = extract_predictions_and_labels(data)

            # Save the results to a new JSON file in the outcome_results folder
            output_file = f"/home/padela/Scrivania/LLMs/gui_xrecs_presc_analytics/outcomes_sub/{case_study}/{json_file}"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)

            print(f"Results saved to {output_file}")

# read_and_process_case_studies()


def concat_skip_first_multiple(directory, case_study, names_list):

    """ for every element of the list:
    - read the json file with the corresponding name, 
    - skip the first element of the list and concatenate the rest of the elements of the list, if the element is not the first one
    - return the concatenated list
    """

    merged_list = []
    for i, name in enumerate(names_list):
        file_path = os.path.join(directory, case_study, name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Skip the first element and concatenate the rest
                if len(data) > 1:
                    if i == 0:
                        merged_list.extend(data)
                    else:
                        merged_list.extend(data[1:])
                else:
                    print(f"Warning: File {name} has no data beyond the first element.")
        else:
            print(f"Warning: File {name} does not exist in the directory {directory}.")

    return merged_list
    

def evaluate_fscore(data):
    """
    Calculate the F1-score between predicted and actual labels
    from a list of dictionaries, skipping the first element.

    Args:
        data (list of dict): Each dict contains 'predicted_from_text' and 'actual_label'

    Returns:
        float: F1-score
    """
    if not data or len(data) <= 1:
        raise ValueError("List must contain at least two entries (first one is skipped).")
    
    # Skip the first entry
    relevant_data = data[1:]

    # Extract predictions and actual labels, prefer text-derived when valid else logits
    y_pred = []
    y_true = []
    for item in relevant_data:
        actual = item.get("actual_label")
        pred_text = item.get("predicted_from_text")
        pred_logits = item.get("predicted_from_logits")
        try:
            if pred_text is not None:
                pred_val = int(round(pred_text))
            else:
                pred_val = int(round(pred_logits))
        except Exception:
            # Skip items with invalid predictions
            continue
        try:
            actual_val = int(actual)
        except Exception:
            continue
        y_pred.append(pred_val)
        y_true.append(actual_val)

    return f1_score(y_true, y_pred, average='weighted')


def merge_json_files(folder_path, case_study, output_filename="merged_n100_hashed.json"):
    """
    Merge all JSON files in a folder that contain both 'n100' and 'hashed' in their names,
    then save the merged content into a new JSON file.

    Args:
        folder_path (str): Path to the folder containing JSON files.
        output_filename (str): Name of the merged output file.

    Returns:
        str: Path to the merged JSON file.
    """
    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter only .json files
    json_files = [f for f in files if f.endswith(".json")]

    # Keep only files with both 'n100' and 'hashed'
    target_files = [f for f in json_files if "n100" in f and "hashed" in f]

    if not target_files:
        raise FileNotFoundError("No JSON files containing both 'n100' and 'hashed' were found.")

    merged_data = concat_skip_first_multiple(os.path.dirname(folder_path), case_study, target_files)

    return merged_data


def produce_final_outcomes(directory='/home/padela/Scrivania/LLMs/gui_xrecs_presc_analytics/outcomes_sub'):
    
    #Read every subfolder in the directory
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    for subfolder in subfolders:
        
        # Get the case study, that is the name of the subfolder
        case_study = os.path.basename(subfolder)
        print(f"Processing case study: {case_study}")
        
        # Do an elif in which you explore the presence or not of the string 'n100' in the name of the subfolder
        n100_hashed = [el for el in os.listdir(subfolder) if 'n100' in el and 'hashed' in el]
        n10_hashed = [el for el in os.listdir(subfolder) if 'n10' in el and 'hashed' in el] 
        n100 = [el for el in os.listdir(subfolder) if 'n100' in el and 'hashed' not in el]
        n10 = [el for el in os.listdir(subfolder) if 'n10' in el and 'hashed' not in el]

        # for opt_name in [n100_hashed, n10_hashed, n100, n10]:
        #     merged_list = concat_skip_first_multiple(directory, case_study, opt_name)
        #     f1_score = evaluate_fscore(merged_list)
        #     merged_list[0]['f1_score'] = f1_score
        #     print(f'The f-score for {case_study} - {opt_name} is: {evaluate_fscore(merged_list)}')
        #     # Save the merged list to a new JSON file
        #     output_file = os.path.join(directory, case_study, f'{opt_name}.json')
        #     with open(output_file, 'w', encoding='utf-8') as f:
        #         json.dump(merged_list, f, indent=4, ensure_ascii=False)
        #     print(f'Merged file saved to: {output_file}')
        #     print(f'\n') #TODO: aggiungi f-score e salva in a file

        options = {
            # "n100_hashed": n100_hashed,
            # "n10_hashed": n10_hashed,
            "n100": n100,
            "n10": n10
        }

        for name, value in options.items():
            merged_list = concat_skip_first_multiple(directory, case_study, value)
            f1 = evaluate_fscore(merged_list) if len(merged_list) > 1 else None
            if len(merged_list) > 0 and isinstance(merged_list[0], dict):
                merged_list[0]['f1_score'] = f1
            # Keep all entries; if needed, cap via slicing with a clear constant
            print(f'The f-score for {case_study} - {name} is: {f1}, len is {len(merged_list)}')

            # Use the name in the filename
            output_file = os.path.join(directory, case_study, f'{name}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_list, f, indent=4, ensure_ascii=False)

            print(f'Merged file saved to: {output_file}\n')

if __name__ == "__main__":
    produce_final_outcomes()