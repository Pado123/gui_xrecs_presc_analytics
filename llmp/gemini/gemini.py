from pathlib import Path
import pandas as pd
import json
import time
import re
import sys
import numpy as np
from typing import List, Dict, Any
import google.generativeai as genai
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from dotenv import dotenv_values
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error, mean_absolute_error, f1_score, precision_recall_fscore_support
from utils import relative_mae, RateLimiter, json_to_dataframe, PromptLoader

class PredictionError(Exception):
    """Custom exception for prediction errors"""
    pass

class GenAI:
    """Base class for GenAI model interactions"""
    def __init__(self, api_key: str, dataset: str, mode: str):
        self.dataset = dataset
        self.mode = mode

        # Load prompts for the dataset and mode
        prompt_loader = PromptLoader(dataset)
        self.system_prompt, self.prompt_template = prompt_loader.get_prompts_for_mode(mode)

        self.model_name = "gemini-2.0-flash" # "gemini-2.5-flash-preview-04-17" # 
        self.rate_limiter = RateLimiter(calls=1, per_seconds=12)
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            self.model_name,
            system_instruction=self.system_prompt
        )

        print(f"GenAI model '{self.model_name}' initialized successfully")
        print(f"System prompt:\n{self.system_prompt}")
        print(f"Prompt template:\n{self.prompt_template}")

    def _extract_answer(self, text: str) -> float:

        # print('il testo comincia qui:\n', text)
        # print('il testo finisce qui:')

        if self.mode in {"agg", "seq"}:
            """Extract numerical answer from model response"""
            if not text or not isinstance(text, str):
                raise PredictionError("Invalid response: empty or non-string response")
                
            # More flexible pattern to catch variations in formatting
            pattern = r"\[\[ ## answer ## \]\]\s*(\d+(?:\.\d+)?)\s*\[\[ ## completed ## \]\]"
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                raise PredictionError(f"Could not find answer in response format: {text}")
            
            try:
                value = float(match.group(1))
                if pd.isna(value) or value <= 0:  # Basic validation
                    raise PredictionError("Invalid numerical answer (NaN or negative)")
                return value
            except ValueError as e:
                raise PredictionError(f"Error converting answer to float: {e}")
            

        elif self.mode in {"agg_outcomepred", "seq_outcomepred"}:

            match = re.search(r"\[\[ ## answer ## \]\]\s*(\d+(?:\.\d+)?)\s*\[\[ ## completed ## \]\]", text, re.DOTALL)

            if match:
                content = match.group(1)
                # Look for a number 0 or 1
                binary_match = re.search(r'\b[01]\b', content)
                if binary_match:
                    value = float(binary_match.group())
                    print("Extracted float value:", value)
                    return value
                else:
                    print("No 0 or 1 found in the section.")
            else:
                print("Pattern not found.")

    def predict(self, train_data: pd.DataFrame, test_instance: pd.Series) -> float:
        """Make a single prediction"""
        self.rate_limiter.wait_if_needed()

        if self.mode in {"agg", "agg_outcomepred"}:
            formatted_content = self.prompt_template.format(
                    examples=train_data.to_string(index=False),
                    test_case=test_instance.to_string(index=False)
                )
            # print(f"Formatted content:\n{formatted_content}")
        
        elif self.mode in {"seq", "seq_outcomepred"}:
            # print(f"train_data:\n{train_data}")
            train_data_text = train_data['text']
            test_instance_text = test_instance['text']
            formatted_content = self.prompt_template.format(
                examples="\n".join(train_data_text.astype(str)),
                test_case=test_instance_text
            )
            print(f"test_instance_text:\n{test_instance_text}")

        try:            
            response = self.model.generate_content(formatted_content)
            if not response or not response.text:
                raise PredictionError("Empty response from model")
            
            answer, text = self._extract_answer(response.text), response.text
            if pd.isna(answer):
                raise PredictionError("Invalid numerical answer")
            
            return answer, text
            
        except Exception as e:
            if isinstance(e, PredictionError):
                raise
            raise PredictionError(f"Error during prediction: {str(e)}")

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for logging"""
        return {
            "model_name": self.model_name,
            "system_prompt": self.system_prompt,
            "prompt_template": self.prompt_template
        }

def setup_paths(data_root: Path, output_root: Path, dataset: str) -> tuple[Path, Path]:
    """
    Setup and verify data and output paths
    
    Args:
        data_root: Root path for data
        output_root: Root path for outputs
        dataset: Name of the dataset
    
    Returns:
        tuple of data path and output path
    """
    data_path = data_root / "pm" / dataset
    data_path = data_path.resolve()
    output_path = output_root / "pm" / dataset
    
    print('data_path:', data_path)

    # Ensure data path exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    return data_path, output_path

class ExperimentManager:
    """Manages experiment execution and results collection"""
    def __init__(self, data_path: Path, output_path: Path, output_filename: str, model: GenAI, kpi: str = "lead_time",
                 base_seed: int = 42, n_seeds: int = 10):
        self.data_path = data_path
        self.output_path = output_path
        self.output_filename = output_filename
        self.model = model
        self.results = {}
        self.base_seed = base_seed
        self.n_seeds = n_seeds
        self.mode = model.mode  # Add this line to store the mode
        self.kpi = kpi  

        print(f"ExperimentManager initialized with data_path={data_path}, output_path={output_path}, output_filename={output_filename} and kpi={kpi}")

    @staticmethod
    def load_data(data_path: Path, n_samples: int, train_seed: int, test_seed: int, kpi: str, mode: str = "agg") -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess the training and test datasets with specific seeds.
        
        Args:
            data_path: Path to the experiment data
            n_samples: Number of training samples to use
            train_seed: Random seed for train split
            test_seed: Random seed for test split
            mode: Data loading mode ("agg" or "seq" in case of lead_time prediction, agg_outcomepred or seq_outcomepred in case of outcome prediction)
            kpi: Key performance indicator to load data for
        
        Returns:
            tuple of training and test DataFrames
        """
        # Determine file suffix based on mode
        file_suffix = "aggr_hist" if mode in {"agg", "agg_outcomepred"} else "sequential"

        print(f"Loading data for mode: {mode}, file_suffix: {file_suffix} and kpi: {kpi}")

        if mode in {"agg", "agg_outcomepred"}:
            # Load data for aggregation mode
            columns_to_drop = ["time:timestamp", "case:concept:name", 
                            "day_of_week", "hour_of_day"]
            
            train = pd.read_csv(data_path / f"preprocessed_log_{file_suffix}_train_{kpi}.csv").drop(columns=columns_to_drop)
            test = pd.read_csv(data_path / f"preprocessed_log_{file_suffix}_test_{kpi}.csv").drop(columns=columns_to_drop)

        elif mode in {"seq", "seq_outcomepred"}:
            train = json_to_dataframe(data_path / f"preprocessed_log_{file_suffix}_train_{kpi}.json", kpi = kpi)
            test = json_to_dataframe(data_path / f"preprocessed_log_{file_suffix}_test_{kpi}.json", test=True, kpi = kpi)
        
        train = train.sample(frac=1, random_state=train_seed).reset_index(drop=True).iloc[:n_samples]
        test = test.sample(frac=1, random_state=test_seed).reset_index(drop=True).head(5) # TODO: Cambia in base al numero di test samples che vuoi
        # print(f"test cose: {test.iloc[:,-1].values} e test shape {test.shape}")
        # print("ATTENTION ONLY 10 SAMPLES")
        # print(f"train quello che vuoi te: {train.iloc[:,-1].mean()}")

        return train, test

    def evaluate_predictions(self, y_true: List[float], y_pred: List[float]) -> tuple[float, float, float, float]:
        """Evaluate predictions with consistent numeric types"""
        
        try:
            # Convert inputs to numeric arrays
            y_true_array = np.array(y_true, dtype=np.float64)
            y_pred_array = np.array(y_pred, dtype=np.float64)
            
            if isinstance(y_pred, list):
                mae = mean_absolute_error(y_true_array, y_pred_array)
                mape = mean_absolute_percentage_error(y_true_array, y_pred_array)
                rmae = relative_mae(y_true_array, y_pred_array)
            else:
                mae = mean_absolute_error([y_true_array], [y_pred_array])
                mape = mean_absolute_percentage_error([y_true_array], [y_pred_array])
                rmae = None
            
            return mae, mape, rmae
            
        except (ValueError, TypeError) as e:
            raise PredictionError(f"Error during evaluation - numeric conversion failed: {e}")

    def run(self, sample_sizes: List[int] = [2, 10, 50, 100, 500]):
            
        if self.mode in {"agg", "seq"}:    
            """Run the complete experiment with multiple train/test splits"""
            for n_samples in sample_sizes:
                print(f"\nRunning experiments with {n_samples} training samples...")
                self.results[n_samples] = {"test_seeds": {}}
                
                # For each test seed
                for test_idx in range(self.n_seeds):
                    test_seed = self.base_seed + test_idx
                    test_results = {"train_seeds": {}, "aggregated_metrics": {}}
                    
                    # Load test data once for this seed
                    _, test_data = self.load_data(self.data_path, n_samples, test_seed, test_seed, mode=self.mode, kpi=self.kpi)
                    y_true_list = test_data.iloc[:, -1].tolist()
                    
                    all_train_metrics = {
                        "MAE": [], "MAPE": [], "RMAE": [],
                        "predictions": [], "output_text": []
                    }
                    
                    # For each train seed
                    for train_idx in range(self.n_seeds):
                        train_seed = self.base_seed + train_idx
                        print(f"\nProcessing test_seed={test_seed}, train_seed={train_seed}")
                        
                        train_data, _ = self.load_data(self.data_path, n_samples, train_seed, test_seed, mode=self.mode, kpi=self.kpi)
                        # print(f"train_data values:\n{train_data.iloc[:,-1].values}")
                        predictions_list = []
                        output_text_list = []
                        aes, apes = [], []
                        
                        for idx in tqdm(range(len(test_data)), 
                                    desc=f"Processing samples (n={n_samples}, test={test_idx}, train={train_idx})"):
                            test_instance = test_data.iloc[idx, :-1]
                            y_true = test_data.iloc[idx, -1]
                            
                            max_retries = 12 #TODO: Questo parametro si può cambiare
                            retry_count = 0
                            while retry_count < max_retries:
                                try:
                                    prediction, output_text = self.model.predict(train_data, test_instance)
                                    predictions_list.append(prediction)
                                    output_text_list.append(output_text)
                                    ae, ape, _ = self.evaluate_predictions(y_true, prediction)
                                    aes.append(ae)
                                    apes.append(ape)
                                    break
                                except PredictionError as e:
                                    retry_count += 1
                                    if retry_count == max_retries:
                                        raise RuntimeError(f"Failed to get prediction after {max_retries} attempts: {str(e)}")
                                    print(f"\nRetrying prediction ({retry_count}/{max_retries}) due to error: {str(e)}")
                                    time.sleep(60 * retry_count**2)  # Add small delay between retries
                        
                        mae, mape, rmae = self.evaluate_predictions(y_true_list, predictions_list)
                        
                        # Store results for this train seed
                        train_results = {
                            "predictions": predictions_list,
                            "output_text": output_text_list,
                            "metrics": {
                                "AE": aes,
                                "APE": apes,
                                "MAE": mae,
                                "MAPE": mape,
                                "RMAE": rmae
                            }
                        }
                        test_results["train_seeds"][train_seed] = train_results

                        # Print results for this train seed
                        
                        print(f"Results for train_seed={train_seed}:")
                        print(f"MAE: {mae:.2f}, MAPE: {mape:.2f}, RMAE: {rmae:.2f}")

                        # Collect metrics for averaging
                        all_train_metrics["MAE"].append(mae)
                        all_train_metrics["MAPE"].append(mape)
                        all_train_metrics["RMAE"].append(rmae)
                        all_train_metrics["predictions"].append(predictions_list)
                        all_train_metrics["output_text"].append(output_text_list)
                    
                    # Compute aggregated metrics for this test seed
                    test_results["aggregated_metrics"] = {
                        "mean_MAE": np.mean(all_train_metrics["MAE"]),
                        "std_MAE": np.std(all_train_metrics["MAE"]),
                        "cv_MAE": (np.std(all_train_metrics["MAE"]) / np.mean(all_train_metrics["MAE"])) * 100,
                        "mean_MAPE": np.mean(all_train_metrics["MAPE"]),
                        "std_MAPE": np.std(all_train_metrics["MAPE"]),
                        "cv_MAPE": (np.std(all_train_metrics["MAPE"]) / np.mean(all_train_metrics["MAPE"])) * 100,
                        "mean_RMAE": np.mean(all_train_metrics["RMAE"]),
                        "std_RMAE": np.std(all_train_metrics["RMAE"]),
                        "cv_RMAE": (np.std(all_train_metrics["RMAE"]) / np.mean(all_train_metrics["RMAE"])) * 100,
                    }
                    
                    # Store results for this test seed
                    self.results[n_samples]["test_seeds"][test_seed] = test_results
                    
                    print(f"\nResults for test_seed={test_seed}:")
                    print(f"Mean MAE: {test_results['aggregated_metrics']['mean_MAE']:.2f} ± {test_results['aggregated_metrics']['std_MAE']:.2f} (CV: {test_results['aggregated_metrics']['cv_MAE']:.2f}%)")
                    print(f"Mean MAPE: {test_results['aggregated_metrics']['mean_MAPE']:.2f} ± {test_results['aggregated_metrics']['std_MAPE']:.2f} (CV: {test_results['aggregated_metrics']['cv_MAPE']:.2f}%)")
                    print(f"Mean RMAE: {test_results['aggregated_metrics']['mean_RMAE']:.2f} ± {test_results['aggregated_metrics']['std_RMAE']:.2f} (CV: {test_results['aggregated_metrics']['cv_RMAE']:.2f}%)")
                
                # Compute final aggregated metrics across all test seeds
                all_test_maes = []
                all_test_mapes = []
                all_test_rmaes = []
                
                for test_seed_results in self.results[n_samples]["test_seeds"].values():
                    all_test_maes.append(test_seed_results["aggregated_metrics"]["mean_MAE"])
                    all_test_mapes.append(test_seed_results["aggregated_metrics"]["mean_MAPE"])
                    all_test_rmaes.append(test_seed_results["aggregated_metrics"]["mean_RMAE"])
                
                self.results[n_samples]["final_metrics"] = {
                    "mean_MAE": np.mean(all_test_maes),
                    "std_MAE": np.std(all_test_maes),
                    "cv_MAE": (np.std(all_test_maes) / np.mean(all_test_maes)) * 100,
                    "mean_MAPE": np.mean(all_test_mapes),
                    "std_MAPE": np.std(all_test_mapes),
                    "cv_MAPE": (np.std(all_test_mapes) / np.mean(all_test_mapes)) * 100,
                    "mean_RMAE": np.mean(all_test_rmaes),
                    "std_RMAE": np.std(all_test_rmaes),
                    "cv_RMAE": (np.std(all_test_rmaes) / np.mean(all_test_rmaes)) * 100,
                }
                
                print(f"\nFinal aggregated results for n_samples={n_samples}:")
                print(f"Mean MAE: {self.results[n_samples]['final_metrics']['mean_MAE']:.2f} ± {self.results[n_samples]['final_metrics']['std_MAE']:.2f} (CV: {self.results[n_samples]['final_metrics']['cv_MAE']:.2f}%)")
                print(f"Mean MAPE: {self.results[n_samples]['final_metrics']['mean_MAPE']:.2f} ± {self.results[n_samples]['final_metrics']['std_MAPE']:.2f} (CV: {self.results[n_samples]['final_metrics']['cv_MAPE']:.2f}%)")
                print(f"Mean RMAE: {self.results[n_samples]['final_metrics']['mean_RMAE']:.2f} ± {self.results[n_samples]['final_metrics']['std_RMAE']:.2f} (CV: {self.results[n_samples]['final_metrics']['cv_RMAE']:.2f}%)")
        
        elif self.mode in {"agg_outcomepred", "seq_outcomepred"}:
            """Run the complete experiment with multiple train/test splits"""
            for n_samples in sample_sizes:
                print(f"\nRunning experiments with {n_samples} training samples...")
                self.results[n_samples] = {"test_seeds": {}}
                
                # For each test seed
                for test_idx in range(self.n_seeds):
                    test_seed = self.base_seed + test_idx
                    test_results = {"train_seeds": {}, "aggregated_metrics": {}}
                    
                    # Load test data once for this seed
                    _, test_data = self.load_data(self.data_path, n_samples, test_seed, test_seed, mode=self.mode, kpi=self.kpi)
                    y_true_list = test_data.iloc[:, -1].tolist()

                    all_train_metrics = {
                        "Precision": [], "Recall": [], "F1": [],
                        "predictions": [], "output_text": []
                    }

                                        # For each train seed
                    for train_idx in range(self.n_seeds):
                        train_seed = self.base_seed + train_idx
                        print(f"\nProcessing test_seed={test_seed}, train_seed={train_seed}")
                        
                        train_data, _ = self.load_data(self.data_path, n_samples, train_seed, test_seed, mode=self.mode, kpi=self.kpi)
                        print(f"train_data values:\n{train_data.iloc[:,-1].values}")
                        predictions_list = []
                        output_text_list = []
                        precision_list, recall_list, f1_list = [], [], []   

                        for idx in tqdm(range(len(test_data)), 
                                    desc=f"Processing samples (n={n_samples}, test={test_idx}, train={train_idx})"):
                            test_instance = test_data.iloc[idx, :-1]
                            y_true = test_data.iloc[idx, -1]

                            max_retries = 5 #TODO: Questo parametro si può cambiare
                            retry_count = 0
                            while retry_count < max_retries:
                                try:
                                    prediction, output_text = self.model.predict(train_data, test_instance)
                                    predictions_list.append(prediction)
                                    output_text_list.append(output_text)
                                    break
                                except PredictionError as e:
                                    retry_count += 1
                                    if retry_count == max_retries:
                                        raise RuntimeError(f"Failed to get prediction after {max_retries} attempts: {str(e)}")
                                    print(f"\nRetrying prediction ({retry_count}/{max_retries}) due to error: {str(e)}")
                                    time.sleep(60 * retry_count**2)  # Add small delay between retries

                        print('y_test_trues', y_true_list)
                        print('y_test_preds', predictions_list)
                        precision, recall, f1 = precision_recall_fscore_support([int(i) for i in y_true_list], [int(i) for i in predictions_list], average='weighted')[:3]
                        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

                        # Store results for this train seed
                        train_results = {
                            "predictions": predictions_list,
                            "output_text": output_text_list,
                            "metrics": {
                                "Precision": precision,
                                "Recall": recall,
                                "F1": f1
                            }
                        }
                        test_results["train_seeds"][train_seed] = train_results
                        # Print results for this train seed
                        print(f"Results for train_seed={train_seed}:")
                        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
                        # Collect metrics for averaging
                        all_train_metrics["Precision"].append(precision)
                        all_train_metrics["Recall"].append(recall)
                        all_train_metrics["F1"].append(f1)
                        all_train_metrics["predictions"].append(predictions_list)
                        all_train_metrics["output_text"].append(output_text_list)

                    # Compute aggregated metrics for this test seed
                    test_results["aggregated_metrics"] = {
                        "mean_Precision": np.mean(all_train_metrics["Precision"]),
                        "std_Precision": np.std(all_train_metrics["Precision"]),
                        "cv_Precision": (np.std(all_train_metrics["Precision"]) / np.mean(all_train_metrics["Precision"])) * 100,
                        "mean_Recall": np.mean(all_train_metrics["Recall"]),
                        "std_Recall": np.std(all_train_metrics["Recall"]),
                        "cv_Recall": (np.std(all_train_metrics["Recall"]) / np.mean(all_train_metrics["Recall"])) * 100,
                        "mean_F1": np.mean(all_train_metrics["F1"]),
                        "std_F1": np.std(all_train_metrics["F1"]),
                        "cv_F1": (np.std(all_train_metrics["F1"]) / np.mean(all_train_metrics["F1"])) * 100,
                    }
                    # Store results for this test seed
                    self.results[n_samples]["test_seeds"][test_seed] = test_results
                    print(f"\nResults for test_seed={test_seed}:")
                    print(f"Mean Precision: {test_results['aggregated_metrics']['mean_Precision']:.2f} ± {test_results['aggregated_metrics']['std_Precision']:.2f} (CV: {test_results['aggregated_metrics']['cv_Precision']:.2f}%)")
                    print(f"Mean Recall: {test_results['aggregated_metrics']['mean_Recall']:.2f} ± {test_results['aggregated_metrics']['std_Recall']:.2f} (CV: {test_results['aggregated_metrics']['cv_Recall']:.2f}%)")
                    print(f"Mean F1: {test_results['aggregated_metrics']['mean_F1']:.2f} ± {test_results['aggregated_metrics']['std_F1']:.2f} (CV: {test_results['aggregated_metrics']['cv_F1']:.2f}%)")
                # Compute final aggregated metrics across all test seeds
                all_test_precisions = []
                all_test_recalls = []
                all_test_f1s = []
                for test_seed_results in self.results[n_samples]["test_seeds"].values():
                    all_test_precisions.append(test_seed_results["aggregated_metrics"]["mean_Precision"])
                    all_test_recalls.append(test_seed_results["aggregated_metrics"]["mean_Recall"])
                    all_test_f1s.append(test_seed_results["aggregated_metrics"]["mean_F1"])
                self.results[n_samples]["final_metrics"] = {
                    "mean_Precision": np.mean(all_test_precisions),
                    "std_Precision": np.std(all_test_precisions),
                    "cv_Precision": (np.std(all_test_precisions) / np.mean(all_test_precisions)) * 100,
                    "mean_Recall": np.mean(all_test_recalls),
                    "std_Recall": np.std(all_test_recalls),
                    "cv_Recall": (np.std(all_test_recalls) / np.mean(all_test_recalls)) * 100,
                    "mean_F1": np.mean(all_test_f1s),
                    "std_F1": np.std(all_test_f1s),
                    "cv_F1": (np.std(all_test_f1s) / np.mean(all_test_f1s)) * 100,
                }
                print(f"\nFinal aggregated results for n_samples={n_samples}:")
                print(f"Mean Precision: {self.results[n_samples]['final_metrics']['mean_Precision']:.2f} ± {self.results[n_samples]['final_metrics']['std_Precision']:.2f} (CV: {self.results[n_samples]['final_metrics']['cv_Precision']:.2f}%)")
                print(f"Mean Recall: {self.results[n_samples]['final_metrics']['mean_Recall']:.2f} ± {self.results[n_samples]['final_metrics']['std_Recall']:.2f} (CV: {self.results[n_samples]['final_metrics']['cv_Recall']:.2f}%)")
                print(f"Mean F1: {self.results[n_samples]['final_metrics']['mean_F1']:.2f} ± {self.results[n_samples]['final_metrics']['std_F1']:.2f} (CV: {self.results[n_samples]['final_metrics']['cv_F1']:.2f}%)")
                print(f"F1 - Precision - recall: {self.results[n_samples]['final_metrics']['mean_F1']:.2f} - {self.results[n_samples]['final_metrics']['mean_Precision']:.2f} - {self.results[n_samples]['final_metrics']['mean_Recall']:.2f}")            
                print(f"Std F1 - std Precision - std recall: {self.results[n_samples]['final_metrics']['std_F1']:.2f} - {self.results[n_samples]['final_metrics']['std_Precision']:.2f} - {self.results[n_samples]['final_metrics']['std_Recall']:.2f}")    
        
        
        # Add model configuration to results
        self.results["model_config"] = self.model.get_config()
        
        # Save results
        output_file = self.output_path / self.output_filename
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"\nResults saved to: {output_file}")

def main():
    """Main execution function"""
    from utils import create_argument_parser, ExperimentArgs
    
    # Parse arguments using the new parser from utils
    parser = create_argument_parser()
    args = ExperimentArgs.from_args(parser.parse_args())
    print(f"Running experiment with args: {args}")
    
    # Setup paths
    root_dir = "/home/padela/Scrivania/LLMs/gui_xrecs_presc_analytics/llmp" # "/home/padela/Desktop/LLMs_PM/llmp" #
    root_dir = Path(root_dir)
    
    output_filename = f"gemini_results_multiple_seeds_n{args.sample_size}{'_seq' if args.mode == 'seq' else ''}.json"
    
    data_path, output_path = setup_paths(
        data_root=root_dir / "data",
        output_root=root_dir / "output",
        dataset=args.dataset
    )
    
    # Create log file for stdout and stderr
    log_file = output_path / f"gemini_n{args.sample_size}{'_seq' if args.mode == 'seq' else ''}.log"
    # assert not log_file.exists(), f"Log file already exists: {log_file}" #TODO: Uncomment this line to check if the log file already exists
    sys.stdout = sys.stderr = open(log_file, 'w')
    
    config = dotenv_values(Path(__file__).parent.parent / ".env")
    api_key = config[args.api_key_name]
    print(f"Using API key: {args.api_key_name}")
    model = GenAI(api_key, args.dataset, args.mode)
    
    experiment = ExperimentManager(data_path, output_path, output_filename, model, n_seeds=5, kpi=args.kpi) #TODO: Cambia il numero di seeds
    experiment.run(sample_sizes=[args.sample_size])
    
    # Close the log file
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

if __name__ == "__main__":
    main()