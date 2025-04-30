import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentArgs:
    sample_size: int
    mode: str
    k2: bool
    dataset: str
    api_key_name: Optional[str] = None
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ExperimentArgs':
        # If api_key_name is explicitly provided, use it directly
        if hasattr(args, 'api_key_name') and args.api_key_name:
            return cls(
                sample_size=args.sample_size,
                mode=args.mode,
                k2=args.k2,
                dataset=args.dataset,
                api_key_name=args.api_key_name
            )
        
        # Otherwise, use the mapping based on sample size
        # Define API key mapping based on k2 flag
        if not args.k2:
            sample_sizes_to_api = {
                2: "GOOGLE_API",
                10: "GOOGLE_API_AP",
                50: "GOOGLE_API_AP2",
                100: "GOOGLE_API_GO",
                500: "GOOGLE_API_FV"
            }
        else:
            sample_sizes_to_api = {
                2: "GOOGLE_API_AP3",
                10: "GOOGLE_API_PF2",
                50: "GOOGLE_API_AP5",
                100: "GOOGLE_API_AP6",
                500: "GOOGLE_API_AP7"
            }
        
        return cls(
            sample_size=args.sample_size,
            mode=args.mode,
            k2=args.k2,
            dataset=args.dataset,
            api_key_name=sample_sizes_to_api[args.sample_size]
        )

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser for experiment settings"""
    parser = argparse.ArgumentParser(
        description="Run Gemini experiments with different sample sizes"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        choices=[2, 10, 50, 100, 500],
        help="Number of training samples to use (must be one of: 2, 10, 50, 100, 500)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["bac", "bpi12", "bpi17", "hospital", "purchasing", "production"],
        help="Dataset to use for the experiment"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["agg", "seq"],
        default="agg",
        help="Mode to run the experiment in (agg=aggregated or seq=sequential)"
    )
    parser.add_argument(
        "--k2",
        action="store_true",
        default=False,
        help="Whether to use the second set of API keys"
    )
    parser.add_argument(
        "--api_key_name",
        type=str,
        help="Explicitly specify API key name to use (overrides automatic selection)"
    )
    return parser

def read_json_file(file_path):
    """
    Read and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Parsed JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        raise ValueError(f"Error reading JSON file: {e}")

def json_to_lines(json_data):
    """
    Convert JSON data to string format with each entry on a new line.
    
    Args:
        json_data (dict/list): JSON data to convert
        
    Returns:
        str: Formatted string with one entry per line
    """
    if isinstance(json_data, dict):
        return '\n'.join([f"{value}" for key, value in json_data.items()])
    elif isinstance(json_data, list):
        return '\n'.join([str(item) for item in json_data])
    else:
        return str(json_data)

def json_to_dataframe(json_data, test=False):
    """
    Convert JSON data to a pandas DataFrame with 'text' and 'lead_time' columns.
    
    The input can be either:
    1. A path to a JSON file
    2. A dictionary where values are dictionaries containing 'lead_time'
    
    Args:
        json_data (Union[str, dict]): Either a path to JSON file or JSON data dictionary
        
    Returns:
        pandas.DataFrame: DataFrame with 'text' and 'lead_time' columns
    """
    # If input is a string, treat it as a file path
    if isinstance(json_data, (str, Path)):
        json_data = read_json_file(json_data)
        if json_data is None:
            raise ValueError("Invalid JSON file")
    
    rows = []
    for entry in json_data.values():
        if test:
            # Extract lead_time
            lead_time = entry.pop('lead_time', None)
        else:
            lead_time = entry['lead_time']
        
        # Convert remaining dictionary to string
        text = str(entry)
        
        rows.append({
            'text': text,
            'lead_time': lead_time
        })
        
    return pd.DataFrame(rows)

def relative_mae(y_true, y_pred):
    """
    Calculate the relative Mean Absolute Error (MAE) between the true and predicted values.

    The relative MAE is defined as the MAE divided by the mean of the true values.

    Parameters:
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.

    Returns:
    float: The relative MAE.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mean_true = np.mean(y_true)
    return mae / mean_true

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, calls: int, per_seconds: int):
        self.calls = calls
        self.per_seconds = per_seconds
        self.timestamps = []
    
    def wait_if_needed(self):
        """Wait if we've exceeded our rate limit"""
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < self.per_seconds]
        
        if len(self.timestamps) >= self.calls:
            sleep_time = self.timestamps[0] + self.per_seconds - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.timestamps = self.timestamps[1:]
        
        self.timestamps.append(now)

class PromptLoader:
    """Handles loading and managing prompts for different datasets and modes"""
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.prompts = None
        self.load_prompts()
    
    def load_prompts(self) -> None:
        """Load prompts from the JSON file for the specified dataset"""
        prompt_file = Path(__file__).parent / "prompts" / f"{self.dataset}.json"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        try:
            self.prompts = read_json_file(prompt_file)
        except Exception as e:
            raise ValueError(f"Error loading prompts for dataset {self.dataset}: {e}")
    
    def get_prompts_for_mode(self, mode: str) -> tuple[str, str]:
        """
        Get system prompt and prompt template for the specified mode
        
        Args:
            mode: The mode to get prompts for ('agg' or 'seq')
            
        Returns:
            tuple of (system_prompt, prompt_template)
            
        Raises:
            ValueError: If mode is not found in prompts file
        """
        if not self.prompts or mode not in self.prompts:
            raise ValueError(f"No prompts found for mode '{mode}' in dataset '{self.dataset}'")
            
        mode_prompts = self.prompts[mode]
        if 'system_prompt' not in mode_prompts or 'prompt_template' not in mode_prompts:
            raise ValueError(f"Invalid prompt format for mode '{mode}' in dataset '{self.dataset}'")
            
        return mode_prompts['system_prompt'], mode_prompts['prompt_template']


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Example usage of ExperimentArgs
    experiment_args = ExperimentArgs.from_args(args)
    print(experiment_args)
    # Example usage of the functions
    path = "/home/frazzetp/gui_xrecs_presc_analytics/llmp/data/pm/bac/preprocessed_log_sequential_train.json"
    # Example usage:
    data = read_json_file(path)
    formatted_text = json_to_lines(data)
    df = json_to_dataframe(data)