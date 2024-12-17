import pandas as pd
import numpy as np
import pdpipe as pdp
import decimal
import pickle
from typing import Dict, Union, Optional
from pathlib import Path


def format_value(val: Union[int, float], num_decimal: Optional[int] = None) -> str:
	"""Convert float or int to string without scientific notation, preserving integer format.

	Args:
		val: Numeric value to format
		num_decimal: Number of decimal places to round to (if None, uses full precision)

	Returns:
		Formatted string representation of the number
	"""
	if pd.isna(val):
		return "NA"
	if isinstance(val, (int, np.integer)):
		return str(val)
	if isinstance(val, (float, np.floating)):
		ctx = decimal.Context()
		ctx.prec = 20
		d1 = ctx.create_decimal(repr(val))
		if num_decimal is not None:
			d1 = round(d1, num_decimal)
		if float(d1).is_integer():
			return str(int(float(d1)))
		return format(d1, 'f').rstrip('0').rstrip('.')
	return str(val)


def df_to_prompt(df: pd.DataFrame, separator: str = ", ", num_decimal: Optional[int] = None) -> np.ndarray:
	"""Convert DataFrame rows to strings with proper number formatting.

	Args:
		df: Input DataFrame
		separator: String to use between values
		num_decimal: Number of decimal places for floating point numbers

	Returns:
		Array of formatted strings, one per row
	"""
	formatted_rows = []
	for _, row in df.iterrows():
		formatted_values = [format_value(val, num_decimal) for val in row]
		formatted_rows.append(separator.join(formatted_values))
	return np.array(formatted_rows)


def process_pm_data(train_path: Union[str, Path],
                    test_path: Union[str, Path],
                    output_path: Union[str, Path],
                    train_frac: float = 1.0,
                    test_frac: float = 1.0,
                    random_state: int = 0) -> Dict:
	"""Process process-mining data and create a formatted dataset for LLM training.

	Args:
		train_path: Path to training CSV file
		test_path: Path to test CSV file
		output_path: Path where to save the pickle file
		train_frac: Fraction of training data to use (default: 1.0)
		test_frac: Fraction of test data to use (default: 1.0)
		random_state: Random seed for reproducibility (default: 0)

	Returns:
		Dictionary containing processed data

	The output dictionary contains:
		- x_train: Array of formatted feature strings for training
		- y_train: Array of target values for training
		- x_test: Array of formatted feature strings for testing
		- y_test: Array of target values for testing
		- x_true: Combined array of all formatted feature strings
		- y_true: Combined array of all target values
		- x_ordering: Dictionary mapping feature strings to indices
	"""
	# Define the processing pipeline
	pipeline = pdp.PdPipeline([
			pdp.ColDrop(['case:concept:name', 'NEXT_ACTIVITY', 'NEXT_RESOURCE',
			             'weekday', 'case:variant', 'org:resource', 'daytime',
			             'start:timestamp']),
			pdp.ApplyByCols('lead_time', np.round, 'lead_time'),
			pdp.ApplyByCols('lead_time', np.int32, 'lead_time'),
			pdp.ApplyByCols(pdp.cq.StartsWith('#'), np.round, drop=False),
			pdp.ApplyByCols(pdp.cq.StartsWith('#'), np.int32, drop=False),
			])

	try:
		# Load and process training data
		train = pd.read_csv(train_path)
		if train_frac < 1.0:
			train = train.sample(frac=train_frac, random_state=random_state)
		train = pipeline(train)

		# Load and process test data
		test = pd.read_csv(test_path)
		if test_frac < 1.0:
			test = test.sample(frac=test_frac, random_state=random_state)
		test = pipeline(test)

		# Create the data dictionary
		data = {
				'x_train': df_to_prompt(train.drop('lead_time', axis=1)),
				'y_train': train['lead_time'].to_numpy(),
				'x_test': df_to_prompt(test.drop('lead_time', axis=1)),
				'y_test': test['lead_time'].to_numpy()
				}

		# Add combined data
		data['x_true'] = np.concatenate([data['x_train'], data['x_test']])
		data['y_true'] = np.concatenate([data['y_train'], data['y_test']])

		# Create ordering dictionary
		data['x_ordering'] = {x: i for i, x in enumerate(data['x_true'])}

		# Save to pickle file
		output_path = Path(output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with open(output_path, "wb") as f:
			pickle.dump(data, f)

		return data

	except Exception as e:
		raise RuntimeError(f"Error processing data: {str(e)}")

if __name__ == '__main__':
	# Set fraction of data to use
	train_frac = 1.0 # Use all 2456 samples
	test_frac = 0.007 # Use 10 samples out of 1475

	# Process the data
	data = process_pm_data(
			train_path='./dfTrain.csv',
			test_path='./dfTest.csv',
			output_path=f"./consulta_{train_frac}_{test_frac}.pkl",
			train_frac=train_frac,
			test_frac=test_frac,
			random_state=0
			)

	# Print the data dictionary
	print(data)