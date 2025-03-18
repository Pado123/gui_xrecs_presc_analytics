"""
This script preprocesses the train and test data for the lead time prediction model. The outputs is a pickle file for llmp.
"""
import argparse
import decimal
import pickle
import pandas as pd
import numpy as np
import pdpipe as pdp


def format_value(val, num_decimal=None):
	"""Convert float or int to string without scientific notation, preserving integer format.

	Args:
		val (float, int, or pd.NA): Value to be formatted.
		num_decimal (int, optional): Number of decimal places. Defaults to None.

	Returns:
		str: Formatted string representation of the value.
	"""
	if pd.isna(val):
		return "NA"
	if isinstance(val, (int, np.integer)):
		return str(val)
	if isinstance(val, (float, np.floating)):
		# Use decimal context for stable string representation
		ctx = decimal.Context()
		ctx.prec = 20
		d1 = ctx.create_decimal(repr(val))
		if num_decimal is not None:
			d1 = round(d1, num_decimal)
		# Check if it's effectively an integer
		if float(d1).is_integer():
			return str(int(float(d1)))
		return format(d1, 'f').rstrip('0').rstrip('.')
	return str(val)


def df_to_prompt(df, separator=", ", num_decimal=None):
	"""Convert DataFrame rows to strings with proper number formatting.

	Args:
		df (pd.DataFrame): Input DataFrame.
		separator (str, optional): Separator between values. Defaults to ", ".
		num_decimal (int, optional): Number of decimal places for formatting. Defaults to None.

	Returns:
		np.ndarray: Array of formatted strings, each representing a row.
	"""
	formatted_rows = []
	for _, row in df.iterrows():
		formatted_values = [format_value(val, num_decimal) for val in row]
		formatted_rows.append(separator.join(formatted_values))
	return np.array(formatted_rows)


def create_pipeline(path):
	"""Creates the data processing pipeline."""
	# Define pipeline
	if "consulta" in path:
		pipeline = pdp.PdPipeline([
				pdp.ColDrop(
						['case:concept:name', 'NEXT_ACTIVITY', 'NEXT_RESOURCE', 'weekday', 'case:variant', 'org:resource',
						 'daytime', 'start:timestamp']),
				pdp.ApplyByCols('lead_time', np.round, 'lead_time'),
				pdp.ApplyByCols('lead_time', np.int32, 'lead_time'),
				pdp.ApplyByCols(pdp.cq.StartsWith('#'), np.round, drop=False),
				pdp.ApplyByCols(pdp.cq.StartsWith('#'), np.int32, drop=False),
				])
	elif "hospital" in path:
		pipeline = pdp.PdPipeline([
				# TODO Check if these columns are meant to be dropped
				#pdp.ColDrop(
				#		['case:concept:name', 'start:timestamp', 'ETA_ACCESSO', 'COLORE_TRIAGE', 'ACCESSO_TRIAGE',
				#		 'C05B_PATOLOGIA_TRIAGE', 'C02_IDUNIVOCOASSISTITO', 'C06_DIAGNOSI_PRINCIPALE', 'C07_ESITO_DIMISSIONE',
				#		 'C103_ATTRIBUTO', 'weekday', 'case:variant',
				#		 'org:resource', 'daytime', 'start_timestamp']),
				pdp.ColDrop(['case:concept:name']),
				#pdp.ApplyByCols('lead_time', np.round, 'lead_time'),
				#pdp.ApplyByCols('lead_time', np.int32, 'lead_time'),
				#pdp.ApplyByCols(pdp.cq.StartsWith('#'), np.round, drop=False),
				#pdp.ApplyByCols(pdp.cq.StartsWith('#'), np.int32, drop=False),
				])
	else:
		print("Invalid path. Please provide a valid path.")
	return pipeline


def load_and_process_data(train_path, test_path, subsample_train=1, subsample_test=1):
	"""Loads, processes, and prepares data for model training and testing.

	Args:
		train_path (str): Path to the training CSV file.
		test_path (str): Path to the testing CSV file.
		subsample_train (float): Fraction of train data to use.
		subsample_test (float): Fraction of test data to use.

	Returns:
		dict: A dictionary containing the processed data.
	"""

	# Load data
	train = pd.read_csv(train_path)
	test = pd.read_csv(test_path)

	print(f"Train data loaded: {len(train)} rows, {len(train.columns)} columns.")

	print(f"Test data loaded: {len(test)} rows, {len(test.columns)} columns.")

	assert train.columns.to_list() == test.columns.to_list(), "Train and test columns do not match."
	print(f"Columns: {train.columns.to_list()}")

	# Subsample data
	train = train.sample(frac=subsample_train, random_state=0)
	test = test.sample(n=20, random_state=0)

	# Apply pipeline
	pipeline = create_pipeline(train_path)
	train = pipeline(train)
	test = pipeline(test)

	# Prepare data for prompt
	data = {}
	data['x_train'] = df_to_prompt(train.drop('lead_time', axis=1))
	data['y_train'] = train['lead_time'].to_numpy()
	data['x_test'] = df_to_prompt(test.drop('lead_time', axis=1))
	data['y_test'] = test['lead_time'].to_numpy()
	data['x_true'] = np.concatenate([data['x_train'], data['x_test']])
	data['y_true'] = np.concatenate([data['y_train'], data['y_test']])
	data['x_train_original'] = train.drop('lead_time', axis=1)
	data['x_test_original'] = test.drop('lead_time', axis=1)
	data['x_ordering'] = {x: i for i, x in enumerate(data['x_true'])}

	# Save the processed dataframes as csv
	train.to_csv(train_path.replace(".csv", "_processed.csv"), index=False)
	test.to_csv(test_path.replace(".csv", "_processed.csv"), index=False)

	return data


def main():
	"""Parses command line arguments and executes the data processing."""
	parser = argparse.ArgumentParser(description="Process train and test CSV files, and output a pickle file.")
	parser.add_argument("train_csv", help="Path to the training CSV file.")
	parser.add_argument("test_csv", help="Path to the testing CSV file.")
	parser.add_argument("output_pkl", help="Path to the output pickle file.")
	parser.add_argument("--subsample_train", type=float, default=1, help="Fraction of train data to use")
	parser.add_argument("--subsample_test", type=float, default=1, help="Fraction of test data to use") # TODO: Change to absolute values

	args = parser.parse_args()

	data = load_and_process_data(args.train_csv, args.test_csv, args.subsample_train, args.subsample_test)

	with open(args.output_pkl, "wb") as f:
		pickle.dump(data, f)

	print(f"Data saved to {args.output_pkl}")


if __name__ == "__main__":
	main()
