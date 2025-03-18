from jsonargparse import ArgumentParser, ActionConfigFile
from llm_processes.run_llm_process import run_llm_process
from llm_processes.hf_api import get_model_and_tokenizer, llm_map
from llm_processes.parse_args import init_option_parser


def main():


	# Get base option parser
	parser = init_option_parser()

	# Default config files
	default_configs = ['--cfg', './experiments/config/pm_exp.yaml']

	# Parse arguments with default configs first
	args = parser.parse_args(default_configs)

	# If additional config file is provided via command line, it will override previous settings
	# Other command line arguments will also override config files
	args = parser.parse_args(namespace=args)

	# Format any template strings in config
	if isinstance(args.experiment_name, str):
		args.experiment_name = args.experiment_name.format(
				llm_type=args.llm_type,
				autoregressive=args.autoregressive
				)

	print(args, flush=True)

	model, tokenizer = get_model_and_tokenizer(args.llm_path, args.llm_type, args.quantization)
	print("Model: ", model)
	print("Tokenizer: ", tokenizer)
	print("Model Config: ", model.config)
	print("Max Token Length: ", model.config.max_position_embeddings)
	print("Model Tokenizer: ", tokenizer)
	
	# Check if the tokenizer has a chat template
	if hasattr(tokenizer, 'chat_template'):
		print("Tokenizer has a chat template")
		print("Chat Template: ", tokenizer.chat_template)

	run_llm_process(args=args, model=model, tokenizer=tokenizer)


if __name__ == '__main__':
	main()
