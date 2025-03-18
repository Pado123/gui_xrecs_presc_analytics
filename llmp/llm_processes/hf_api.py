from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
    StaticCache
)
import torch
import bitsandbytes as bnb

llm_map = {
    "llama-2-7B": "meta-llama/Llama-2-7b-hf",
    "llama-2-70B": "meta-llama/Llama-2-70b",
    "llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    "llama-3-70B": "meta-llama/Meta-Llama-3-70B",
    "llama-3.1-70B": "meta-llama/Meta-Llama-3.1-70B",
    "llama-3.1-70B-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "mixtral-8x7B": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral-8x7B-instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "phi-3-mini-128k-instruct": "microsoft/Phi-3-mini-128k-instruct",
    # "llama-3.1-Nemotron-70B": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", TODO Fix below about retrival
    "llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct"
}


DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_tokenizer(llm_path, llm_type):
    if llm_path is None:
        llm_path = llm_map[llm_type]
    if "llama-2" in llm_type:
        tokenizer = LlamaTokenizer.from_pretrained(
            llm_path,
            use_fast=False,
            padding_side="left"
        )
    #elif "llama-3.1-8B" in llm_type:
    #    generator = pipeline(model=llm_path, device="auto", torch_dtype=torch.bfloat16)
    elif "llama-3" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            padding_side="left",
            legacy=False
        )
    elif "phi-3" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
    elif "mixtral" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
        )       
    else:
        assert False

    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model_and_tokenizer(llm_path, llm_type, quantization="8bit"):
    """
    Load model and tokenizer with optional quantization.
    
    Args:
        llm_path: Path to model or None to use default
        llm_type: Type of LLM from llm_map
        quantization: One of ["none", "8bit", "4bit"]
    """
    if llm_path is None:
        llm_path = llm_map[llm_type]

    # Get tokenizer
    tokenizer = get_tokenizer(llm_path, llm_type)

        # Setup quantization configuration
    if quantization == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    elif quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quantization_config = None

    # Common model loading arguments
    model_args = {
        "device_map": "auto",
        "quantization_config": quantization_config if quantization != "none" else None,
        "torch_dtype": torch.float16 if quantization == "none" else "auto",
    }

    # Load the model based on type
    if "llama-2" in llm_type:
        model = LlamaForCausalLM.from_pretrained(llm_path, **model_args)
    elif "llama-3" in llm_type or "llama-3.1" in llm_type:
        model_args["torch_dtype"] = torch.bfloat16 if quantization == "none" else None
        model = AutoModelForCausalLM.from_pretrained(llm_path, **model_args)
    elif any(x in llm_type for x in ["phi-3", "mixtral"]):
        model_args.update({
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2"
        })
        model = AutoModelForCausalLM.from_pretrained(llm_path, **model_args)
    else:
        raise ValueError(f"Unknown model type: {llm_type}")

    # Post-loading optimizations
    model.eval()
    
    # Enable gradient checkpointing if available
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        
    try:
        #model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        #model.forward = torch.compile(model.forward)
        #model = torch.compile(
        #    model, 
        #    mode="reduce-overhead",
        #    fullgraph=True,
        #    dynamic=False,
        #)
        # print("Model compiled successfully")
        pass
    except Exception as e:
        print(f"Model compilation failed: {e}")

    # Print memory info
    print(f"Model loaded with {quantization} quantization")
    print(f"Memory footprint: {model.get_memory_footprint() / (1024**3):.2f} GB")

    return model, tokenizer


# This assumes that there is only a single prompt and it gets replicated batch_size times
@torch.inference_mode()
def hf_generate(
    model,
    tokenizer,
    input_str,
    batch_size,
    temp, 
    top_p,
    max_new_tokens,
    past_key_values=None
    ):
    batch = tokenizer([input_str], return_tensors="pt", pad_to_multiple_of=4).to(model.device)
    batch = {k: v.repeat(batch_size, 1).cuda() for k, v in batch.items()}
    num_input_ids = batch['input_ids'].shape[1]

    generate_ids = model.generate(
        **batch,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temp, 
        top_p=top_p, 
        renormalize_logits=False,
        pad_token_id=tokenizer.eos_token_id,
        cache_implementation="offloaded"
        # past_key_values=past_key_values
        # cache_implementation="quantized", cache_config={"nbits": 4, "backend": "quanto", "device": "cuda", "compute_dtype": model.dtype}
        # use_cache=False,
        # cache_implementation="offloaded"
    )

    gen_strs = tokenizer.batch_decode(
        generate_ids[:, num_input_ids:],
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    del generate_ids, batch
    torch.cuda.empty_cache()

    return gen_strs


# this assumes a batch of different prompts that may be different lengths
@torch.inference_mode()
def hf_generate_batch(
    model,
    tokenizer,
    prompts,
    temp,
    top_p,
    max_new_tokens
    ):
    batch = tokenizer(prompts, return_tensors="pt", padding=True)
    batch = {k: v.cuda() for k, v in batch.items()}
    num_input_ids = batch['input_ids'].shape[1]

    generate_ids = model.generate(
        **batch,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        top_p=top_p,
        renormalize_logits=False,
        pad_token_id=tokenizer.eos_token_id
    )

    gen_strs = tokenizer.batch_decode(
        generate_ids[:, num_input_ids:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return gen_strs
