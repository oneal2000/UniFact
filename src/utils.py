import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

ENTITY_MARKER = "[ENTITY_MARKER]"

def load_model_and_tokenizer(model_name_or_path: str,
                             torch_dtype_str: str = "auto",
                             add_entity_marker: bool = True,
                             trust_remote_code: bool = True,
                             use_device_map: bool = True):
    print(f"Loading model and tokenizer for: {model_name_or_path}...")
    primary_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch_dtype_str == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch_dtype_str == "float16":
        torch_dtype = torch.float16
    elif torch_dtype_str == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    print(f"Using torch_dtype: {torch_dtype}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code
        )
    except Exception as e:
        print(f"Error loading tokenizer from {model_name_or_path}: {e}")
        raise

    num_added_tokens = 0
    if add_entity_marker and ENTITY_MARKER not in tokenizer.vocab:
        num_added_tokens += tokenizer.add_special_tokens({'additional_special_tokens': [ENTITY_MARKER]})
        print(f"Added special token: {ENTITY_MARKER}")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Tokenizer pad_token was None, set to eos_token: {tokenizer.pad_token}")
        else:
            fallback_pad = '<|pad|>'
            num_added_tokens += tokenizer.add_special_tokens({'pad_token': fallback_pad})
            print(f"Tokenizer pad_token and eos_token were None. Added '{fallback_pad}'.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        print(f"Set tokenizer pad_token_id to: {tokenizer.pad_token_id}")

    model_kwargs = {"trust_remote_code": trust_remote_code}
    model_kwargs["torch_dtype"] = torch_dtype

    if use_device_map and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_kwargs["device_map"] = "auto"
        print("Attempting to load model with device_map='auto' across available GPUs.")
    elif use_device_map and torch.cuda.is_available() and torch.cuda.device_count() == 1:
        print("Only one GPU available, loading model to cuda:0 (device_map='auto' behaves like single GPU).")
        model_kwargs["device_map"] = "auto"
    elif use_device_map:
        print("No CUDA GPUs available, loading model to CPU (device_map not applicable).")

    try:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            output_hidden_states=True
        )
        hidden_size = getattr(config, 'hidden_size', None)
        if hidden_size is None:
            raise ValueError("Could not automatically determine hidden_size from model config.")
        print(f"Detected hidden size: {hidden_size}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            **model_kwargs
        )
        if num_added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
            print(f"Resized model token embeddings for {num_added_tokens} added special token(s).")

        if "device_map" not in model_kwargs:
            model.to(primary_device)
            print(f"Model loaded successfully to {primary_device}.")
        else:
            print(f"Model loaded with device_map. Main device (first layer): {model.device}")
            if hasattr(model, 'hf_device_map'):
                 print(f"Model layer distribution: {model.hf_device_map}")


        model.eval()

    except Exception as e:
        print(f"Error loading model from {model_name_or_path}: {e}")
        raise

    effective_device = model.device if "device_map" in model_kwargs else primary_device

    return model, tokenizer, effective_device, hidden_size
