import os
import sys
import argparse
import json
import pandas as pd
import torch
from tqdm import tqdm
import gc
import glob

from src.utils import load_model_and_tokenizer

@torch.no_grad()
def get_last_token_embedding(statement: str, model, tokenizer, device, layer_index: int = -1):
    if not statement:
        return None

    try:
        inputs = tokenizer(statement, return_tensors="pt", truncation=True, padding=False)

        if inputs.input_ids.shape[1] == 0:
            return None

        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        if not hidden_states or len(hidden_states) <= abs(layer_index):
             return None

        target_layer_states = hidden_states[layer_index]
        last_token_hidden_state = target_layer_states[0, -1, :]

        return last_token_hidden_state.cpu().tolist()

    except Exception as e:
        print(f"\nError extracting embedding for statement '{statement[:50]}...': {e}")
        return None

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Embeddings for SAPLMA Probe Training from CSV files in a directory")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or HF identifier of the target LLM.")
    parser.add_argument("--model_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Datatype for model loading.")
    parser.add_argument("--trust_remote_code", action='store_true', help="Trust remote code for models like Qwen.")
    parser.add_argument("--input_dir_path", type=str, required=True, help="Path to the input directory containing CSV files (e.g., 'SAPLMA_Dataset').")
    parser.add_argument("--output_dir_path", type=str, required=True, help="Path to the directory to save the output JSON embedding files.")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index to extract embeddings from (e.g., -1 for last layer).")
    parser.add_argument("--device", type=str, default="auto", help="Device for model inference ('cuda', 'cpu', 'auto').")
    return parser.parse_args()

def main():
    args = parse_arguments()

    print("--- SAPLMA Embedding Generation Configuration ---")
    for arg, value in sorted(vars(args).items()): print(f"{arg}: {value}")
    print("-----------------------------------------------")

    try:
        os.makedirs(args.output_dir_path, exist_ok=True)
        print(f"Output directory: {args.output_dir_path}")
    except Exception as e:
        print(f"Error creating output directory {args.output_dir_path}: {e}")
        return

    print(f"Loading target LLM: {args.model_name_or_path}...")
    try:
        model, tokenizer, device, _ = load_model_and_tokenizer(
            args.model_name_or_path,
            torch_dtype_str=args.model_dtype,
            add_entity_marker=False,
            trust_remote_code=args.trust_remote_code
        )
        if args.device != "auto":
            device = torch.device(args.device)
            model.to(device)
            print(f"Moved model to specified device: {device}")
        else:
            print(f"Using auto-detected device: {device}")
    except Exception as e:
        print(f"FATAL: Failed to load model/tokenizer: {e}")
        return

    csv_files = glob.glob(os.path.join(args.input_dir_path, '*.csv'))
    if not csv_files:
        print(f"Error: No CSV files found in the input directory: {args.input_dir_path}")
        return
    print(f"Found {len(csv_files)} CSV files to process:")
    for f in csv_files: print(f"  - {os.path.basename(f)}")

    total_processed_statements = 0
    total_failed_statements = 0

    for csv_file_path in csv_files:
        results_list = []
        base_csv_filename = os.path.basename(csv_file_path)
        print(f"\n--- Processing file: {base_csv_filename} ---")

        try:
            input_df = pd.read_csv(csv_file_path)
            if 'statement' not in input_df.columns or 'label' not in input_df.columns:
                print(f"Warning: Skipping {base_csv_filename}. Missing 'statement' or 'label' column.")
                continue
            print(f"Loaded {len(input_df)} statements from {base_csv_filename}.")
        except Exception as e:
            print(f"Error loading CSV {base_csv_filename}: {e}. Skipping file.")
            continue

        file_processed_count = 0
        file_failed_count = 0
        for index, row in tqdm(input_df.iterrows(), total=len(input_df), desc=f"Statements in {base_csv_filename}"):
            statement = row['statement']
            label = row['label']

            try:
                 label = int(label)
                 if label not in [0, 1]:
                     file_failed_count += 1
                     continue
            except (ValueError, TypeError):
                 file_failed_count += 1
                 continue

            embedding = get_last_token_embedding(
                statement, model, tokenizer, device, args.layer
            )

            if embedding is not None:
                results_list.append({
                    "statement": statement,
                    "label": label,
                    "embedding": embedding,
                    "layer": args.layer,
                    "source_file": base_csv_filename,
                    "source_index": index
                })
                file_processed_count += 1
            else:
                file_failed_count += 1

            if (index + 1) % 100 == 0:
                 gc.collect()
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache()

        total_processed_statements += file_processed_count
        total_failed_statements += file_failed_count
        print(f"Finished processing {base_csv_filename}: {file_processed_count} embeddings generated, {file_failed_count} failed.")

        if results_list:
            model_name_safe = args.model_name_or_path.replace('/', '_')
            output_json_filename = f"embeddings_{base_csv_filename.replace('.csv', '')}_{model_name_safe}_layer{args.layer}.json"
            output_json_full_path = os.path.join(args.output_dir_path, output_json_filename)

            print(f"Saving {len(results_list)} embeddings to {output_json_full_path}")
            try:
                with open(output_json_full_path, 'w', encoding='utf-8') as f_out:
                    json.dump(results_list, f_out)
                print("Embeddings saved successfully.")
            except Exception as e:
                print(f"Error saving output JSON for {base_csv_filename}: {e}")
        else:
            print(f"No embeddings generated for {base_csv_filename}, skipping save.")

        del results_list
        del input_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    print("\n--- SAPLMA Embedding Generation Complete ---")
    print(f"Total statements processed across all files: {total_processed_statements}")
    print(f"Total statements failed/skipped: {total_failed_statements}")

if __name__ == "__main__":
    main()