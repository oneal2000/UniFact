import os
import argparse
import json
from tqdm import tqdm
import torch
import gc

from src.utils import load_model_and_tokenizer
from src.detection_methods.hallucination_detection.mind_wrapper import extract_features_for_text

def main_feature_parser():
    parser = argparse.ArgumentParser(description="Extract MIND Hidden State Features")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or HF identifier of the target LLM.")
    parser.add_argument("--model_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Datatype for model loading.")
    parser.add_argument("--trust_remote_code", action='store_true', help="Trust remote code for models like Qwen.")
    parser.add_argument("--generated_data_dir", type=str, required=True, help="Directory containing data_{split}.json files from generate_mind_data.py.")
    parser.add_argument("--output_feature_dir", type=str, required=True, help="Directory to save the extracted feature files.")
    parser.add_argument("--device", type=str, default=None, help="Optional: specify device (e.g., cuda:0). Auto-detects if None.")
    return parser.parse_args()

def main():
    args = main_feature_parser()

    print("--- Feature Extraction Configuration ---")
    for arg, value in sorted(vars(args).items()): print(f"{arg}: {value}")
    print("--------------------------------------")

    os.makedirs(args.output_feature_dir, exist_ok=True)

    print("Loading model and tokenizer for feature extraction...")
    model, tokenizer, device, _ = load_model_and_tokenizer(
        args.model_name_or_path,
        torch_dtype_str=args.model_dtype,
        add_entity_marker=False,
        trust_remote_code=args.trust_remote_code
    )
    if args.device:
        device = torch.device(args.device)
        model.to(device)
        print(f"Moved model to specified device: {device}")


    for data_type in ["train", "valid", "test"]:
        input_data_file = os.path.join(args.generated_data_dir, f"data_{data_type}.json")
        if not os.path.exists(input_data_file):
            print(f"Warning: Generated data file not found: {input_data_file}. Skipping {data_type}.")
            continue

        print(f"\n--- Extracting features for {data_type} data from {input_data_file} ---")
        with open(input_data_file, encoding='utf-8') as f:
            source_data = json.load(f)

        features_last_token_avg_split = []
        features_last_layer_avg_split = []

        for item_idx, item in tqdm(enumerate(source_data), total=len(source_data), desc=f"Extracting {data_type}"):
            original_text = item.get("original_text")
            title = item.get("title", f"UnknownTitle_{item_idx}")
            hallucinated_texts = item.get("texts", [])

            ft_orig_last_token, ft_orig_last_layer = extract_features_for_text(original_text, title, model, tokenizer, device, model.config)

            if ft_orig_last_token is None or ft_orig_last_layer is None:
                print(f"Skipping item for title '{title}' due to feature extraction failure on original text.")
                continue

            hallu_features_last_token = []
            hallu_features_last_layer = []

            for hallu_text in hallucinated_texts:
                ft_hallu_last_token, ft_hallu_last_layer = extract_features_for_text(hallu_text, title, model, tokenizer, device, model.config)
                if ft_hallu_last_token and ft_hallu_last_layer:
                    hallu_features_last_token.append(ft_hallu_last_token)
                    hallu_features_last_layer.append(ft_hallu_last_layer)

            if not hallucinated_texts or hallu_features_last_token:
                features_last_token_avg_split.append({
                    "right": ft_orig_last_token,
                    "hallu": hallu_features_last_token
                })
                features_last_layer_avg_split.append({
                    "right": ft_orig_last_layer,
                    "hallu": hallu_features_last_layer
                })
            elif hallucinated_texts and not hallu_features_last_token:
                 print(f"Warning: Original text processed for '{title}', but no valid features for its {len(hallucinated_texts)} hallucinated texts.")


            if item_idx % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


        output_file_last_token = os.path.join(args.output_feature_dir, f"last_token_mean_{data_type}.json")
        output_file_last_layer = os.path.join(args.output_feature_dir, f"last_mean_{data_type}.json")

        print(f"Saving last_token_mean features ({len(features_last_token_avg_split)} entries) for {data_type} to {output_file_last_token}")
        with open(output_file_last_token, "w", encoding='utf-8') as f:
            json.dump(features_last_token_avg_split, f)

        print(f"Saving last_mean features ({len(features_last_layer_avg_split)} entries) for {data_type} to {output_file_last_layer}")
        with open(output_file_last_layer, "w", encoding='utf-8') as f:
            json.dump(features_last_layer_avg_split, f)

    print("\n--- MIND Feature extraction complete ---")

if __name__ == "__main__":
    main()