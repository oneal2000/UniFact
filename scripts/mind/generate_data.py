import os
import argparse
import json
from tqdm import tqdm
import torch
import spacy
import re
import gc


from src.utils import load_model_and_tokenizer

def delete_substrings(lst):
    if not lst: return []
    lst = sorted(list(set(lst)), key=len, reverse=True)
    final_list = []
    for s_outer in lst:
        is_substring = False
        for s_inner in final_list:
            if s_outer != s_inner and s_outer in s_inner:
                is_substring = True
                break
        if not is_substring:
            final_list.append(s_outer)
    return final_list

def find_boundaries(text, words):
    boundaries = []
    unique_words = sorted(list(set(words)), key=len, reverse=True)
    for word in unique_words:
        if not word.strip(): continue
        try:
            for match in re.finditer(re.escape(word), text):
                start_idx, end_idx = match.span()
                is_start_boundary = (start_idx == 0) or (text[start_idx-1].isspace())
                is_end_boundary = (end_idx == len(text)) or (text[end_idx].isspace())
                if is_start_boundary and is_end_boundary:
                    boundaries.append(text[start_idx:end_idx])
        except Exception: pass
    return list(set(boundaries))

def get_entities(text, nlp_processor):
    try:
        doc = nlp_processor(text)
        entities_ = list(set([str(e).strip() for e in doc.ents if str(e).strip()]))
        if not entities_: return []
        entities_ = find_boundaries(text, entities_)
        entities = delete_substrings(entities_)
        all_entities_with_indices = []
        processed_spans = set()

        for e_text in sorted(list(set(entities)), key=len, reverse=True):
            if not e_text: continue
            try:
                for match in re.finditer(re.escape(e_text), text):
                    start_idx, end_idx = match.span()
                    current_span_covered = False
                    for ps_start, ps_end in processed_spans:
                        if max(ps_start, start_idx) < min(ps_end, end_idx):
                            current_span_covered = True
                            break
                    if current_span_covered:
                        continue

                    is_start_boundary = (start_idx == 0) or (text[start_idx-1].isspace())
                    is_end_boundary = (end_idx == len(text)) or (text[end_idx].isspace())

                    if is_start_boundary and is_end_boundary:
                        all_entities_with_indices.append((e_text, start_idx))
                        processed_spans.add((start_idx, end_idx))
            except Exception: pass
        all_entities_with_indices.sort(key=lambda x: x[1])
        return all_entities_with_indices
    except Exception as e:
        print(f"Error getting entities for text '{text[:50]}...': {e}")
        return []


def format_prompt_for_generation(tokenizer, model_config, title: str, prefix_text: str = ""):
    """
    Formats a prompt for generation, intended to elicit continuation after prefix_text.
    Returns a list of token IDs.
    """
    model_family = model_config.model_type.lower() if hasattr(model_config, 'model_type') else "unknown"
    is_chat = (hasattr(tokenizer, 'apply_chat_template') and 
               tokenizer.chat_template is not None)

    messages = []
    if is_chat:
        if "llama" in model_family or "qwen" in model_family:
            if prefix_text.strip():
                 user_content = f"Question: Tell me something about {title}.\nAnswer: {prefix_text.strip()}"
            else:
                 user_content = f"Question: Tell me something about {title}.\nAnswer:"
            messages.append({"role": "user", "content": user_content})
        else:
            prompt_str = f"USER: Question: Tell me something about {title}.\nAnswer: {prefix_text.strip()} ASSISTANT:"
            return tokenizer.encode(prompt_str, add_special_tokens=True)

        try:
            return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        except Exception as e:
            print(f"Error applying chat template for {model_family} (title: {title}): {e}. Falling back.")
            fallback_prompt = f"Question: Tell me something about {title}.\nAnswer: {prefix_text.strip()}"
            return tokenizer.encode(fallback_prompt, add_special_tokens=True)
    else:
        base_prompt = f"Question: Tell me something about {title}.\nAnswer: {prefix_text.strip()}"
        return tokenizer.encode(base_prompt, add_special_tokens=True)


def main_arg_parser():
    parser = argparse.ArgumentParser(description="Generate Pseudo-Hallucination Data for MIND Training")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or HF identifier of the target LLM.")
    parser.add_argument("--model_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Datatype for model loading.")
    parser.add_argument("--trust_remote_code", action='store_true', help="Trust remote code for models like Qwen.")
    parser.add_argument("--wiki_data_dir", type=str, default="data/train/mind/wiki", help="Directory containing wiki_train.json, wiki_valid.json, wiki_test.json.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated data_{split}.json files.")
    parser.add_argument("--topk_first_token", type=int, default=5, help="Top-k check for the first token after entity.")
    parser.add_argument("--topk_next_token", type=int, default=5, help="Top-k check for subsequent tokens.")
    parser.add_argument("--window_size", type=int, default=10, help="Max additional tokens to generate for hallucinated part / to search for recovery.")
    parser.add_argument("--min_entity_len", type=int, default=3, help="Minimum character length for an entity to be considered.")
    parser.add_argument("--max_context_sentences", type=int, default=2, help="Number of sentences from Wikipedia to use as context.")
    parser.add_argument("--device", type=str, default=None, help="Optional: specify device (e.g., cuda:0). Auto-detects if None.")
    return parser.parse_args()

def main():
    args = main_arg_parser()

    print("--- Configuration ---")
    for arg, value in sorted(vars(args).items()): print(f"{arg}: {value}")
    print("---------------------")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model and tokenizer...")
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

    print("Loading SpaCy model...")
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Downloading spacy en_core_web_sm model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load('en_core_web_sm')

    for data_type in ["train", "valid", "test"]:
        result_for_split = []
        wiki_file = os.path.join(args.wiki_data_dir, f"wiki_{data_type}.json")
        if not os.path.exists(wiki_file):
            print(f"Warning: Wiki file not found: {wiki_file}. Skipping {data_type}.")
            continue

        print(f"\n--- Processing {data_type} data from {wiki_file} ---")
        with open(wiki_file, encoding='utf-8') as f:
            wiki_data = json.load(f)

        for item_index, d in tqdm(enumerate(wiki_data), total=len(wiki_data), desc=f"Generating {data_type}"):
            original_full_text = " ".join(d.get("sentences", [])[:args.max_context_sentences]).strip()
            title = d.get("title", f"UnknownTitle_{item_index}")
            if not original_full_text: continue

            entities_found = get_entities(original_full_text, nlp)

            generated_hallucinations_for_item = []
            newly_generated_entities = []
            original_entities_replaced = []

            item_data_entry = {
                "original_text": original_full_text,
                "title": title,
                "qid": d.get("_id", d.get("id", f"gen_{data_type}_{item_index}"))
            }

            for entity_text, entity_char_start_index in entities_found:
                if not entity_text or len(entity_text) < args.min_entity_len or entity_text in title:
                    continue

                text_before_entity = original_full_text[:entity_char_start_index]
                text_after_entity = original_full_text[entity_char_start_index + len(entity_text):]

                input_ids_for_gen_list = format_prompt_for_generation(tokenizer, model.config, title, text_before_entity)
                if not input_ids_for_gen_list: continue
                input_ids_t = torch.tensor([input_ids_for_gen_list]).to(device)
                attention_mask_t = torch.ones_like(input_ids_t)

                original_entity_tokens = tokenizer.encode(entity_text, add_special_tokens=False)
                if not original_entity_tokens: continue
                first_original_entity_token_id = original_entity_tokens[0]

                tokens_after_original_entity = tokenizer.encode(text_after_entity.strip(), add_special_tokens=False)
                original_token_after_entity_id = tokens_after_original_entity[0] if tokens_after_original_entity else tokenizer.eos_token_id

                gen_config_one_token = {
                    "max_new_tokens": 1, "num_beams": 1, "do_sample": False,
                    "output_scores": True, "return_dict_in_generate": True,
                    "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
                }
                try:
                    with torch.no_grad():
                        output_first = model.generate(
                            input_ids=input_ids_t,
                            attention_mask=attention_mask_t,
                            **gen_config_one_token
                        )
                    scores_first = output_first.scores[0]
                    _, top_k_indices = torch.topk(scores_first, k=args.topk_first_token)

                    if first_original_entity_token_id in top_k_indices[0].tolist():
                        continue

                    current_sequence_ids_t = output_first.sequences
                    hallucinated_token_ids = [current_sequence_ids_t[0, -1].item()]
                    found_original_path_recovery = False

                    max_hallu_steps = len(original_entity_tokens) + args.window_size

                    for _ in range(max_hallu_steps):
                        gen_config_next = {
                            "max_new_tokens": 1, "num_beams": 1, "do_sample": False,
                            "output_scores": True, "return_dict_in_generate": True,
                            "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
                        }
                        current_attention_mask_loop = torch.ones_like(current_sequence_ids_t).to(device)
                        with torch.no_grad():
                            output_next = model.generate(
                                current_sequence_ids_t,
                                attention_mask=current_attention_mask_loop,
                                **gen_config_next
                            )
                        scores_next = output_next.scores[0]
                        _, top_k_next_indices = torch.topk(scores_next, k=args.topk_next_token)

                        if original_token_after_entity_id in top_k_next_indices[0].tolist():
                            found_original_path_recovery = True
                            break

                        next_generated_token_id = output_next.sequences[0, -1].item()
                        if next_generated_token_id == tokenizer.eos_token_id:
                            break

                        hallucinated_token_ids.append(next_generated_token_id)
                        current_sequence_ids_t = output_next.sequences
                        if len(hallucinated_token_ids) > max_hallu_steps + 5: # Safety break
                             print("Warning: Hallucination generation exceeded max steps + safety margin.")
                             break


                    if not found_original_path_recovery:
                        continue

                    hallucinated_entity_part_text = tokenizer.decode(hallucinated_token_ids, skip_special_tokens=True).strip()

                    if not hallucinated_entity_part_text or \
                       hallucinated_entity_part_text.lower() == entity_text.lower() or \
                       entity_text.lower() in hallucinated_entity_part_text.lower() or \
                       hallucinated_entity_part_text.lower() in entity_text.lower():
                        continue

                    prefix_for_reconstruction = text_before_entity.strip()
                    final_hallucinated_text = f"{prefix_for_reconstruction} {hallucinated_entity_part_text} {text_after_entity.strip()}".strip()
                    final_hallucinated_text = re.sub(r'\s+', ' ', final_hallucinated_text)


                    if final_hallucinated_text.lower() == original_full_text.lower():
                        continue

                    generated_hallucinations_for_item.append(final_hallucinated_text)
                    newly_generated_entities.append(hallucinated_entity_part_text)
                    original_entities_replaced.append((entity_text, entity_char_start_index))

                except Exception as e_inner_loop:
                    print(f"\nError in generation sub-loop for entity '{entity_text}' (QID: {item_data_entry['qid']}): {e_inner_loop}")

            if generated_hallucinations_for_item:
                item_data_entry["texts"] = generated_hallucinations_for_item
                item_data_entry["new_entities"] = newly_generated_entities
                item_data_entry["original_entities"] = original_entities_replaced
                result_for_split.append(item_data_entry)
            
            if item_index % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


        output_file = os.path.join(args.output_dir, f"data_{data_type}.json")
        print(f"Finished {data_type}. Generated {len(result_for_split)} entries with hallucinated examples.")
        print(f"Saving {data_type} data to {output_file}")
        try:
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(result_for_split, f, indent=4)
        except Exception as save_e:
            print(f"Error saving data for {data_type}: {save_e}")

    print("\n--- Pseudo-data generation complete ---")

if __name__ == "__main__":
    main()