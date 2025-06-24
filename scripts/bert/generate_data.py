import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
import time
import gc

from src.retrieve.retriever import bm25_retrieve
from src.data_loader import load_structured_qa_dataset
from src.answer_generator import AnswerGenerator
from src.llm_judge import LLMJudge

def sanitize_path_component(name: str) -> str:
    if name is None:
        return "none"
    return name.replace('/', '_').replace('-', '_').replace('.', '_').lower()

def parse_dataset_type_arg(dataset_type_strings: list[str]) -> dict[str, list[str]]:
    """Parses dataset_types argument like 'dataset_name:type1,type2'"""
    parsed = {}
    if not dataset_type_strings:
        return parsed
    for item in dataset_type_strings:
        if ':' not in item:
            print(f"Warning: Dataset '{item}' specified without types. Will attempt to process based on hardcoded logic or 'total'.")
            continue
        dataset_name, types_str = item.split(':', 1)
        parsed[dataset_name] = [t.strip() for t in types_str.split(',')]
    return parsed

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Training Data for Fact Verification BERT Classifier with BM25 Retrieval")
    # Core Model & Generation Args
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model for answer generation.")
    parser.add_argument("--base_seed", type=int, default=42, help="Base random seed for generation.")
    parser.add_argument("--device", type=str, default="auto", choices=['cuda', 'cpu', 'auto'], help="Device for answer generator.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max new tokens for answer generations.")
    parser.add_argument("--main_temp", type=float, default=0.7, help="Temperature for answer generation.")
    parser.add_argument("--main_top_p", type=float, default=0.9, help="Top-p for answer generation.")

    # Judge Args
    parser.add_argument("--judge_model_name_or_path", type=str, required=True, help="Model for LLM-as-a-judge.")
    parser.add_argument("--judge_device", type=str, default="auto", choices=['cuda', 'cpu', 'auto'], help="Device for judge model.")
    parser.add_argument("--judge_max_retries", type=int, default=3, help="Max retries for LLM judge if parsing fails.")
    parser.add_argument("--judge_retry_delay", type=int, default=0, help="Delay (seconds) between LLM judge retries.")
    parser.add_argument("--cache_base_dir", type=str, default="cache", help="Base directory to store judge caches.")

    # Data and Output Args
    parser.add_argument("--data_base_dir", type=str, default="data", help="Base directory where dataset-specific subdirectories are located.")
    parser.add_argument(
        "--datasets_and_types", nargs='+',
        default=[
            "2wikimultihopqa:bridge_comparison,comparison",
            "hotpotqa:comparison",
            "popqa:total",
            "triviaqa:total",
            "nq:total"
        ],
        help="List of 'dataset_name:type1,type2' strings. 'total' can be used for datasets like popqa."
    )
    parser.add_argument("--output_file", type=str, default="data/train/bert/training_data.json", help="Path to save the generated training data.")
    parser.add_argument("--start_index", type=int, default=1000, help="Start index for slicing dataset questions (0-indexed).")
    parser.add_argument("--end_index", type=int, default=1300, help="End index for slicing dataset questions (exclusive).")

    # Retrieval Args
    parser.add_argument("--retrieval_topk", type=int, default=3, help="Number of passages to retrieve using BM25.")

    return parser.parse_args()

def main():
    args = parse_arguments()
    script_start_time = time.time()
    print("--- Starting Fact Verification Training Data Generation with BM25 Retrieval ---")

    sane_target_llm_name = sanitize_path_component(args.model_name_or_path)
    sane_judge_llm_name = sanitize_path_component(args.judge_model_name_or_path)

    print(f"Target LLM for Answer Generation: {args.model_name_or_path} (Sanitized: {sane_target_llm_name})")
    print(f"Judge LLM: {args.judge_model_name_or_path} (Sanitized: {sane_judge_llm_name})")
    print(f"Data slice: Questions from index {args.start_index} to {args.end_index-1}")
    print(f"BM25 Retrieval Top K: {args.retrieval_topk}")
    print(f"Output will be saved to: {args.output_file}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    answer_generator = None
    llm_judge = None
    all_training_data = []

    try:
        print("\nInitializing Answer Generator...")
        answer_generator = AnswerGenerator(args.model_name_or_path, args.device, args.base_seed)

        datasets_to_process_map = parse_dataset_type_arg(args.datasets_and_types)
        if not datasets_to_process_map:
            datasets_to_process_map = {
                "2wikimultihopqa": ["bridge_comparison", "comparison"],
                "hotpotqa": ["comparison"],
                "popqa": ["total"],
                "triviaqa": ["total"],
                "nq": ["total"]
            }

        for dataset_name, types_to_process in datasets_to_process_map.items():
            print(f"\n--- Processing Dataset: {dataset_name} (Types: {', '.join(types_to_process)}) ---")
            
            for type_key in types_to_process:
                sane_dataset_name = sanitize_path_component(dataset_name)
                sane_dataset_type = sanitize_path_component(type_key)
                
                current_run_judge_cache_dir = os.path.join(
                    args.cache_base_dir, sane_target_llm_name, sane_dataset_name,
                    sane_dataset_type, sane_judge_llm_name
                )
                current_run_judge_cache_file = os.path.join(current_run_judge_cache_dir, "judge_cache.json")
                os.makedirs(current_run_judge_cache_dir, exist_ok=True)
                print(f"\nJudge cache for {dataset_name}-{type_key} will be saved to: {current_run_judge_cache_file}")

                if llm_judge:
                    del llm_judge
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                print(f"\nInitializing LLM Judge for {dataset_name}-{type_key}...")
                llm_judge = LLMJudge(
                    args.judge_model_name_or_path,
                    args.judge_device,
                    current_run_judge_cache_file
                )

                try:
                    loaded_data_map = load_structured_qa_dataset(
                        dataset_name=dataset_name, dataset_base_dir=args.data_base_dir
                    )
                except FileNotFoundError as e:
                    print(f"Data loading error for {dataset_name}: {e}. Skipping.")
                    continue
                except ValueError as e: # Handles unsupported dataset name
                    print(f"Data loading error for {dataset_name}: {e}. Skipping.")
                    continue

                if type_key not in loaded_data_map:
                    print(f"Warning: Type '{type_key}' not found in loaded data for '{dataset_name}'. Skipping.")
                    continue

                dataset_full = loaded_data_map[type_key]
                if not dataset_full:
                    print(f"Warning: No data for {dataset_name} - {type_key}. Skipping.")
                    continue

                if args.start_index >= len(dataset_full):
                    print(f"Start index {args.start_index} is out of bounds for {dataset_name} - {type_key} (length {len(dataset_full)}). Skipping.")
                    continue
                dataset_slice = dataset_full[args.start_index:args.end_index]

                if not dataset_slice:
                    print(f"Warning: Slice [{args.start_index}:{args.end_index}] resulted in empty data for {dataset_name} - {type_key}. Skipping.")
                    continue

                print(f"Processing {len(dataset_slice)} questions for {dataset_name} - {type_key}")

                for item_data in tqdm(dataset_slice, desc=f"Generating for {dataset_name}-{type_key}"):
                    original_qid = item_data.get('qid', f"unknown_qid_{len(all_training_data)}")
                    question_text = item_data.get('question')

                    if not question_text:
                        print(f"Skipping item {original_qid} due to missing question text.")
                        continue

                    # 1. Generate Answer
                    generated_results = answer_generator.generate(
                        item_data,
                        num_samples=0,
                        max_new_tokens=args.max_new_tokens,
                        main_do_sample=True,
                        main_temperature=args.main_temp,
                        main_top_p=args.main_top_p,
                        methods_require_samples=[],
                        methods_require_logprobs=[]
                    )
                    main_answer = generated_results.get('main_answer')

                    if not main_answer or main_answer.startswith("Error:"):
                        print(f"Skipping item {original_qid} due to answer generation error: {main_answer}")
                        continue

                    # 2. Retrieve External Passages using BM25 - Question only and Q+A
                    question_only_passages = ""
                    qa_combined_passages = ""
                    try:
                        # Question-only retrieval
                        retrieved_passages_list = bm25_retrieve(question_text, topk=args.retrieval_topk)
                        if isinstance(retrieved_passages_list, list) and all(isinstance(p, str) for p in retrieved_passages_list):
                            question_only_passages = " ".join(retrieved_passages_list).strip()
                        else:
                            print(f"Warning: question-only bm25_retrieve did not return a list of strings for QID {original_qid}. Using empty passage.")

                        # Question + Answer retrieval
                        if main_answer:
                            combined_query = f"{question_text} {main_answer}"
                            qa_passages_list = bm25_retrieve(combined_query, topk=args.retrieval_topk)
                            if isinstance(qa_passages_list, list) and all(isinstance(p, str) for p in qa_passages_list):
                                qa_combined_passages = " ".join(qa_passages_list).strip()
                            else:
                                print(f"Warning: Q+A bm25_retrieve did not return a list of strings for QID {original_qid}. Using empty passage.")
                    except Exception as retrieve_e:
                        print(f"Warning: BM25 retrieval failed for QID {original_qid}: {retrieve_e}. Using empty passages.")

                    # 3. Judge Answer (using the original golden_answer and golden_passages from item_data for the judge's context)
                    label = -1
                    try:
                        verdict = llm_judge.judge(
                            item_data,
                            main_answer,
                            max_retries=args.judge_max_retries,
                            retry_delay=args.judge_retry_delay
                        )
                        if verdict == "accurate":
                            label = 0
                        elif verdict == "inaccurate":
                            label = 1
                        else:
                            print(f"Skipping item {original_qid} due to unparsed judge verdict: {verdict}")
                            continue
                    except ValueError as judge_ve:
                        print(f"Skipping item {original_qid} due to judge failure: {judge_ve}")
                        continue
                    except Exception as judge_e: # Catch any other exception from judge
                        print(f"Skipping item {original_qid} due to unexpected judge error: {judge_e}")
                        continue

                    # 4. Store training item
                    training_item = {
                        "question": question_text,
                        "generated_answer": main_answer,
                        "question_only_passages": question_only_passages,
                        "qa_combined_passages": qa_combined_passages,
                        "label": label,
                        "original_qid": original_qid,
                        "dataset_source": f"{dataset_name}_{type_key}"
                    }
                    all_training_data.append(training_item)

                if llm_judge and hasattr(llm_judge, 'save_cache'):
                    print(f"\nSaving judge cache after processing {dataset_name} - {type_key}...")
                    llm_judge.save_cache()

        # Save all collected data
        print(f"\nSaving {len(all_training_data)} training items to {args.output_file}...")
        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            json.dump(all_training_data, f_out, indent=2)
        print("Training data generation complete.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during the main process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up models...")
        if llm_judge and hasattr(llm_judge, 'save_cache'):
            print("Saving final judge cache before cleanup...")
            llm_judge.save_cache()
            
        if answer_generator and hasattr(answer_generator, 'model'): del answer_generator.model
        if answer_generator and hasattr(answer_generator, 'tokenizer'): del answer_generator.tokenizer
        del answer_generator
        if llm_judge and hasattr(llm_judge, 'model'): del llm_judge.model
        if llm_judge and hasattr(llm_judge, 'tokenizer'): del llm_judge.tokenizer
        del llm_judge

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Cleanup finished.")

    script_end_time = time.time()
    print(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()