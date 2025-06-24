import os
import json
import argparse
import torch
import time
import gc
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from src.llm_judge import LLMJudge
from src.data_loader import load_structured_qa_dataset

def sanitize_path_component(name: str) -> str:
    if name is None:
        return "none"
    return name.replace('/', '_').replace('-', '_').replace('.', '_').lower()

def parse_arguments():
    parser = argparse.ArgumentParser("Evaluation using LLM-as-a-judge")
    # Judge Args
    parser.add_argument("--judge_model_name_or_path", type=str, required=True, help="Model for LLM-as-a-judge.")
    parser.add_argument("--judge_device", type=str, default="auto", choices=['cuda', 'cpu', 'auto'], help="Device for the judge model.")
    parser.add_argument("--cache_base_dir", type=str, default="cache", help="Base directory for storing judge cache.")
    parser.add_argument("--judge_max_retries", type=int, default=5, help="Maximum number of retries for LLM judgment when parsing fails.")
    parser.add_argument("--judge_retry_delay", type=int, default=0, help="Delay between LLM judgment retries (in seconds).")

    # Input Args
    parser.add_argument("--results_base_dir", type=str, default="results", help="Results directory path.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Target model name (for path building).")
    parser.add_argument("--dataset_name", type=str, required=True, choices=['hotpotqa', 'popqa', 'complexwebq', '2wikimultihopqa','nq','triviaqa'], help="Dataset name.")
    parser.add_argument("--dataset_type", type=str, default="total", help="Dataset type.")
    
    # Data Args
    parser.add_argument("--data_base_dir", type=str, default="data", help="Base directory for ground truth data.")
    
    # Evaluation Args
    parser.add_argument("--detection_methods", nargs='+', default=None, help="Detection methods to evaluate. If not specified, will automatically discover from first stage results.")

    return parser.parse_args()

def load_detection_results(results_dir: str, methods: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
    """Load detection results from JSON files"""
    detection_results = {}
    
    # If no methods are specified, automatically discover from first stage results
    if methods is None:
        methods = []
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if file.endswith('.json') and file != 'basic_results.json' and not file.endswith('_evaluated.json') and file != 'evaluation_summary.json':
                    method_name = file[:-5]  # Remove .json suffix
                    methods.append(method_name)
            print(f"Automatically discovered detection methods: {methods}")
        else:
            print(f"Warning: Results directory does not exist: {results_dir}")
            return {}
    
    # Load results for each method
    for method in methods:
        method_file = os.path.join(results_dir, f"{method}.json")
        if os.path.exists(method_file):
            with open(method_file, 'r', encoding='utf-8') as f:
                detection_results[method] = json.load(f)
            print(f"Loaded {method} results: {len(detection_results[method])} items")
        else:
            print(f"Warning: No results found for method {method}: {method_file}")
    
    return detection_results

def load_ground_truth_data(dataset_name: str, dataset_type: str, data_base_dir: str) -> Dict[str, Dict]:
    """Load ground truth data and return qid to data mapping"""
    
    loaded_data_map = load_structured_qa_dataset(
        dataset_name=dataset_name, dataset_base_dir=data_base_dir
    )
    
    if dataset_type == "total":
        if dataset_name in ["popqa","nq","triviaqa"] and "total" in loaded_data_map:
            dataset = loaded_data_map["total"]
        else:
            dataset = []
            for type_key in loaded_data_map:
                dataset.extend(loaded_data_map[type_key])
    elif dataset_type in loaded_data_map:
        dataset = loaded_data_map[dataset_type]
    else:
        raise ValueError(f"Dataset type '{dataset_type}' not found in '{dataset_name}'.")
    
    # Build qid to question data mapping
    qid_to_data = {}
    for i, item in enumerate(dataset):
        qid = item.get('qid', f'index_{i}')
        qid_to_data[qid] = item
    
    return qid_to_data

def run_judge_evaluation(detection_results: Dict[str, List[Dict]], 
                        qid_to_data: Dict[str, Dict],
                        llm_judge: LLMJudge,
                        args) -> Dict[str, List[Dict]]:
    """Run LLM judge evaluation on detection results"""
    evaluated_results = {}
    
    for method_name, method_results in detection_results.items():
        print(f"\n--- Evaluating method: {method_name} ---")
        
        evaluated_method_results = []
        valid_scores = []
        judge_verdicts = []
        
        for item in tqdm(method_results, desc=f"Judging {method_name}"):
            qid = item.get("qid")
            main_answer = item.get("main_answer")
            detection_score = item.get("detection_score")
            
            # Get ground truth data
            ground_truth_data = qid_to_data.get(qid)
            if not ground_truth_data:
                print(f"Warning: No ground truth data found for QID {qid}")
                continue
            
            # Run LLM judgment
            judge_verdict = None
            if main_answer and isinstance(main_answer, str) and not main_answer.startswith("Error:"):
                try:
                    judge_verdict = llm_judge.judge(
                        ground_truth_data,
                        main_answer,
                        max_retries=args.judge_max_retries,
                        retry_delay=args.judge_retry_delay
                    )
                except Exception as judge_e:
                    judge_verdict = f"judging_error"
                    print(f"Error judging QID {qid}: {judge_e}")
            else:
                judge_verdict = "judgment_skipped_generation_error"
            
            # Build evaluation item
            eval_item = {
                "qid": qid,
                "question": item.get("question"),
                "main_answer": main_answer,
                "detection_score": detection_score,
                "judge_verdict": judge_verdict
            }
            evaluated_method_results.append(eval_item)
            
            # Collect valid scores and judgments for AUROC calculation
            if (detection_score is not None and 
                judge_verdict in ["accurate", "inaccurate"]):
                # Additional safety check: ensure detection_score is finite
                if isinstance(detection_score, (int, float)) and np.isfinite(detection_score):
                    valid_scores.append(detection_score)
                    # For AUROC: inaccurate (hallucination) = 1, accurate = 0
                    if judge_verdict == "inaccurate":
                        judge_verdicts.append(1)
                    else:
                        judge_verdicts.append(0)
                elif isinstance(detection_score, (int, float)):
                    print(f"Warning: Skipping non-finite detection_score for QID {qid}: {detection_score}")
                else:
                    print(f"Warning: Unexpected detection_score type for QID {qid}: {type(detection_score)}")
        
        # Calculate AUROC
        auroc_score = None
        if len(valid_scores) > 0 and len(set(judge_verdicts)) > 1:  # Need at least two different classes
            try:
                auroc_score = roc_auc_score(judge_verdicts, valid_scores)
                print(f"{method_name} AUROC: {auroc_score:.4f}")
            except Exception as auroc_e:
                print(f"Error calculating {method_name} AUROC: {auroc_e}")
        else:
            unique_labels = len(set(judge_verdicts)) if judge_verdicts else 0
            print(f"Warning: {method_name} cannot calculate AUROC - valid scores: {len(valid_scores)}, unique labels: {unique_labels}")
        
        # Add evaluation metrics to results
        evaluation_summary = {
            "method": method_name,
            "total_items": len(method_results),
            "valid_scores": len(valid_scores),
            "auroc": auroc_score,
            "positive_cases": sum(judge_verdicts) if judge_verdicts else 0,  # inaccurate/hallucinated
            "negative_cases": len(judge_verdicts) - sum(judge_verdicts) if judge_verdicts else 0  # accurate
        }
        
        evaluated_results[method_name] = {
            "results": evaluated_method_results,
            "evaluation_summary": evaluation_summary
        }
    
    return evaluated_results

def save_evaluation_results(evaluated_results: Dict[str, Dict], output_dir: str):
    """Save evaluation results to JSON files"""
    # Save detailed results for each method
    for method_name, method_data in evaluated_results.items():
        method_file = os.path.join(output_dir, f"{method_name}_evaluated.json")
        with open(method_file, 'w', encoding='utf-8') as f:
            json.dump(method_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {method_name} evaluation results to: {method_file}")
    
    # Save evaluation summary
    summary = {}
    for method_name, method_data in evaluated_results.items():
        summary[method_name] = method_data["evaluation_summary"]
    
    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation summary to: {summary_file}")
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    for method_name, summary_data in summary.items():
        auroc = summary_data.get("auroc")
        if auroc is not None:
            print(f"{method_name}: AUROC = {auroc:.4f}, valid samples = {summary_data.get('valid_scores', 0)}")
        else:
            print(f"{method_name}: AUROC = N/A, valid samples = {summary_data.get('valid_scores', 0)}")

def main():
    args = parse_arguments()
    script_start_time = time.time()
    print("--- Starting evaluation ---")

    sane_target_llm_name = sanitize_path_component(args.model_name_or_path)
    sane_judge_llm_name = sanitize_path_component(args.judge_model_name_or_path)
    sane_dataset_name = sanitize_path_component(args.dataset_name)
    sane_dataset_type = sanitize_path_component(args.dataset_type)

    print(f"Target LLM: {args.model_name_or_path}")
    print(f"Judge LLM: {args.judge_model_name_or_path}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Dataset type: {args.dataset_type}")

    # Set output and cache directories
    current_run_judge_cache_dir = os.path.join(
        args.cache_base_dir, sane_target_llm_name, sane_dataset_name,
        sane_dataset_type, sane_judge_llm_name
    )
    current_run_judge_cache_file = os.path.join(current_run_judge_cache_dir, "judge_cache.json")

    # Results directory where detection results are stored
    results_input_dir = os.path.join(
        args.results_base_dir, sane_target_llm_name, sane_dataset_name, sane_dataset_type
    )
    
    # Output directory for evaluation results (same as input for this case)
    current_run_output_dir = results_input_dir

    print(f"Detection results input directory: {results_input_dir}")
    print(f"Evaluation output directory: {current_run_output_dir}")
    print(f"Judge cache file: {current_run_judge_cache_file}")

    try:
        os.makedirs(args.cache_base_dir, exist_ok=True)
        os.makedirs(args.results_base_dir, exist_ok=True)
        os.makedirs(current_run_judge_cache_dir, exist_ok=True)
        os.makedirs(current_run_output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error setting output/cache directories: {e}")
        return

    # Load detection results from the correct directory
    print(f"\nLoading detection results from: {results_input_dir}")
    try:
        detection_results = load_detection_results(results_input_dir, args.detection_methods)
        if not detection_results:
            print("Error: No detection results found")
            return
        print(f"Successfully loaded {len(detection_results)} detection methods results")
    except Exception as e:
        print(f"Error loading detection results: {e}")
        import traceback; traceback.print_exc()
        return

    print("\nLoading ground truth data...")
    try:
        qid_to_data = load_ground_truth_data(args.dataset_name, args.dataset_type, args.data_base_dir)
        print(f"Successfully loaded {len(qid_to_data)} ground truth items")
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        import traceback; traceback.print_exc()
        return

    print("\nInitializing LLM judge...")
    try:
        llm_judge = LLMJudge(args.judge_model_name_or_path, args.judge_device, current_run_judge_cache_file)
        print("LLM judge initialization successful")
    except Exception as e:
        print(f"LLM judge initialization failed: {e}")
        import traceback; traceback.print_exc()
        return

    print("\n--- Starting evaluation execution ---")
    try:
        evaluated_results = run_judge_evaluation(
            detection_results,
            qid_to_data,
            llm_judge,
            args
        )
    except Exception as e:
        print(f"Error during evaluation execution: {e}")
        import traceback; traceback.print_exc()
        return

    # Save judge cache
    try:
        llm_judge.save_cache()
        print(f"Saved judge cache for this run to: {current_run_judge_cache_file}")
    except Exception as e:
        print(f"Error saving judge cache: {e}")

    # Save evaluation results
    save_evaluation_results(evaluated_results, current_run_output_dir)

    # Clean up models
    print("\nCleaning up models...")
    del llm_judge
    gc.collect()
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
    print("Model cleanup completed.")

    script_end_time = time.time()
    print(f"\nEvaluation completed. Total time: {script_end_time - script_start_time:.2f} seconds.")
    print(f"Results for this run are saved in: {current_run_output_dir}")

if __name__ == "__main__":
    main()