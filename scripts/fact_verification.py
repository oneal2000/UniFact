import os
import json
import argparse
import time
from tqdm import tqdm
import torch
import glob
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

from src.retrieve.retriever import bm25_retrieve
from src.detection_methods.fact_verification.llm_fv import LLMFactVerifier
from src.detection_methods.fact_verification.bert_fv import BertFactVerifier

class RetrievalType(Enum):
    QUESTION_ONLY = "question_only"
    QUESTION_ANSWER = "question_answer"

@dataclass
class FVMethod:
    name: str
    verifier_type: str  # "llm" or "bert"
    retrieval_type: RetrievalType
    model_path: str

def sanitize_path_component(name: str) -> str:
    if name is None: return "none"
    return name.replace('/', '_').replace('-', '_').replace('.', '_').lower()

def load_input_data(data_file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON or JSONL file"""
    data = []
    with open(data_file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON array format
            data = json.loads(content)
        else:
            # JSONL format
            for line in content.split('\n'):
                if line.strip(): 
                    data.append(json.loads(line))
    return data

def get_retrieved_passages(question: str, answer: str, retrieval_type: RetrievalType, topk: int) -> List[str]:
    try:
        if retrieval_type == RetrievalType.QUESTION_ONLY:
            return bm25_retrieve(question, topk=topk)
        else:  # QUESTION_ANSWER
            combined_query = f"{question} {answer}"
            return bm25_retrieve(combined_query, topk=topk)
    except Exception as e:
        print(f"Warning: Retrieval failed for Q '{question[:30]}...': {e}")
        return []

def initialize_verifiers(fv_methods: List[FVMethod], args) -> Dict[str, Any]:
    """Initialize verifiers with model caching to avoid duplicate loading"""
    verifiers = {}
    model_cache = {}  # Cache for loaded models: {(verifier_type, model_path): verifier_instance}
    
    for method in fv_methods:
        cache_key = (method.verifier_type, method.model_path)
        
        # Check if we already have a verifier for this model
        if cache_key in model_cache:
            print(f"Reusing existing {method.verifier_type.upper()} verifier for method {method.name}")
            verifiers[method.name] = model_cache[cache_key]
        else:
            # Create new verifier instance
            try:
                if method.verifier_type == 'llm':
                    verifier = LLMFactVerifier(
                        model_name=method.model_path,
                        device=args.device
                    )
                else:  # bert
                    verifier = BertFactVerifier(
                        model_dir=method.model_path,
                        device=args.device,
                        max_length=args.max_seq_length
                    )
                
                model_cache[cache_key] = verifier
                verifiers[method.name] = verifier
                print(f"Initialized new {method.verifier_type.upper()} verifier for method {method.name}")
                
            except Exception as e:
                print(f"Failed to initialize verifier for {method.name}: {e}")
    
    return verifiers

def process_item(item_data: Dict[str, Any], 
                fv_methods: List[FVMethod],
                verifiers: Dict[str, Any],
                retrieval_topk: int) -> Dict[str, Any]:
    new_item = item_data.copy()
    question = str(item_data.get('question', ''))
    answer = str(item_data.get('main_answer', item_data.get('generated_answer', '')))

    if not question or not answer:
        print(f"Skipping item with missing Q or A.")
        new_item['fv_scores'] = {method.name: None for method in fv_methods}
        return new_item

    retrieved_passages_cache = {}
    for retrieval_type in [RetrievalType.QUESTION_ONLY, RetrievalType.QUESTION_ANSWER]:
        if any(method.retrieval_type == retrieval_type for method in fv_methods):
            retrieved_passages_cache[retrieval_type] = get_retrieved_passages(
                question=question,
                answer=answer,
                retrieval_type=retrieval_type,
                topk=retrieval_topk
            )

    new_item['fv_scores'] = {}
    for method in fv_methods:
        retrieved_passages = retrieved_passages_cache.get(method.retrieval_type, [])

        verifier = verifiers.get(method.name)
        if not verifier:
            print(f"Warning: Verifier not found for method {method.name}")
            new_item['fv_scores'][method.name] = None
            continue

        try:
            score = verifier.verify(
                question=question,
                answer=answer,
                retrieved_passages=retrieved_passages
            )
            new_item['fv_scores'][method.name] = score
        except Exception as e:
            print(f"Error during verification for method {method.name}: {e}")
            new_item['fv_scores'][method.name] = None

    return new_item

def save_fv_results_by_method(results_list: List[Dict[str, Any]], output_dir: str):
    """Save fact verification results by method to separate json files"""
    # Collect all methods from fv_scores
    all_methods = set()
    for result in results_list:
        if 'fv_scores' in result:
            all_methods.update(result['fv_scores'].keys())
    
    # Create separate json files for each method
    for method in all_methods:
        method_results = []
        for result in results_list:
            method_result = {
                "qid": result.get("qid"),
                "question": result.get("question"),
                "main_answer": result.get("main_answer"),
                "detection_score": result.get("fv_scores", {}).get(method)
            }
            method_results.append(method_result)
        
        method_file = os.path.join(output_dir, f"fv_{method.replace('+', '_')}.json")
        with open(method_file, 'w', encoding='utf-8') as f:
            json.dump(method_results, f, ensure_ascii=False, indent=2)
        print(f"Saved FV {method} results to: {method_file}")

def process_basic_input(input_file: str, fv_methods: List[FVMethod], verifiers: Dict[str, Any], 
                       retrieval_topk: int, output_dir: str):
    """Process basic input file (qid, question, main_answer) and save results by method"""
    print(f"Processing basic input file: {input_file}")
    
    # Load basic input data
    input_data = load_input_data(input_file)
    if not input_data:
        print(f"No data found in {input_file}")
        return
    
    print(f"Processing {len(input_data)} items...")
    results_list = []
    
    for item in tqdm(input_data, desc="Processing items"):
        processed_item = process_item(
            item_data=item,
            fv_methods=fv_methods,
            verifiers=verifiers,
            retrieval_topk=retrieval_topk
        )
        results_list.append(processed_item)
    
    # Save results by method
    save_fv_results_by_method(results_list, output_dir)
    
    return results_list

def cleanup_verifiers(verifiers: Dict[str, Any]):
    """Clean up verifiers, ensuring each unique model instance is only cleaned up once"""
    cleaned_instances = set()
    
    for method_name, verifier in verifiers.items():
        verifier_id = id(verifier)  # Use object ID to identify unique instances
        
        if verifier_id not in cleaned_instances:
            try:
                verifier.cleanup()
                cleaned_instances.add(verifier_id)
                print(f"Cleaned up verifier instance (used by method {method_name})")
            except Exception as e:
                print(f"Error cleaning up verifier for {method_name}: {e}")

def main(args):
    start_time = time.time()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    fv_methods = []
    for method_str in args.fv_methods:
        parts = method_str.split('+')
        if len(parts) != 2 or parts[1] not in ['Q', 'QA']:
            print(f"Invalid method format: {method_str}. Should be 'LLM+Q', 'LLM+QA', 'BERT+Q', or 'BERT+QA'")
            continue
        
        verifier_type = parts[0].lower()
        retrieval_type = RetrievalType.QUESTION_ONLY if parts[1] == 'Q' else RetrievalType.QUESTION_ANSWER
        
        if verifier_type == 'llm':
            if not args.fv_llm_model_name:
                print(f"LLM model path not provided, skipping {method_str}")
                continue
            model_path = args.fv_llm_model_name
        else:  # bert
            model_dir = args.bert_fv_q_model_dir if parts[1] == 'Q' else args.bert_fv_qa_model_dir
            if not model_dir:
                print(f"BERT model directory not provided for {method_str}")
                continue
            model_path = model_dir
            
        fv_methods.append(FVMethod(
            name=method_str,
            verifier_type=verifier_type,
            retrieval_type=retrieval_type,
            model_path=model_path
        ))

    if not fv_methods:
        print("No valid FV methods configured. Exiting.")
        return

    # Initialize verifiers with caching to avoid duplicate model loading
    verifiers = initialize_verifiers(fv_methods, args)

    if not verifiers:
        print("No verifiers could be initialized. Exiting.")
        return

    # Check if input file is specified (for basic input processing)
    if args.input_file:
        output_dir = args.output_dir or os.path.dirname(args.input_file)
        process_basic_input(
            input_file=args.input_file,
            fv_methods=fv_methods,
            verifiers=verifiers,
            retrieval_topk=args.retrieval_topk,
            output_dir=output_dir
        )
    else:
        # Original behavior: process existing results files
        sane_target_llm = sanitize_path_component(args.target_llm_name)
        sane_dataset_name = sanitize_path_component(args.dataset_name)
        sane_dataset_type = sanitize_path_component(args.dataset_type)
        current_target_results_dir = os.path.join(args.results_base_dir, sane_target_llm, sane_dataset_name, sane_dataset_type)
        print(f"Target results directory: {current_target_results_dir}")

        input_file_pattern = os.path.join(current_target_results_dir, "basic_results.json")
        discovered_files = glob.glob(input_file_pattern)
        if not discovered_files:
            print(f"No 'basic_results.json' files found in {current_target_results_dir}. Exiting.")
            return
        print(f"Found {len(discovered_files)} result file(s) to process: {discovered_files}")

        all_output_items = []

        for input_filepath in discovered_files:
            print(f"\nProcessing file: {input_filepath}")
            input_items = load_input_data(input_filepath)
            if not input_items:
                print(f"No data in {input_filepath}, skipping.")
                continue

            print(f"Processing {len(input_items)} items...")
            output_items = []
            for item in tqdm(input_items, desc=f"Processing {os.path.basename(input_filepath)}"):
                processed_item = process_item(
                    item_data=item,
                    fv_methods=fv_methods,
                    verifiers=verifiers,
                    retrieval_topk=args.retrieval_topk
                )
                output_items.append(processed_item)

            all_output_items.extend(output_items)
            print(f"Processed {len(output_items)} items from {input_filepath}")

        # Save by method for existing results processing
        save_fv_results_by_method(all_output_items, current_target_results_dir)

    # Clean up verifiers properly
    cleanup_verifiers(verifiers)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nProcessing finished in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fact Verification")
    
    # Input options
    parser.add_argument("--input_file", type=str, help="Input file for basic processing (JSON/JSONL with qid, question, main_answer)")
    parser.add_argument("--output_dir", type=str, help="Output directory for results (default: same as input file directory)")
    
    # Original options for processing existing results
    parser.add_argument("--results_base_dir", type=str, default="results")
    parser.add_argument("--target_llm_name", type=str, help="Target LLM name (required if not using --input_file)")
    parser.add_argument("--dataset_name", type=str, help="Dataset name (required if not using --input_file)")
    parser.add_argument("--dataset_type", type=str, help="Dataset type (required if not using --input_file)")

    parser.add_argument("--fv_methods", nargs='+', default=['LLM+Q', 'LLM+QA', 'BERT+Q', 'BERT+QA'],
                        help="List of FV methods to use. Each method should be one of: LLM+Q, LLM+QA, BERT+Q, BERT+QA")
    
    parser.add_argument("--fv_llm_model_name", type=str, help="Model name/path for LLM-based fact verification")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries for LLM-based FV score parsing")
    
    parser.add_argument("--bert_fv_q_model_dir", type=str, help="Model directory for BERT+Q fact verification")
    parser.add_argument("--bert_fv_qa_model_dir", type=str, help="Model directory for BERT+QA fact verification")
    parser.add_argument("--max_seq_length", type=int, default=512)
    
    parser.add_argument("--retrieval_topk", type=int, default=3, help="Number of passages to retrieve")
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input_file and not all([args.target_llm_name, args.dataset_name, args.dataset_type]):
        parser.error("Either --input_file must be provided, or all of --target_llm_name, --dataset_name, --dataset_type must be provided")
    
    main(args)