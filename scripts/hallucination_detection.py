import os
import json
import argparse
import torch
from tqdm import tqdm
import time
import gc
import numpy as np
from src.data_loader import load_structured_qa_dataset
from src.answer_generator import AnswerGenerator
from src.detection_methods.hallucination_detection.selfcheckgpt_wrapper import SelfCheckGPTWrapper
from src.detection_methods.hallucination_detection.semantic_entropy import SemanticEntropyCalculator
from src.detection_methods.hallucination_detection.seu_calculator import SemanticEmbeddingUncertaintyCalculator
from src.detection_methods.hallucination_detection.sindex_calculator import SIndexCalculator
from src.detection_methods.hallucination_detection.mind_wrapper import MINDWrapper
from src.detection_methods.hallucination_detection.saplma_wrapper import SAPLMAWrapper
from src.detection_methods.hallucination_detection.eubhd_calculator import EUBHDCalculator

def sanitize_path_component(name: str) -> str:
    if name is None:
        return "none"
    return name.replace('/', '_').replace('-', '_').replace('.', '_').lower()

def parse_arguments():
    parser = argparse.ArgumentParser("Hallucination Detection")
    # Core Model & Generation Args
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model for answer generation.")
    parser.add_argument("--base_seed", type=int, default=42, help="Base random seed for generation.")
    parser.add_argument("--device", type=str, default="auto", choices=['cuda', 'cpu', 'auto'], help="Device for answer generator, NLI model, and embedding model.")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Maximum number of new tokens for all generations.")
    parser.add_argument("--main_temp", type=float, default=0.8, help="Temperature for main answer generation.")
    parser.add_argument("--main_top_p", type=float, default=0.9, help="Top-p for main answer generation.")
    parser.add_argument("--sample_temp", type=float, default=0.8, help="Temperature for extra sample generation.")
    parser.add_argument("--sample_top_p", type=float, default=0.9, help="Top-p for extra sample generation.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of extra samples for methods like SelfCheckGPT, PTrue, SemanticEntropy, SEU, SIndex.")

    # Data and Output Args
    parser.add_argument("--data_base_dir", type=str, default="data", help="Base directory for dataset-specific subdirectories.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=['hotpotqa', 'popqa', 'complexwebq', '2wikimultihopqa','nq','triviaqa'], help="Name of the dataset to load.")
    parser.add_argument("--dataset_type", type=str, default="total", help="Type of QA to filter (e.g. 'bridge', 'comparison', 'total').")
    parser.add_argument("--results_base_dir", type=str, default="results", help="Base directory for saving results.")
    parser.add_argument("--start_question", type=int, default=0, help="Start question index.")
    parser.add_argument("--end_question", type=int, default=500, help="End question index.")

    # Detection Method Args
    parser.add_argument(
        "--detect_methods", nargs='+',
        default=['mqag', 'bertscore', 'ngram', 'nli', 'ptrue', 'lnpp', 'lnpe', 'semantic_entropy', 'seu', 'sindex'],
        choices=[
            'mqag', 'bertscore', 'ngram', 'nli', 'ptrue',
            'lnpp', 'lnpe', 'semantic_entropy', 'seu', 'sindex', 'mind', 'saplma', 'eubhd'
        ],
        help="List of detection methods to run."
    )
    # Method Specific Args
    parser.add_argument("--sindex_embedding_model", type=str, default=None, help="SIndex embedding model. Default is 'all-MiniLM-L6-v2'.")
    parser.add_argument("--sindex_threshold", type=float, default=0.95, help="SIndex clustering cosine similarity threshold (0 to 1).")

    parser.add_argument("--split_sentences_for_detection", action='store_true', help="Split main answer into sentences for *SelfCheckGPT* detection method.")
    parser.add_argument("--aggregation_strategy", type=str, default="max", choices=['max', 'avg'], help="How to aggregate *SelfCheckGPT* method's *sentence* scores.")

    parser.add_argument("--mind_classifier_path", type=str, default=None, help="Path to pre-trained MIND MLP classifier checkpoint (.pt file).")

    parser.add_argument("--saplma_probe_path", type=str, default=None, help="Path to pre-trained SAPLMA probe checkpoint (.pt file).")
    parser.add_argument("--saplma_layer", type=int, default=-1, help="SAPLMA probe layer index.")

    parser.add_argument("--eubhd_idf_path", type=str, default=None, help="Path to pickled token IDF numpy array (.pkl file).")
    parser.add_argument("--eubhd_gamma", type=float, default=0.9, help="EUBHD discount factor (gamma).")
    parser.add_argument("--eubhd_no_penalty", action='store_true', help="Disable EUBHD penalty transfer.")
    parser.add_argument("--eubhd_all_tokens", action='store_true', help="Apply EUBHD scoring to all tokens.")
    parser.add_argument("--eubhd_no_idf", action='store_true', help="Disable IDF weighting in EUBHD.")

    return parser.parse_args()

def run_generation_and_detection_pipeline(args, dataset, answer_generator, selfcheck_wrapper=None, 
                                        se_calculator=None, seu_calculator=None, sindex_calculator=None,
                                        mind_wrapper=None, saplma_wrapper=None, eubhd_calculator=None):
    results_list = []
    processed_count = 0
    error_count = 0

    PP_PE_METRIC_KEYS = ["lnpp", "lnpe"]
    SELFCHECK_SCORE_KEYS_MAP = {
        "mqag": "mqag",
        "bertscore": "bertscore",
        "nli": "nli",
        "ngram": "ngram_unigram_max",
    }
    METHODS_REQUIRING_SAMPLES_BASE = {'mqag', 'bertscore', 'ngram', 'nli', 'ptrue', 'semantic_entropy', 'seu', 'sindex'}
    METHODS_REQUIRING_LOGPROBS_BASE = {'semantic_entropy'}
    INF_REPLACEMENT_VALUE = 1e9

    methods_requiring_samples_list = [
        m for m in args.detect_methods if m in METHODS_REQUIRING_SAMPLES_BASE
    ]
    methods_requiring_logprobs_list = [
        m for m in args.detect_methods if m in METHODS_REQUIRING_LOGPROBS_BASE
    ]
    
    for question_data in tqdm(dataset, desc="Processing questions"):
        qid = question_data.get('qid', f'index_{processed_count}')
        processed_count += 1
        item_result = {}

        try:
            # 1. Generate answers and probabilities/metrics
            generated_results = answer_generator.generate(
                question_data,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                main_do_sample=True,
                main_temperature=args.main_temp,
                main_top_p=args.main_top_p,
                sample_temperature=args.sample_temp,
                sample_top_p=args.sample_top_p,
                methods_require_samples=methods_requiring_samples_list,
                methods_require_logprobs=methods_requiring_logprobs_list
            )
            main_answer = generated_results.get('main_answer', f"Error: Generation failed QID {qid}")
            sample_answers = generated_results.get('sample_answers', [])
            pp_pe_metrics = generated_results.get('pp_pe_metrics', {})
            sequence_log_probs = generated_results.get('sequence_log_probs', [])

            item_result.update({
                "qid": qid,
                "question": question_data.get('question', ''),
                "main_answer": main_answer,
                "sample_answers": sample_answers,
            })

            # 2. Run detection methods
            detection_scores = {}
            samples_available = bool(sample_answers)
            can_run_sample_methods = main_answer and not main_answer.startswith("Error:") and samples_available
            valid_samples = []
            if can_run_sample_methods:
                valid_samples = [s for s in sample_answers if isinstance(s, str) and s.strip() and not s.startswith("Error:")]
                if not valid_samples and len(sample_answers) > 0:
                    can_run_sample_methods = False

            # PP/PE
            for key in PP_PE_METRIC_KEYS:
                if key in args.detect_methods:
                    score_value = pp_pe_metrics.get(key)
                    if score_value is not None:
                        if np.isinf(score_value): 
                            score_value = INF_REPLACEMENT_VALUE if score_value > 0 else -INF_REPLACEMENT_VALUE
                        elif np.isnan(score_value): 
                            score_value = None
                        else: 
                            score_value = float(score_value)
                    detection_scores[key] = score_value

            # Run SelfCheckGPT method
            if can_run_sample_methods and selfcheck_wrapper:
                scg_methods = [m for m in args.detect_methods if m in SELFCHECK_SCORE_KEYS_MAP]
                if scg_methods:
                    try:
                        scg_scores = selfcheck_wrapper.detect(
                            main_answer=main_answer,
                            sample_answers=valid_samples,
                            question=question_data.get('question'),
                            methods_to_run=scg_methods,
                            split_sentences=args.split_sentences_for_detection,
                            aggregation_strategy=args.aggregation_strategy
                        )
                        for method in scg_methods:
                            score_key = SELFCHECK_SCORE_KEYS_MAP.get(method)
                            if score_key in scg_scores:
                                detection_scores[method] = scg_scores[score_key]
                    except Exception as scg_e:
                        print(f"\nError during SelfCheck detection for QID {qid}: {scg_e}")

            # Run other detection methods
            if can_run_sample_methods:
                # PTrue
                if 'ptrue' in args.detect_methods:
                    try:
                        ptrue_prob = answer_generator.calculate_ptrue_probability(
                            question=question_data['question'],
                            main_answer=main_answer,
                            sample_answers=valid_samples,
                            num_samples_in_prompt=args.num_samples
                        )
                        if ptrue_prob is not None:
                            detection_scores['ptrue'] = 1.0 - ptrue_prob
                    except Exception as ptrue_e:
                        print(f"\nError during PTrue calculation for QID {qid}: {ptrue_e}")

                # Semantic Entropy
                if 'semantic_entropy' in args.detect_methods and se_calculator:
                    try:
                        se_score = se_calculator.calculate_score(
                            question=question_data['question'],
                            answers=[main_answer] + valid_samples,
                            sequence_log_probs=sequence_log_probs
                        )
                        detection_scores['semantic_entropy'] = se_score
                    except Exception as se_e:
                        print(f"\nError during Semantic Entropy calculation for QID {qid}: {se_e}")

                # SEU
                if 'seu' in args.detect_methods and seu_calculator:
                    try:
                        seu_score = seu_calculator.calculate_score(
                            main_answer=main_answer,
                            sample_answers=valid_samples
                        )
                        detection_scores['seu'] = seu_score
                    except Exception as seu_e:
                        print(f"\nError during SEU calculation for QID {qid}: {seu_e}")

                # SIndex
                if 'sindex' in args.detect_methods and sindex_calculator:
                    try:
                        sindex_score = sindex_calculator.calculate_score(
                            question=question_data['question'],
                            main_answer=main_answer,
                            sample_answers=valid_samples
                        )
                        detection_scores['sindex'] = sindex_score
                    except Exception as sindex_e:
                        print(f"\nError during SIndex calculation for QID {qid}: {sindex_e}")

            # Run model-dependent methods
            if main_answer and not main_answer.startswith("Error:"):
                # MIND
                if 'mind' in args.detect_methods and mind_wrapper:
                    try:
                        mind_score = mind_wrapper.calculate_score(
                            question=question_data['question'],
                            llm_answer=main_answer,
                            target_model=answer_generator.model,
                            target_tokenizer=answer_generator.tokenizer
                        )
                        detection_scores['mind'] = mind_score
                    except Exception as mind_e:
                        print(f"\nError during MIND calculation for QID {qid}: {mind_e}")

                # SAPLMA
                if 'saplma' in args.detect_methods and saplma_wrapper:
                    try:
                        saplma_score = saplma_wrapper.calculate_score(
                            question=question_data['question'],
                            llm_answer=main_answer,
                            target_model=answer_generator.model,
                            target_tokenizer=answer_generator.tokenizer,
                            layer_index=args.saplma_layer
                        )
                        detection_scores['saplma'] = saplma_score
                    except Exception as saplma_e:
                        print(f"\nError during SAPLMA calculation for QID {qid}: {saplma_e}")

                # EUBHD
                if 'eubhd' in args.detect_methods and eubhd_calculator:
                    try:
                        eubhd_score = eubhd_calculator.calculate_score(
                            question=question_data['question'],
                            llm_answer=main_answer,
                            target_model=answer_generator.model,
                            target_tokenizer=answer_generator.tokenizer
                        )
                        detection_scores['eubhd'] = eubhd_score
                    except Exception as eubhd_e:
                        print(f"\nError during EUBHD calculation for QID {qid}: {eubhd_e}")

            item_result["detection_scores"] = detection_scores

        except Exception as loop_e:
            print(f"\nFatal error processing QID {qid}: {loop_e}")
            import traceback; traceback.print_exc()
            error_count += 1
            item_result = {
                "qid": qid,
                "question": question_data.get('question', 'N/A'),
                "error_message": f"Loop failed - {loop_e}",
                "main_answer": "Error: Loop failed",
                "sample_answers": [],
                "detection_scores": {}
            }

        results_list.append(item_result)

    return results_list

def save_detection_results_by_method(results_list, output_dir):
    """
    Save results by detection method to separate json files
    """
    # Collect all methods
    all_methods = set()
    for result in results_list:
        if 'detection_scores' in result:
            all_methods.update(result['detection_scores'].keys())
    
    # Create separate json files for each method
    for method in all_methods:
        method_results = []
        for result in results_list:
            method_result = {
                "qid": result.get("qid"),
                "question": result.get("question"),
                "main_answer": result.get("main_answer"),
                "detection_score": result.get("detection_scores", {}).get(method)
            }
            method_results.append(method_result)
        
        method_file = os.path.join(output_dir, f"{method}.json")
        with open(method_file, 'w', encoding='utf-8') as f:
            json.dump(method_results, f, ensure_ascii=False, indent=2)
        print(f"Saved {method} results to: {method_file}")

def save_basic_results_for_fv(results_list, output_dir):
    """
    Save basic results (qid, question, main_answer) for fact verification
    """
    basic_results = []
    for result in results_list:
        basic_result = {
            "qid": result.get("qid"),
            "question": result.get("question"),
            "main_answer": result.get("main_answer")
        }
        basic_results.append(basic_result)
    
    basic_file = os.path.join(output_dir, "basic_results.json")
    with open(basic_file, 'w', encoding='utf-8') as f:
        json.dump(basic_results, f, ensure_ascii=False, indent=2)
    print(f"Saved basic results for FV to: {basic_file}")

def main():
    args = parse_arguments()
    script_start_time = time.time()

    sane_target_llm_name = sanitize_path_component(args.model_name_or_path)
    sane_dataset_name = sanitize_path_component(args.dataset_name)
    sane_dataset_type = sanitize_path_component(args.dataset_type)

    print(f"Target LLM: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Seed: {args.base_seed}")

    current_run_output_dir = os.path.join(
        args.results_base_dir, sane_target_llm_name, sane_dataset_name, sane_dataset_type
    )

    try:
        os.makedirs(args.results_base_dir, exist_ok=True)
        os.makedirs(current_run_output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error setting output directory: {e}")
        return

    # Validate method-specific paths
    if 'mind' in args.detect_methods and (not args.mind_classifier_path or not os.path.exists(args.mind_classifier_path)):
        print(f"Error: --mind_classifier_path '{args.mind_classifier_path}' is required and must exist in 'mind' method.")
        return
    if 'saplma' in args.detect_methods and (not args.saplma_probe_path or not os.path.exists(args.saplma_probe_path)):
        print(f"Error: --saplma_probe_path '{args.saplma_probe_path}' is required and must exist in 'saplma' method.")
        return
    if 'eubhd' in args.detect_methods and (not args.eubhd_no_idf and (not args.eubhd_idf_path or not os.path.exists(args.eubhd_idf_path))):
        print(f"Error: EUBHD IDF weighting is enabled, but --eubhd_idf_path '{args.eubhd_idf_path}' is required and must exist.")
        return

    print("\nLoading data...")
    try:
        loaded_data_map = load_structured_qa_dataset(
            dataset_name=args.dataset_name, dataset_base_dir=args.data_base_dir
        )

        if args.dataset_type == "total":
            if args.dataset_name in ["popqa","nq","triviaqa"] and "total" in loaded_data_map:
                dataset = loaded_data_map["total"]
            else:
                dataset = []
                for type_key in loaded_data_map:
                    dataset.extend(loaded_data_map[type_key])
                if not dataset and loaded_data_map:
                     print(f"Warning: Aggregated 'total' for {args.dataset_name} is empty. Found classified types: {list(loaded_data_map.keys())}")
        elif args.dataset_type in loaded_data_map:
            dataset = loaded_data_map[args.dataset_type]
        else:
            print(f"Error: Dataset type '{args.dataset_type}' not found in '{args.dataset_name}'. Available: {list(loaded_data_map.keys())}")
            return

        if not dataset:
            print(f"Error: No data loaded for {args.dataset_name}, type '{args.dataset_type}'.")
            return

        if args.start_question is not None and args.end_question is not None:
            dataset = dataset[args.start_question:args.end_question]
            print(f"Processing a subset of {len(dataset)} questions for {args.dataset_name} ({args.dataset_type}).")

        print(f"Successfully loaded {len(dataset)} questions for {args.dataset_name} (selected type: {args.dataset_type}).")

    except Exception as e:
        print(f"Data loading error: {e}")
        import traceback; traceback.print_exc()
        return

    print("\nInitializing models...")
    try:
        answer_generator = AnswerGenerator(args.model_name_or_path, args.device, args.base_seed)

        selfcheck_wrapper = None
        se_calculator = None
        seu_calculator = None
        sindex_calculator = None
        mind_wrapper = None
        saplma_wrapper = None
        eubhd_calculator = None

        if any(m in ['mqag', 'bertscore', 'ngram', 'nli'] for m in args.detect_methods):
            try: selfcheck_wrapper = SelfCheckGPTWrapper(device=args.device)
            except Exception as e: print(f"Warning: Failed to initialize SelfCheckGPTWrapper: {e}")

        if 'semantic_entropy' in args.detect_methods:
            try: se_calculator = SemanticEntropyCalculator(nli_model_name_or_path="microsoft/deberta-large-mnli", device=args.device)
            except Exception as e: print(f"Warning: Failed to initialize SemanticEntropyCalculator: {e}")

        if 'seu' in args.detect_methods:
            try: seu_calculator = SemanticEmbeddingUncertaintyCalculator(device=args.device)
            except Exception as e: print(f"Warning: Failed to initialize SEUCalculator: {e}")

        if 'sindex' in args.detect_methods:
            try: sindex_calculator = SIndexCalculator(model_name_or_path=args.sindex_embedding_model, device=args.device, clustering_threshold=args.sindex_threshold)
            except Exception as e: print(f"Warning: Failed to initialize SIndexCalculator: {e}")

        if 'mind' in args.detect_methods and args.mind_classifier_path:
            try: mind_wrapper = MINDWrapper(classifier_model_path=args.mind_classifier_path, device=args.device)
            except Exception as e: print(f"Warning: Failed to initialize MINDWrapper: {e}")

        if 'saplma' in args.detect_methods and args.saplma_probe_path:
            try: saplma_wrapper = SAPLMAWrapper(probe_model_path=args.saplma_probe_path, device=args.device)
            except Exception as e: print(f"Warning: Failed to initialize SAPLMAWrapper: {e}")

        if 'eubhd' in args.detect_methods:
            try: eubhd_calculator = EUBHDCalculator(
                    device=args.device, idf_path=args.eubhd_idf_path, gamma=args.eubhd_gamma,
                    only_keyword=(not args.eubhd_all_tokens), use_penalty=(not args.eubhd_no_penalty),
                    use_idf=(not args.eubhd_no_idf))
            except Exception as e: print(f"Warning: Failed to initialize EUBHDCalculator: {e}")
    except Exception as e:
        print(f"Fatal error during model initialization: {e}")
        import traceback; traceback.print_exc()
        return

    setup_end_time = time.time()
    print(f"Model and wrapper setup completed in {setup_end_time - script_start_time:.2f} seconds.")
    print(f"Methods to run: {args.detect_methods}")

    print("\nStarting pipeline execution...")
    results_list = run_generation_and_detection_pipeline(
        args,
        dataset,
        answer_generator,
        selfcheck_wrapper,
        se_calculator,
        seu_calculator,
        sindex_calculator,
        mind_wrapper,
        saplma_wrapper,
        eubhd_calculator
    )

    # Save results
    save_detection_results_by_method(results_list, current_run_output_dir)
    save_basic_results_for_fv(results_list, current_run_output_dir)

    # Clean up models
    print("\nCleaning up models...")
    del answer_generator
    if selfcheck_wrapper is not None: del selfcheck_wrapper
    if se_calculator is not None: 
        try:
            se_calculator.cleanup()
        except Exception as se_cleanup_e:
            print(f"Error during SE cleanup: {se_cleanup_e}")
        del se_calculator
    if seu_calculator is not None: del seu_calculator
    if sindex_calculator is not None: del sindex_calculator
    if mind_wrapper is not None: del mind_wrapper
    if saplma_wrapper is not None: del saplma_wrapper
    if eubhd_calculator is not None: del eubhd_calculator
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("Model cleanup completed.")

    script_end_time = time.time()
    print(f"\nHallucination detection completed. Total time: {script_end_time - script_start_time:.2f} seconds.")
    print(f"Results for this run are in: {current_run_output_dir}")

if __name__ == "__main__":
    main()