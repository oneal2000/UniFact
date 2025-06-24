import json
import os
import pandas as pd
import ast
from typing import List, Dict, Any

def _load_hotpotqa_data(dataset_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    file_name = "hotpot_dev_distractor_v1.json"
    file_path = os.path.join(dataset_dir, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"HotpotQA file '{file_name}' not found in directory {dataset_dir}")

    print(f"Loading HotpotQA data from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as fin:
        raw_dataset = json.load(fin)

    items_by_type: Dict[str, List[Dict[str, Any]]] = {}
    total_loaded_count = 0

    for idx, item in enumerate(raw_dataset):
        contexts = {title: "".join(sentences) for title, sentences in item["context"]}
        supporting_passages_raw = []
        for fact_title, _ in item["supporting_facts"]: 
            if fact_title in contexts:
                passage = contexts[fact_title]
                if passage not in supporting_passages_raw:
                    supporting_passages_raw.append(passage)

        item_type = item.get("type", "unknown").lower()
        if not item_type or item_type == "unknown":
            print(f"Warning (HotpotQA): Item {item['_id']} has an 'unknown' or missing type.")

        processed_item = {
            "qid": f"hotpotqa_{item['_id']}",
            "question_idx": idx,
            "question": item["question"],
            "golden_answer": str(item["answer"]),
            "golden_passages": supporting_passages_raw,
            "type": item_type
        }
        
        if item_type not in items_by_type:
            items_by_type[item_type] = []
        items_by_type[item_type].append(processed_item)
        total_loaded_count +=1

    print(f"Loaded {total_loaded_count} questions from HotpotQA. Types found: {list(items_by_type.keys())}")
    return items_by_type

def _load_popqa_data(dataset_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    file_name = "popQA.tsv"
    file_path = os.path.join(dataset_dir, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PopQA file '{file_name}' not found in directory {dataset_dir}")

    print(f"Loading PopQA data from {file_path}...")
        
    raw_dataset = pd.read_csv(file_path, sep="\t", on_bad_lines='warn')

    all_processed_items: List[Dict[str, Any]] = []
    
    required_cols = ['id', 'question', 'obj', 'o_aliases']
    for col in required_cols:
        if col not in raw_dataset.columns:
            raise KeyError(f"PopQA file '{file_path}' is missing required column '{col}'. Available columns: {list(raw_dataset.columns)}")

    for idx, row in raw_dataset.iterrows():
        primary_answer = str(row["obj"])
        aliases_list_str = str(row.get("o_aliases", "[]"))
        try:
            aliases = ast.literal_eval(aliases_list_str)
            if not isinstance(aliases, list): aliases = []
        except: aliases = []
        
        all_ans_options = [primary_answer] + [str(a) for a in aliases if a]
        golden_answer_str = str(all_ans_options[0]) if all_ans_options else primary_answer

        processed_item = {
            "qid": f"popqa_{row['id']}" if 'id' in row and pd.notna(row['id']) else f"popqa_{idx}",
            "question_idx": idx,
            "question": str(row["question"]),
            "golden_answer": golden_answer_str,
            "golden_passages": [],
            "type": "total",
        }
        all_processed_items.append(processed_item)

    output_map = {"total": all_processed_items}
    print(f"Loaded {len(all_processed_items)} questions from PopQA.")
    return output_map

def _load_2wikimultihopqa_data(dataset_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    dev_file_name = "dev.json"
    aliases_file_name = "id_aliases.json" 
    dev_file_path = os.path.join(dataset_dir, dev_file_name)
    aliases_file_path = os.path.join(dataset_dir, aliases_file_name)

    if not os.path.isfile(dev_file_path): raise FileNotFoundError(f"2Wiki dev file missing in {dataset_dir}")
    if not os.path.isfile(aliases_file_path): raise FileNotFoundError(f"2Wiki aliases file missing in {dataset_dir}")

    print(f"Loading 2WikiMultihopQA data from {dataset_dir}...")
    with open(dev_file_path, "r", encoding="utf-8") as fin: raw_dataset = json.load(fin)
    aliases_map = {}
    with open(aliases_file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)
            aliases_map[entry["Q_id"]] = entry["aliases"] 

    items_by_type: Dict[str, List[Dict[str, Any]]] = {}
    total_loaded_count = 0

    for idx, item in enumerate(raw_dataset):
        ans_id = item.get("answer_id")
        golden_answer = [item.get("answer", [])]
        if ans_id and ans_id in aliases_map and aliases_map[ans_id]:
            golden_answer.extend(aliases_map[ans_id])
        golden_answer_str = str(sorted(list(set(ans for ans in golden_answer if ans))))

        contexts = {name: " ".join(sents) for name, sents in item["context"]}
        supporting_passages_raw = []
        if "supporting_facts" in item:
            for fact_title, _ in item["supporting_facts"]: 
                if fact_title in contexts:
                    passage = contexts[fact_title]
                    if passage not in supporting_passages_raw: supporting_passages_raw.append(passage)
        
        item_type = item.get("type", "unknown").lower()
        processed_item = {
            "qid": f"2wikimultihopqa_{item['_id']}",
            "question_idx": idx, 
            "question": item["question"],
            "golden_answer": golden_answer_str, 
            "golden_passages": supporting_passages_raw,
            "type": item_type
        }
        if item_type not in items_by_type: items_by_type[item_type] = []
        items_by_type[item_type].append(processed_item)
        total_loaded_count +=1
        
    print(f"Loaded {total_loaded_count} questions from 2WikiMultihopQA. Types found: {list(items_by_type.keys())}")
    return items_by_type

def _load_nq_data(dataset_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    file_name = "NQ-open.efficientqa.dev.1.1.jsonl"
    file_path = os.path.join(dataset_dir, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"NQ file '{file_name}' not found in directory {dataset_dir}")

    print(f"Loading NQ data from {file_path}...")
    all_processed_items: List[Dict[str, Any]] = []
    total_loaded_count = 0
    
    with open(file_path, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if line.strip():
                item = json.loads(line)
                answers = item.get("answer", [])
                if not answers:
                    continue
                    
                golden_answer_str = str(answers)
                
                processed_item = {
                    "qid": f"nq_{idx}",
                    "question_idx": idx,
                    "question": str(item["question"]),
                    "golden_answer": golden_answer_str,
                    "golden_passages": [],
                    "type": "total"
                }
                all_processed_items.append(processed_item)
                total_loaded_count += 1

    output_map = {"total": all_processed_items}
    print(f"Loaded {total_loaded_count} questions from Natural Questions.")
    return output_map

def _load_triviaqa_data(dataset_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    file_name = "unfiltered-web-dev.json"
    file_path = os.path.join(dataset_dir, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"TriviaQA file '{file_name}' not found in directory {dataset_dir}")

    print(f"Loading TriviaQA data from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as fin:
        raw_dataset = json.load(fin)["Data"]
    
    all_processed_items: List[Dict[str, Any]] = []
    total_loaded_count = 0
    
    for idx, item in enumerate(raw_dataset):
        answers = item.get("Answer", {}).get("NormalizedAliases", [])
        if not answers:
            continue
            
        processed_item = {
            "qid": f"triviaqa_{item.get('QuestionId', idx)}",
            "question_idx": idx,
            "question": str(item["Question"]),
            "golden_answer": str(answers),
            "golden_passages": [],
            "type": "total"
        }
        all_processed_items.append(processed_item)
        total_loaded_count += 1

    output_map = {"total": all_processed_items}
    print(f"Loaded {total_loaded_count} questions from TriviaQA.")
    return output_map

def load_structured_qa_dataset(
    dataset_name: str, 
    dataset_base_dir: str,
) -> Dict[str, List[Dict[str, Any]]]:
    dataset_specific_path = os.path.join(dataset_base_dir, dataset_name)
    if not os.path.isdir(dataset_specific_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_specific_path}.")

    if dataset_name == "hotpotqa": return _load_hotpotqa_data(dataset_specific_path)
    elif dataset_name == "popqa": return _load_popqa_data(dataset_specific_path)
    elif dataset_name == "2wikimultihopqa": return _load_2wikimultihopqa_data(dataset_specific_path)
    elif dataset_name == "nq": return _load_nq_data(dataset_specific_path)
    elif dataset_name == "triviaqa": return _load_triviaqa_data(dataset_specific_path)
    else:
        raise ValueError(f"Unsupported dataset: '{dataset_name}'.")

if __name__ == '__main__':
    actual_data_base_dir = "data"
    
    if not os.path.isdir(actual_data_base_dir):
        print(f"ERROR: Test data base directory not found: {actual_data_base_dir}")
    else:
        datasets_to_test = ["hotpotqa", "popqa", "2wikimultihopqa", "nq", "triviaqa"]
        for name in datasets_to_test:
            print(f"\n--- Testing {name} ---")
            dataset_specific_dir_for_test = os.path.join(actual_data_base_dir, name)
            if not os.path.isdir(dataset_specific_dir_for_test):
                print(f"Skipping: Dir not found {dataset_specific_dir_for_test}")
                continue
            try:
                data_map = load_structured_qa_dataset(name, actual_data_base_dir)
                if data_map:
                    for type_name, data_list in data_map.items():
                        print(f"  Type: {type_name}, Count: {len(data_list)}")
                        print(f"    Sample: {data_list[0]}")
                else: print(f"No data returned for {name}.")
            except Exception as e:
                print(f"ERROR loading {name}: {e}")