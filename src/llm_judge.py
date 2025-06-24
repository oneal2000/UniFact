import torch
import os
import json
import hashlib
import re
import time
from transformers import GenerationConfig
from typing import Dict, Any, Optional, List

from src.utils import load_model_and_tokenizer

class LLMJudge:
    def __init__(self,
                 model_name_or_path: str,
                 device_preference: str = 'auto',
                 cache_path: Optional[str] = "cache/judge_cache.json",
                 trust_remote_code: bool = True):
        self.model_name_or_path = model_name_or_path
        print(f"Initializing LLMJudge with model: {model_name_or_path} using device preference: {device_preference}")

        self.model, self.tokenizer, self.device, _ = load_model_and_tokenizer(
            model_name_or_path=self.model_name_or_path,
            torch_dtype_str="auto",
            add_entity_marker=False,
            trust_remote_code=trust_remote_code
        )
        self.config = self.model.config

        self.cache_path = cache_path
        self.cache = {}
        self.cache_loaded = False
        if self.cache_path:
            try:
                cache_dir = os.path.dirname(self.cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                self.load_cache()
            except Exception as e:
                print(f"Error creating cache directory or loading cache: {e}")
                self.cache_path = None

        # --- Judge Prompt Templates ---
        self.judge_prompt_template = """You are an expert evaluator assessing the factual accuracy of a Generated Answer based *only* on the provided Golden Answer and Golden Passages for a given Question. Determine if the Generated Answer introduces significant factual errors, hallucinations, or contradictions compared to the trusted information.

[Instruction]
Evaluate the Generated Answer based *strictly* on the Golden Answer and Golden Passages. Do not use external knowledge. Identify if there are clear factual inaccuracies or information present in the Generated Answer that is not supported by or contradicts the provided golden information.

[Question]
{question}

[Golden Answer]
{golden_answer}

[Golden Passages]
{golden_passages_formatted}

[Generated Answer]
{generated_answer}

[Evaluation Task]
Choose the option that best describes the Generated Answer's factuality based *only* on the provided Golden Answer and Golden Passages:
(A) Accurate: The Generated Answer is factually consistent with the Golden Answer and Golden Passages. It does not contain significant inaccuracies or unsupported claims.
(B) Inaccurate/Hallucinated: The Generated Answer contains significant factual errors, contradictions, or information not supported by the Golden Answer and Golden Passages.

Provide your final choice by writing only the letter (A or B).
Choice:"""
        self.judge_prompt_template_no_passages = """You are an expert evaluator assessing the factual accuracy of a Generated Answer based *only* on the provided Golden Answer for a given Question. Your evaluation must rely solely on the Golden Answer. Determine if the Generated Answer introduces significant factual errors, hallucinations, or contradictions compared to the trusted Golden Answer.

[Instruction]
Evaluate the Generated Answer based *strictly* on the Golden Answer. Do not use external knowledge. Identify if there are clear factual inaccuracies or information present in the Generated Answer that is not supported by or contradicts the provided Golden Answer.

[Question]
{question}

[Golden Answer]
{golden_answer}

[Generated Answer]
{generated_answer}

[Evaluation Task]
Choose the option that best describes the Generated Answer's factuality based *only* on the provided Golden Answer:
(A) Accurate: The Generated Answer is factually consistent with the Golden Answer. It does not contain significant inaccuracies or unsupported claims.
(B) Inaccurate/Hallucinated: The Generated Answer contains significant factual errors, contradictions, or information not supported by the Golden Answer.

Provide your final choice by writing only the letter (A or B).
Choice:"""

        self.supports_chat_template = hasattr(self.tokenizer, 'apply_chat_template') and \
                                      self.tokenizer.chat_template is not None
        if self.supports_chat_template:
            print(f"LLMJudge: Model {model_name_or_path} likely supports chat templates.")
        else:
            print(f"LLMJudge: Model {model_name_or_path} does not seem to support chat templates. Using basic prompting.")

        print(f"LLMJudge initialized. Model is on device(s) (first layer: {self.model.device}). Effective operating device for inputs: {self.device}")


    def load_cache(self):
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                self.cache_loaded = True
                print(f"Loaded {len(self.cache)} entries from judge cache: {self.cache_path}")
            except json.JSONDecodeError:
                print(f"Warning: Cache file {self.cache_path} is corrupted. Starting with an empty cache.")
                self.cache = {}
            except Exception as e:
                 print(f"Warning: Could not load cache file {self.cache_path}. Error: {e}. Starting empty.")
                 self.cache = {}
        else:
             print("No judge cache file found or caching disabled. Starting with an empty cache.")
             self.cache = {}


    def save_cache(self):
        if self.cache_path:
            try:
                with open(self.cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, indent=2)
            except Exception as e:
                print(f"Error saving judge cache to {self.cache_path}: {e}")

    def _get_cache_key(self, question: str, golden_answer: str, generated_answer: str, passages_present: bool) -> str:
        gen_answer_key_part = generated_answer[:1000]
        key_string = (f"judge_model={self.model_name_or_path}::q={question}::ga={golden_answer}"
                      f"::gen_a={gen_answer_key_part}::passages_present={passages_present}")
        return hashlib.md5(key_string.encode()).hexdigest()

    def _build_judge_prompt(self, question: str, golden_answer: str, golden_passages: List[str], generated_answer: str) -> str:
        golden_answer_fmt = golden_answer if golden_answer else "N/A"
        if golden_passages:
            passages_formatted = "\n---\n".join(golden_passages)
            return self.judge_prompt_template.format(
                question=question,
                golden_answer=golden_answer_fmt,
                golden_passages_formatted=passages_formatted,
                generated_answer=generated_answer
            )
        else:
            return self.judge_prompt_template_no_passages.format(
                question=question,
                golden_answer=golden_answer_fmt,
                generated_answer=generated_answer
            )

    @torch.no_grad()
    def _generate_judge_response(self, prompt: str, attempt: int = 0) -> str:
        max_new_tokens_judge = 10
        gen_config_dict = {
            "max_new_tokens": max_new_tokens_judge,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if attempt > 0:
             gen_config_dict["do_sample"] = True
             gen_config_dict["temperature"] = 0.2
             gen_config_dict["top_p"] = 0.9
             gen_config_dict.pop("num_beams", None)

        target_device_for_inputs = self.model.device

        if self.supports_chat_template:
            messages = [{'role': 'user', 'content': prompt}]
            try:
                input_ids = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(target_device_for_inputs)
                attention_mask = torch.ones_like(input_ids).to(target_device_for_inputs)
            except Exception as e:
                print(f"Error applying chat template for judge: {e}. Using basic tokenization.")
                encoding = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest")
                input_ids = encoding.input_ids.to(target_device_for_inputs)
                attention_mask = encoding.attention_mask.to(target_device_for_inputs)
        else:
            encoding = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest")
            input_ids = encoding.input_ids.to(target_device_for_inputs)
            attention_mask = encoding.attention_mask.to(target_device_for_inputs)

        max_model_len = getattr(self.tokenizer, 'model_max_length', getattr(self.config, 'max_position_embeddings', 512))
        if input_ids.shape[1] >= max_model_len:
             print(f"Warning: Judge prompt length ({input_ids.shape[1]}) exceeds or equals model max length ({max_model_len}). Truncating.")
             safe_len = max_model_len - max_new_tokens_judge - 5
             if safe_len <= 0: safe_len = max_model_len // 2
             input_ids = input_ids[:, :safe_len]
             attention_mask = attention_mask[:, :safe_len]

        input_ids_len = input_ids.shape[1]
        try:
            safe_gen_config_dict = gen_config_dict.copy()
            if not safe_gen_config_dict.get("do_sample", False):
                 safe_gen_config_dict.pop("temperature", None)
                 safe_gen_config_dict.pop("top_p", None)
            gen_config = GenerationConfig(**safe_gen_config_dict)
        except TypeError as e:
             print(f"Warning: Could not create GenerationConfig from dict: {e}. Passing dict directly.")
             gen_config = gen_config_dict
        if isinstance(gen_config, GenerationConfig):
            if gen_config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
                gen_config.pad_token_id = self.tokenizer.pad_token_id
            if gen_config.eos_token_id is None and self.tokenizer.eos_token_id is not None:
                gen_config.eos_token_id = self.tokenizer.eos_token_id
        else:
             if gen_config.get("pad_token_id") is None and self.tokenizer.pad_token_id is not None:
                  gen_config["pad_token_id"] = self.tokenizer.pad_token_id
             if gen_config.get("eos_token_id") is None and self.tokenizer.eos_token_id is not None:
                  gen_config["eos_token_id"] = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config
        )
        output_tokens = outputs[0, input_ids_len:]
        response_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        return response_text

    def _parse_judge_output(self, response_text: str) -> Optional[str]:
        response_text_upper = response_text.strip().upper()
        choice_match = re.search(r"CHOICE:\s*\(?([AB])\)?", response_text_upper)
        if choice_match:
             choice = choice_match.group(1)
             return "accurate" if choice == 'A' else "inaccurate"
        match = re.search(r"^\(?([AB])\)?", response_text_upper)
        if match:
            choice = match.group(1)
            return "accurate" if choice == 'A' else "inaccurate"
        if 'A' in response_text_upper[:5] and 'B' not in response_text_upper[:5]: return "accurate"
        if 'B' in response_text_upper[:5] and 'A' not in response_text_upper[:5]: return "inaccurate"
        print(f"Warning: Could not parse judge output: '{response_text}'. Returning None.")
        return None

    def judge(self,
              question_data: Dict[str, Any],
              generated_answer: str,
              max_retries: int = 5,
              retry_delay: int = 0
              ) -> str:
        if not generated_answer or generated_answer.strip() == "" or generated_answer.startswith("Error:"):
             return "judgment_skipped_error_in_answer"

        question = question_data.get('question', 'N/A')
        golden_answer = question_data.get('golden_answer', '')
        golden_passages = question_data.get('golden_passages', [])
        qid = question_data.get('qid', 'N/A')
        gen_answer_for_key = generated_answer[:1000]
        passages_are_present = bool(golden_passages)
        cache_key = self._get_cache_key(question, golden_answer, gen_answer_for_key, passages_are_present)

        if self.cache_path and cache_key in self.cache:
            cached_entry = self.cache[cache_key]
            if isinstance(cached_entry, dict) and cached_entry.get("verdict") in ["accurate", "inaccurate"]:
                 return cached_entry["verdict"]
            else:
                 print(f"Warning: Invalid or old format verdict found in cache for QID {qid}, key {cache_key}. Regenerating.")

        full_prompt = self._build_judge_prompt(
            question=question,
            golden_answer=golden_answer,
            golden_passages=golden_passages,
            generated_answer=generated_answer
        )
        verdict = None
        last_error = None
        judge_raw_response = ""
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                     print(f"Retrying judgment... Attempt {attempt + 1}/{max_retries + 1} for QID {qid}")
                judge_response_text = self._generate_judge_response(full_prompt, attempt=attempt)
                judge_raw_response = judge_response_text
                verdict = self._parse_judge_output(judge_response_text)
                if verdict in ["accurate", "inaccurate"]:
                    break
                last_error = f"Parsing failed: Response='{judge_response_text}'"
                print(f"Attempt {attempt + 1} failed for QID {qid}: {last_error}")
            except Exception as e:
                last_error = e
                print(f"Error during judge generation/parsing on attempt {attempt + 1} for QID {qid}: {e}")
            if verdict not in ["accurate", "inaccurate"] and attempt < max_retries:
                print(f"Waiting {retry_delay} seconds before retry for QID {qid}...")
                time.sleep(retry_delay)

        if verdict in ["accurate", "inaccurate"]:
             if self.cache_path:
                 cache_entry = {
                     "verdict": verdict,
                     "question": question,
                     "golden_answer": golden_answer,
                     "golden_passages": golden_passages,
                     "judged_answer": generated_answer,
                     "judge_raw_response": judge_raw_response,
                 }
                 self.cache[cache_key] = cache_entry
             return verdict
        else:
            error_message = f"Failed to get valid judgment for QID {qid} after {max_retries + 1} attempts."
            if last_error: error_message += f" Last error: {last_error}"
            print(error_message)
            raise ValueError(error_message)