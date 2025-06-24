import torch
import random
import numpy as np
import hashlib
from transformers import GenerationConfig
from torch.distributions import Categorical
from typing import Dict, List, Any, Optional, Tuple

from src.utils import load_model_and_tokenizer

class AnswerGenerator:
    def __init__(self,
                 model_name_or_path: str,
                 device_preference: str = 'auto',
                 base_seed: int = 42,
                 trust_remote_code: bool = True):
        self.model_name_or_path = model_name_or_path
        self.base_seed = base_seed
        self.assistant_prompt_continuation: str = ""
        print(f"Initializing AnswerGenerator for model: {model_name_or_path} with device preference: {device_preference}")
        print(f"Using hardcoded assistant prompt continuation: '{self.assistant_prompt_continuation}'")

        self.model, self.tokenizer, self.device, self.hidden_size = load_model_and_tokenizer(
            model_name_or_path=self.model_name_or_path,
            torch_dtype_str="auto",
            add_entity_marker=False,
            trust_remote_code=trust_remote_code,
        )
        self.config = self.model.config

        self.supports_chat_template = hasattr(self.tokenizer, 'apply_chat_template') and \
                                        self.tokenizer.chat_template is not None
        if self.supports_chat_template:
            print("Model likely supports chat templates. Will use apply_chat_template.")
        else:
            print("Model does not seem to support chat templates or template not defined. Using basic QA prompt.")

        print("AnswerGenerator initialized.")

    def _get_question_seed(self, question_identifier: Any) -> int:
        hash_input = f"{self.base_seed}_{str(question_identifier)}"
        hash_object = hashlib.sha256(hash_input.encode())
        seed = int.from_bytes(hash_object.digest()[:4], byteorder='big')
        return seed

    def _build_qa_prompt(self, question: str) -> str:
        return f"Question: {question}\nAnswer: {self.assistant_prompt_continuation}"

    def _get_ptrue_prompt(self, question: str, possible_answer: str, sample_answers: List[str], num_samples_to_use: int = 5) -> str:
        brainstormed_ideas_list = sample_answers[:num_samples_to_use]
        while len(brainstormed_ideas_list) < num_samples_to_use:
            brainstormed_ideas_list.append("[NO SAMPLE]")

        brainstormed_ideas_formatted = '\n'.join(brainstormed_ideas_list)

        prompt_template = """Question: {question}
Here are some brainstormed ideas:
{ideas}
Possible Answer: {possible_answer}
Is the possible answer:
(A) True
(B) False
The possible answer is:"""

        prompt = prompt_template.format(
            question=question,
            ideas=brainstormed_ideas_formatted,
            possible_answer=possible_answer
        )
        return prompt

    @torch.no_grad()
    def _generate_text(self,
                        prompt: str,
                        generation_config_dict: Dict[str, Any],
                        is_chat_model: bool = False,
                        calculate_scores_and_ids: bool = False
                        ) -> Dict[str, Any]:
        results = {"generated_text": "", "token_scores": None, "token_ids": None, "error": None}
        try:
            if is_chat_model and self.supports_chat_template:
                messages = [{'role': 'user', 'content': prompt}]
                try:
                    input_ids_templated_cpu = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )

                    if self.assistant_prompt_continuation.strip():
                        continuation_ids_cpu = self.tokenizer.encode(
                            self.assistant_prompt_continuation,
                            add_special_tokens=False,
                            return_tensors="pt"
                        )
                        if continuation_ids_cpu.dim() == 1:
                            continuation_ids_cpu = continuation_ids_cpu.unsqueeze(0)
                        input_ids = torch.cat([input_ids_templated_cpu, continuation_ids_cpu], dim=1).to(self.device)
                    else:
                        input_ids = input_ids_templated_cpu.to(self.device)
                    attention_mask = torch.ones_like(input_ids).to(self.device)

                except Exception as e:
                    print(f"Warning: Error applying chat template or hardcoded assistant continuation: {e}. Falling back to basic tokenization.")
                    effective_prompt_for_fallback = f"Question: {prompt}\nAnswer: {self.assistant_prompt_continuation}"
                    encoding = self.tokenizer(effective_prompt_for_fallback, return_tensors="pt", truncation=True, padding="longest")
                    input_ids = encoding.input_ids.to(self.device)
                    attention_mask = encoding.attention_mask.to(self.device)
            else:
                encoding = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="longest")
                input_ids = encoding.input_ids.to(self.device)
                attention_mask = encoding.attention_mask.to(self.device)

            max_model_len = getattr(self.tokenizer, 'model_max_length', getattr(self.config, 'max_position_embeddings', 512))
            max_new_tokens = generation_config_dict.get("max_new_tokens", 100)
            if input_ids.shape[1] >= max_model_len:
                print(f"Warning: Input prompt length ({input_ids.shape[1]}) exceeds or equals model max length ({max_model_len}). Truncating.")
                safe_len = max_model_len - max_new_tokens - 5
                if safe_len <= 0:
                    safe_len = max(1, max_model_len // 2)
                    print(f"Warning: max_new_tokens too large for model length. Truncating input significantly to {safe_len}.")
                input_ids = input_ids[:, :safe_len]
                attention_mask = attention_mask[:, :safe_len]

            input_ids_len = input_ids.shape[1]

            gen_config_dict_copy = generation_config_dict.copy()
            if calculate_scores_and_ids:
                gen_config_dict_copy["output_scores"] = True
                gen_config_dict_copy["return_dict_in_generate"] = True

            if "pad_token_id" not in gen_config_dict_copy and self.tokenizer.pad_token_id is not None:
                gen_config_dict_copy["pad_token_id"] = self.tokenizer.pad_token_id
            if "eos_token_id" not in gen_config_dict_copy and self.tokenizer.eos_token_id is not None:
                gen_config_dict_copy["eos_token_id"] = self.tokenizer.eos_token_id

            try:
                gen_config = GenerationConfig(**gen_config_dict_copy)
            except TypeError as e:
                print(f"Warning: Could not create GenerationConfig from dict: {e}. Passing dict directly.")
                gen_config = gen_config_dict_copy

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config
            )

            output_tokens_only = None
            if calculate_scores_and_ids and hasattr(outputs, 'sequences') and hasattr(outputs, 'scores'):
                output_tokens_only = outputs.sequences[0, input_ids_len:]
                results["token_scores"] = outputs.scores
                results["token_ids"] = output_tokens_only
            elif isinstance(outputs, torch.Tensor):
                output_tokens_only = outputs[0, input_ids_len:]
                if calculate_scores_and_ids:
                    results["error"] = "Scores not returned by model.generate (output was Tensor)"
            elif hasattr(outputs, 'sequences') and not hasattr(outputs, 'scores'):
                output_tokens_only = outputs.sequences[0, input_ids_len:]
                results["token_ids"] = output_tokens_only
                if calculate_scores_and_ids:
                    results["error"] = "Scores missing in generate output dict"
            else:
                error_msg = f"Unexpected output type from model.generate: {type(outputs)}"
                results["error"] = error_msg
                results["generated_text"] = f"Error: {error_msg}"
                return results

            if output_tokens_only is not None:
                generated_text = self.tokenizer.decode(output_tokens_only, skip_special_tokens=True).strip()
                results["generated_text"] = generated_text
            elif not results["error"]:
                results["error"] = "Failed to extract output tokens"
                results["generated_text"] = "Error: Failed to extract output tokens"

        except Exception as e:
            print(f"Error during _generate_text for prompt '{prompt[:100]}...': {e}")
            import traceback; traceback.print_exc()
            results["error"] = str(e)
            results["generated_text"] = f"Error: {e}"
        return results

    def _calculate_pp_pe_metrics_and_logprob(self,
                                token_scores: Optional[Tuple[torch.Tensor]],
                                token_ids: Optional[torch.Tensor]
                                ) -> Dict[str, Optional[float]]:
        metrics = {
            "lnpp": None,
            "lnpe": None,
            "sequence_log_prob": None
        }
        if token_scores is None or token_ids is None or len(token_ids) == 0:
            return metrics

        token_nlls_list = []
        token_entropies_list = []
        token_log_probs_list = []
        token_ids_cpu = token_ids.cpu()

        try:
            for i, step_logits in enumerate(token_scores):
                step_logits_squeeze = step_logits.squeeze(0)
                log_probs = torch.log_softmax(step_logits_squeeze, dim=-1)
                entropy = Categorical(logits=step_logits_squeeze).entropy().item()
                if not np.isnan(entropy) and not np.isinf(entropy):
                    token_entropies_list.append(entropy)

                generated_token_id = token_ids_cpu[i].item()
                if 0 <= generated_token_id < log_probs.shape[-1]:
                    token_log_prob = log_probs[generated_token_id].item()

                    if not np.isnan(token_log_prob) and not np.isinf(token_log_prob):
                         token_log_probs_list.append(token_log_prob)
                    token_nll = -token_log_prob
                    if not np.isnan(token_nll) and not np.isinf(token_nll):
                            token_nlls_list.append(token_nll)
            if token_nlls_list:
                metrics["lnpp"] = float(np.mean(token_nlls_list))
            if token_entropies_list:
                metrics["lnpe"] = float(np.mean(token_entropies_list))
            if token_log_probs_list:
                 metrics["sequence_log_prob"] = float(sum(token_log_probs_list))
        except Exception as e:
            print(f"Error during PP/PE/LogProb metric calculation: {e}")
        return metrics

    @torch.no_grad()
    def generate(self,
                 question_data: Dict[str, Any],
                 num_samples: int = 5,
                 max_new_tokens: int = 100,
                 main_do_sample: bool = True,
                 main_temperature: float = 0.7,
                 main_top_p: float = 0.9,
                 sample_temperature: float = 0.7,
                 sample_top_p: float = 0.9,
                 methods_require_samples: Optional[List[str]] = None,
                 methods_require_logprobs: Optional[List[str]] = None
                 ) -> Dict[str, Any]:
        question = question_data.get('question', 'N/A')
        question_id = question_data.get('qid', 'N/A')

        if self.supports_chat_template:
            prompt_content_for_generation = question
        else:
            prompt_content_for_generation = self._build_qa_prompt(question)

        results = {
            'main_answer': None, 'sample_answers': [],
            'pp_pe_metrics': {}, 'sequence_log_probs': []
        }
        should_generate_samples = bool(methods_require_samples)
        should_calculate_logprobs = bool(methods_require_logprobs)
        question_seed = self._get_question_seed(question_id)
        torch.manual_seed(question_seed)
        random.seed(question_seed)
        np.random.seed(question_seed)

        main_gen_config_dict = {
            "max_new_tokens": max_new_tokens, "do_sample": main_do_sample,
            "temperature": main_temperature if main_do_sample else 1.0,
            "top_p": main_top_p if main_do_sample else 1.0, "num_beams": 1,
        }
        main_answer = f"Error: Generation failed for QID {question_id}"
        main_log_prob = None
        pp_pe_metrics_main = {k: None for k in ["lnpp", "lnpe", "sequence_log_prob"]}

        try:
            main_gen_output = self._generate_text(
                prompt=prompt_content_for_generation,
                generation_config_dict=main_gen_config_dict,
                is_chat_model=self.supports_chat_template,
                calculate_scores_and_ids=True
            )
            main_answer = main_gen_output["generated_text"]
            if main_gen_output["error"] is None and main_answer and not main_answer.startswith("Error:"):
                pp_pe_metrics_main = self._calculate_pp_pe_metrics_and_logprob(
                    token_scores=main_gen_output["token_scores"], token_ids=main_gen_output["token_ids"]
                )
                main_log_prob = pp_pe_metrics_main.get("sequence_log_prob")
        except Exception as e:
            print(f"Error generating main answer or metrics for qid {question_id}: {e}")
            main_answer = f"Error: {e}"

        results['main_answer'] = main_answer
        results['pp_pe_metrics'] = {k: v for k, v in pp_pe_metrics_main.items() if k != "sequence_log_prob"}
        results['sequence_log_probs'].append(main_log_prob)

        if should_generate_samples:
            sample_gen_config_dict = {
                "max_new_tokens": max_new_tokens, "do_sample": True,
                "temperature": sample_temperature, "top_p": sample_top_p, "num_beams": 1,
            }
            for i in range(num_samples):
                sample_answer = f"Error: Sample gen failed (QID {question_id}, sample {i+1})"
                sample_log_prob = None
                try:
                    calculate_scores_for_sample = should_calculate_logprobs
                    sample_gen_output = self._generate_text(
                        prompt=prompt_content_for_generation,
                        generation_config_dict=sample_gen_config_dict,
                        is_chat_model=self.supports_chat_template,
                        calculate_scores_and_ids=calculate_scores_for_sample
                    )
                    sample_answer = sample_gen_output["generated_text"]

                    if calculate_scores_for_sample and sample_gen_output["error"] is None and \
                       sample_answer and not sample_answer.startswith("Error:"):
                        temp_metrics = self._calculate_pp_pe_metrics_and_logprob(
                             token_scores=sample_gen_output["token_scores"], token_ids=sample_gen_output["token_ids"]
                        )
                        sample_log_prob = temp_metrics.get("sequence_log_prob")
                except Exception as e:
                    print(f"Error generating sample {i+1} or logprob for qid {question_id}: {e}")
                    sample_answer = f"Error: {e}"
                results['sample_answers'].append(sample_answer)
                results['sequence_log_probs'].append(sample_log_prob)
        return results

    @torch.no_grad()
    def calculate_ptrue_probability(self,
                                    question: str,
                                    main_answer: str,
                                    sample_answers: List[str],
                                    num_samples_in_prompt: int = 5) -> Optional[float]:
        if not main_answer or not sample_answers:
            return None
        ptrue_prompt = self._get_ptrue_prompt(question, main_answer, sample_answers, num_samples_in_prompt)
        try:
            if self.supports_chat_template:
                messages = [{'role': 'user', 'content': ptrue_prompt}]
                input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt").to(self.device)
            else:
                encoding = self.tokenizer(ptrue_prompt, return_tensors="pt", truncation=False)
                input_ids = encoding.input_ids.to(self.device)

            max_model_len = getattr(self.tokenizer, 'model_max_length', getattr(self.config, 'max_position_embeddings', 512))
            if input_ids.shape[1] >= max_model_len:
                print(f"Warning: PTrue prompt length ({input_ids.shape[1]}) > model max length ({max_model_len}). Returning None.")
                return None
        except Exception as e:
            print(f"Error preparing PTrue input: {e}")
            return None
        try:
            outputs = self.model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            token_id_A, token_id_B = None, None
            for tok_str in ["A", " A"]:
                 ids = self.tokenizer.encode(tok_str, add_special_tokens=False)
                 if len(ids) == 1: token_id_A = ids[0]; break
            for tok_str in ["B", " B"]:
                 ids = self.tokenizer.encode(tok_str, add_special_tokens=False)
                 if len(ids) == 1: token_id_B = ids[0]; break
            if token_id_A is None or token_id_B is None:
                 print(f"Warning: Could not find single token IDs for 'A' or 'B'. PTrue failed.")
                 return None
            logits_A = next_token_logits[0, token_id_A]
            logits_B = next_token_logits[0, token_id_B]
            log_probs_AB = torch.log_softmax(torch.stack([logits_A, logits_B]), dim=0)
            prob_A = torch.exp(log_probs_AB[0]).item()
            if np.isnan(prob_A) or np.isinf(prob_A):
                print(f"Warning: PTrue probability is NaN or Inf. Returning None.")
                return None
            return float(prob_A)
        except Exception as e:
            print(f"Error calculating PTrue logits/probabilities: {e}")
            return None