import os
import torch
import numpy as np
import spacy
import pickle
import gc
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

DEFAULT_EUBHD_GAMMA = 0.9

class EUBHDCalculator:
    """
    Calculates the EUBHD (Focus) hallucination score based on token loss,
    attention patterns, and optional IDF weighting. Adapts logic from the
    Focus repository (https://github.com/zthang/focus).
    """
    def __init__(self,
                 device: str = 'auto',
                 idf_path: Optional[str] = None,
                 gamma: float = DEFAULT_EUBHD_GAMMA,
                 only_keyword: bool = True,
                 use_penalty: bool = True,
                 use_idf: bool = True
                 ):
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Initializing EUBHDCalculator on device: {self.device}")

        self.gamma = gamma if use_penalty else 0.0
        self.only_keyword = only_keyword
        self.use_penalty = use_penalty
        self.use_idf = use_idf

        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("SpaCy model 'en_core_web_sm' loaded for EUBHD.")
        except OSError:
            print("\nERROR: Spacy 'en_core_web_sm' model not found. EUBHD requires it.")
            print("Please run: python -m spacy download en_core_web_sm")
            raise ImportError("SpaCy model 'en_core_web_sm' not found, required for EUBHD.")
        except Exception as e:
             print(f"Error loading SpaCy model: {e}")
             raise 
        
        self.token_idf_np = None
        self.token_idf_tensor = None
        if self.use_idf:
            if idf_path and os.path.exists(idf_path):
                try:
                    with open(idf_path, "rb") as f:
                        self.token_idf_np = pickle.load(f)

                    if not isinstance(self.token_idf_np, np.ndarray):
                         print(f"Warning: IDF data loaded from {idf_path} is not a NumPy array. IDF weighting disabled.")
                         self.token_idf_np = None
                         self.use_idf = False
                    else:
                         print(f"Token IDF data loaded successfully from {idf_path} (Shape: {self.token_idf_np.shape}).")
                         self.token_idf_tensor = torch.tensor(self.token_idf_np, dtype=torch.float32).to(self.device)

                except Exception as e:
                    print(f"Error loading token IDF data from {idf_path}: {e}. Disabling IDF weighting.")
                    self.token_idf_np = None
                    self.token_idf_tensor = None
                    self.use_idf = False
            else:
                print(f"Warning: use_idf=True but idf_path '{idf_path}' not provided or not found. Disabling IDF weighting.")
                self.use_idf = False

        self.NER_type = ['PERSON', 'DATE', 'ORG', "GPE", "NORP", 'ORDINAL', 'PRODUCT', 'CARDINAL', 'LOC', "FAC", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "TIME", "PERCENT", "MONEY", "QUANTITY"]
        self.pos_tag = ["NOUN", "NUM", "PROPN"]

        print(f"EUBHDCalculator initialized with: gamma={self.gamma}, only_keyword={self.only_keyword}, use_penalty={self.use_penalty}, use_idf={self.use_idf}")


    @torch.no_grad()
    def calculate_score(self,
                        question: str,
                        llm_answer: str,
                        target_model: AutoModelForCausalLM,
                        target_tokenizer: AutoTokenizer
                        ) -> Optional[float]:

        if not self.nlp:
            print("Error: SpaCy model not available in EUBHDCalculator.")
            return None
        if not llm_answer or llm_answer.startswith("Error:"):
             print(f"Skipping EUBHD scoring due to invalid llm_answer: {llm_answer[:100]}...")
             return None

        prompt_prefix = f"Question: {question}\nAnswer:\n"
        full_prompt = prompt_prefix + llm_answer

        start_token_idx = 0
        try:
            encodings = target_tokenizer(full_prompt, return_tensors="pt", truncation=False, padding=False)
            input_ids = encodings.input_ids.to(target_model.device)

            prefix_tokens = target_tokenizer(prompt_prefix, return_tensors="pt", add_special_tokens=False).input_ids
            start_token_idx = prefix_tokens.shape[1]

            if target_tokenizer.bos_token and input_ids.shape[1] > 0 and input_ids[0, 0] == target_tokenizer.bos_token_id:
                 if prefix_tokens.shape[1] == 0 or prefix_tokens[0, 0] != target_tokenizer.bos_token_id:
                    start_token_idx += 1


            max_model_len = getattr(target_tokenizer, 'model_max_length', getattr(target_model.config, 'max_position_embeddings', 2048))
            if input_ids.shape[1] > max_model_len:
                 print(f"Warning [EUBHD]: Input length ({input_ids.shape[1]}) > model max length ({max_model_len}). Truncating input_ids.")
                 input_ids = input_ids[:, :max_model_len]

            if input_ids.shape[1] <= 1:
                 print("Warning [EUBHD]: Input token length <= 1 after tokenization/truncation. Cannot calculate score.")
                 return None


        except Exception as e:
            print(f"Error during EUBHD tokenization or start index finding: {e}")
            return None

        attentions = None
        token_entropies = []

        try:
            original_output_attentions = getattr(target_model.config, 'output_attentions', False)
            target_model.config.output_attentions = True

            outputs = target_model(input_ids=input_ids, labels=input_ids, output_attentions=True, output_hidden_states=False)

            target_model.config.output_attentions = original_output_attentions

            logits = outputs.logits # Shape: [batch_size=1, sequence_length, vocab_size]
            attentions = outputs.attentions # Tuple of attentions per layer

            if logits is None or attentions is None:
                 print("Error [EUBHD]: Failed to get logits or attentions from target model.")
                 return None

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            vocab_size = shift_logits.shape[-1]
            log_softmax_logits = F.log_softmax(shift_logits, dim=-1) # [1, seq_len-1, vocab_size]

            shift_labels_cpu = shift_labels.view(-1).unsqueeze(-1).cpu() # [seq_len-1, 1]
            log_probs_all = log_softmax_logits.view(-1, vocab_size) # [seq_len-1, vocab_size]

            valid_indices_mask = (shift_labels_cpu >= 0) & (shift_labels_cpu < vocab_size)
            if not valid_indices_mask.all():
                print(f"Warning [EUBHD]: Some label indices out of vocab bounds. Clamping.")
                shift_labels_cpu = torch.clamp(shift_labels_cpu, 0, vocab_size - 1)

            token_log_probs_shifted = torch.gather(log_probs_all, 1, shift_labels_cpu.to(log_probs_all.device)).squeeze() # [seq_len-1]
            loss_per_token_shifted = -token_log_probs_shifted # NLL = -log_prob

            answer_loss_start_index = max(0, start_token_idx - 1)
            loss_per_token_answer = loss_per_token_shifted[answer_loss_start_index:]

            for i in range(answer_loss_start_index, shift_logits.shape[1]):
                 step_logits = shift_logits[0, i, :] # Logits predicting token i+1
                 if not torch.isnan(step_logits).any():
                      step_probs = F.softmax(step_logits, dim=-1)
                      step_probs = step_probs + 1e-9 # Add small epsilon
                      entropy = -torch.sum(step_probs * torch.log(step_probs)).item()
                      if not np.isnan(entropy) and not np.isinf(entropy):
                           token_entropies.append(entropy)
                      else: token_entropies.append(0.0) # Assign 0 entropy if NaN/Inf
                 else:
                      print(f"Warning [EUBHD]: NaN logits found at step {i}, assigning 0 entropy.")
                      token_entropies.append(0.0)

            loss = loss_per_token_answer.clone().detach() # Shape [answer_len]


        except Exception as e:
            print(f"Error during EUBHD forward pass or loss calculation: {e}")
            import traceback
            traceback.print_exc()
            return None

        if attentions is None or loss is None:
             print("Error: Attentions or loss tensor is None after forward pass.")
             return None

        last_layer_attention = attentions[-1].squeeze(0) # [num_heads, seq_len, seq_len]
        attention_avg_heads = torch.mean(last_layer_attention, dim=0) # [seq_len, seq_len]

        attention_for_loss = attention_avg_heads[:-1, :-1] # [seq_len-1, seq_len-1]

        if attention_for_loss.shape[0] != loss_per_token_shifted.shape[0]:
             print(f"Warning [EUBHD]: Attention dimension ({attention_for_loss.shape[0]}) mismatch with loss dimension ({loss_per_token_shifted.shape[0]}). Skipping penalty calculation.")
             self.use_penalty = False # Disable penalty dynamically if mismatch
             attention_for_loss = torch.zeros_like(loss_per_token_shifted.unsqueeze(-1).repeat(1, loss_per_token_shifted.shape[0])) # Dummy


        token_idf_weights = torch.ones_like(loss)
        if self.use_idf and self.token_idf_tensor is not None:
            answer_token_ids = input_ids[0, start_token_idx:].cpu().numpy()
            if len(target_tokenizer.vocab) != len(self.token_idf_tensor):
                print(f"Warning [EUBHD]: IDF vocab size ({len(self.token_idf_tensor)}) != tokenizer vocab size ({len(target_tokenizer.vocab)}). Disabling IDF.")
                self.use_idf = False # Disable dynamically
            else:
                try:
                    safe_answer_token_ids = np.clip(answer_token_ids, 0, len(self.token_idf_tensor) - 1)
                    idf_vals = self.token_idf_tensor[safe_answer_token_ids].to(loss.device)

                    if len(idf_vals) == len(loss):
                        token_idf_weights = idf_vals
                    else:
                        print(f"Warning [EUBHD]: IDF values length ({len(idf_vals)}) mismatch with answer loss length ({len(loss)}). Disabling IDF.")
                        self.use_idf = False
                        token_idf_weights = torch.ones_like(loss)

                except IndexError:
                    print("Error [EUBHD]: Index error looking up IDF values. Disabling IDF.")
                    self.use_idf = False
                    token_idf_weights = torch.ones_like(loss)


        total_score = 0.0
        num_scored_tokens = 0

        doc = self.nlp(llm_answer)

        for sent in doc.sents:
            for span in sent:
                word_text = span.text
                start_char, end_char = span.idx, span.idx + len(span)

                span_token_indices_full = []
                try:
                    first_token = encodings.char_to_token(start_char + len(prompt_prefix))
                    last_token = encodings.char_to_token(end_char -1 + len(prompt_prefix)) # Inclusive last char
                    if first_token is not None and last_token is not None:
                        span_token_indices_full = list(range(first_token, last_token + 1))

                except Exception as char_map_e:
                     print(f"Warning [EUBHD]: Error mapping chars {start_char}-{end_char} ('{word_text}') to tokens: {char_map_e}")
                     continue # Skip this word if mapping fails

                if not span_token_indices_full:
                     continue

                span_loss_indices = [idx - start_token_idx for idx in span_token_indices_full if idx >= start_token_idx]

                if not span_loss_indices:
                     continue

                word_avg_loss = 0.0
                word_token_count = 0
                is_keyword = (span.ent_type_ in self.NER_type or span.pos_ in self.pos_tag)

                for loss_idx in span_loss_indices:
                    if 0 <= loss_idx < len(loss):
                        current_loss = loss[loss_idx].item() # Base loss for this token

                        if self.use_penalty and is_keyword:
                             attn_step_idx = start_token_idx + loss_idx -1 # Index in the original attention matrix
                             if 0 <= attn_step_idx < attention_for_loss.shape[0]:
                                 attention_row = attention_for_loss[attn_step_idx]
                                 attn_from_answer_tokens = attention_row[answer_loss_start_index : attn_step_idx]
                                 losses_from_answer_tokens = loss_per_token_shifted[answer_loss_start_index : attn_step_idx]

                                 if attn_from_answer_tokens.numel() > 0 and losses_from_answer_tokens.numel() == attn_from_answer_tokens.numel():
                                     weight = attn_from_answer_tokens / (torch.sum(attn_from_answer_tokens) + 1e-9)
                                     penalty_val = torch.sum(weight * losses_from_answer_tokens).item() # Penalty is weighted average of previous losses
                                     current_loss += self.gamma * penalty_val

                        if self.use_idf:
                             idf_weight = token_idf_weights[loss_idx].item()
                             current_loss *= idf_weight # Multiply loss by IDF

                        word_avg_loss += current_loss
                        word_token_count += 1

                if word_token_count > 0:
                    word_avg_loss /= word_token_count

                    if not self.only_keyword or is_keyword:
                         total_score += word_avg_loss * word_token_count # Weight by num tokens in word
                         num_scored_tokens += word_token_count


        if num_scored_tokens == 0:
             print("Warning [EUBHD]: No tokens were scored (check keyword filtering?). Returning None.")
             return None

        final_passage_score = total_score / num_scored_tokens

        final_score = final_passage_score

        if np.isnan(final_score) or np.isinf(final_score):
            print(f"Warning [EUBHD]: Final score is NaN or Inf. Raw score before norm: {final_passage_score}. Returning None.")
            return None

        return float(final_score)


    def cleanup(self):
        print("Cleaning up EUBHDCalculator...")
        del self.nlp
        del self.token_idf_np
        del self.token_idf_tensor
        self.nlp = None
        self.token_idf_np = None
        self.token_idf_tensor = None
        gc.collect()