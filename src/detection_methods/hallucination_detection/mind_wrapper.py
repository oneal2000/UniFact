import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
import gc

def format_input_for_feature_extraction(tokenizer, model_config, title: str, text_answer: str):
    model_family = model_config.model_type.lower() if hasattr(model_config, 'model_type') else "unknown"
    is_chat = (hasattr(tokenizer, 'apply_chat_template') and 
               tokenizer.chat_template is not None)

    messages = []
    if is_chat:
        if "llama" in model_family or "qwen" in model_family:
            messages.append({"role": "user", "content": f"Question: Tell me something about {title}.\nAnswer:"})
            messages.append({"role": "assistant", "content": text_answer.strip()})
        else:
            prompt_str = f"USER: Question: Tell me something about {title}.\nAnswer: ASSISTANT: {text_answer.strip()}"
            return tokenizer.encode(prompt_str, add_special_tokens=True, truncation=True, max_length=tokenizer.model_max_length or 2048)

        try:
            return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, truncation=True, max_length=tokenizer.model_max_length or 2048)
        except Exception as e:
            print(f"Error applying chat template for feature extraction (model: {model_family}, title: {title}): {e}. Falling back.")
            fallback_prompt = f"Question: Tell me something about {title}.\nAnswer: {text_answer.strip()}"
            return tokenizer.encode(fallback_prompt, add_special_tokens=True, truncation=True, max_length=tokenizer.model_max_length or 2048)
    else: 
        base_prompt = f"Question: Tell me something about {title}.\nAnswer:"
        return tokenizer.encode(base_prompt + text_answer.strip(), add_special_tokens=True, truncation=True, max_length=tokenizer.model_max_length or 2048)


@torch.no_grad()
def extract_features_for_text(text_to_process: str, title: str, model, tokenizer, device, model_config):
    input_ids_list = format_input_for_feature_extraction(tokenizer, model_config, title, text_to_process)
    if not input_ids_list:
        print(f"Warning: Tokenization failed for text (title: {title}). Skipping feature extraction.")
        return None, None

    input_ids = torch.tensor([input_ids_list]).to(device)
    attention_mask = torch.ones_like(input_ids)

    if input_ids.shape[1] == 0:
        print(f"Warning: Empty input_ids after tokenization for text (title: {title}). Skipping.")
        return None, None
    if input_ids.shape[1] > (tokenizer.model_max_length or 2048):
        print(f"Warning: Input length {input_ids.shape[1]} exceeds max length for text (title: {title}). Might be truncated by model.")


    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        all_hidden_states = outputs.hidden_states

        if not all_hidden_states or len(all_hidden_states) <= 1:
            print(f"Warning: Failed to retrieve sufficient hidden states from model pass for title: {title}.")
            return None, None

        # 1. hd_last_token_avg_layers: Mean pooling of last token's hidden state across relevant transformer layers.
        last_token_hidden_states_across_layers = []
        for i in range(1, len(all_hidden_states)):
            last_token_hs_layer_i = all_hidden_states[i][0, -1, :]
            last_token_hidden_states_across_layers.append(last_token_hs_layer_i.clone().detach())

        if not last_token_hidden_states_across_layers:
            print(f"Warning: No transformer layer hidden states found for last token (title: {title}).")
            hd_last_token_avg = None
        else:
            stacked_hs_last_token = torch.stack(last_token_hidden_states_across_layers, dim=0) # (num_layers, hidden_size)
            hd_last_token_avg = torch.mean(stacked_hs_last_token, dim=0).cpu().tolist()


        # 2. hd_last_layer_avg_tokens: Mean pooling of all tokens' hidden states from the *last* transformer layer.
        model_family = model.config.model_type.lower() if hasattr(model.config, 'model_type') else "unknown"
        is_chat = (hasattr(tokenizer, 'apply_chat_template') and 
                   tokenizer.chat_template is not None)
        start_at_idx = 0
        if is_chat:
            prompt_only_messages = []
            if "llama" in model.config.model_type.lower() or "qwen" in model.config.model_type.lower():
                prompt_only_messages.append({"role": "user", "content": f"Question: Tell me something about {title}.\nAnswer:"})

            if prompt_only_messages:
                try:
                    prompt_only_ids = tokenizer.apply_chat_template(prompt_only_messages, tokenize=True, add_generation_prompt=True)
                    start_at_idx = len(prompt_only_ids)
                except Exception:
                    print(f"Error applying chat template for prompt-only tokenization (model: {model.config.model_type}, title: {title}). Falling back to default.")
                    start_at_idx = 0
        else:
            base_prompt = f"Question: Tell me something about {title}.\nAnswer:"
            base_prompt_ids = tokenizer.encode(base_prompt, add_special_tokens=True, truncation=True, max_length=tokenizer.model_max_length or 2048)
            start_at_idx = len(base_prompt_ids)

        last_layer_all_token_hs = all_hidden_states[-1][0] 

        if start_at_idx < last_layer_all_token_hs.shape[0]:
            generated_part_hs = last_layer_all_token_hs[start_at_idx:, :]
            if generated_part_hs.shape[0] > 0:
                hd_last_layer_avg_tokens = torch.mean(generated_part_hs, dim=0).clone().detach().cpu().tolist()
            else:
                print(f"Warning: No tokens found after start_at_idx {start_at_idx} for last_layer_avg (title: {title}). Averaging all tokens from last layer.")
                hd_last_layer_avg_tokens = torch.mean(last_layer_all_token_hs, dim=0).clone().detach().cpu().tolist()
        else:
            print(f"Warning: start_at_idx {start_at_idx} out of bounds for last_layer_avg (title: {title}). Averaging all tokens from last layer.")
            hd_last_layer_avg_tokens = torch.mean(last_layer_all_token_hs, dim=0).clone().detach().cpu().tolist()


        return hd_last_token_avg, hd_last_layer_avg_tokens

    except Exception as e:
        print(f"Error during feature extraction for text (title: {title}): {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed trace
        return None, None

class MindMLP(nn.Module):
    def __init__(self, input_size: int, dropout_rate: float = 0.2):
        super().__init__()
        if input_size <= 0:
            raise ValueError(f"Invalid input_size for MLP: {input_size}")
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Output logits for 2 classes (0: non-hallu, 1: hallu)
        )
        print(f"MindMLP created with input size {input_size}")

    def forward(self, x):
        return self.layers(x)

class MINDWrapper:
    def __init__(self, classifier_model_path: str, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.classifier_model_path = classifier_model_path
        self.classifier: Optional[MindMLP] = None
        self.input_size: Optional[int] = None

        print(f"Initializing MINDWrapper on device: {self.device}")
        print(f"Loading MIND classifier from: {self.classifier_model_path}")

        try:
            if not os.path.exists(self.classifier_model_path):
                raise FileNotFoundError(f"MIND classifier checkpoint not found at {self.classifier_model_path}")

            checkpoint = torch.load(self.classifier_model_path, map_location='cpu') # Load to CPU first

            if 'input_size' not in checkpoint:
                raise KeyError("Checkpoint must contain 'input_size' key.")
            self.input_size = checkpoint['input_size']
            if not isinstance(self.input_size, int) or self.input_size <= 0:
                 raise ValueError(f"Invalid 'input_size' ({self.input_size}) found in checkpoint.")
            print(f"Loaded MLP input size from checkpoint: {self.input_size}")

            self.classifier = MindMLP(input_size=self.input_size)
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            self.classifier.to(self.device)
            self.classifier.eval()

            print("MIND classifier loaded successfully.")

        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            raise
        except KeyError as e:
            print(f"ERROR: Missing key {e} in MIND checkpoint file: {self.classifier_model_path}")
            raise ValueError(f"Invalid checkpoint format: Missing key {e}") from e
        except Exception as e:
            print(f"ERROR: Failed to load MIND classifier from {self.classifier_model_path}: {e}")
            import traceback; traceback.print_exc()
            self.classifier = None # Ensure classifier is None on failure
            raise # Re-raise the exception to signal failure


    def _extract_mind_features_for_qa(self,
                                      question: str,
                                      llm_answer: str,
                                      target_model, # The actual LLM being evaluated
                                      target_tokenizer, # The tokenizer for the target LLM
                                      device: torch.device
                                      ) -> Optional[List[float]]:

        if not llm_answer:
             print("Warning [MIND Feature Extraction]: llm_answer is empty. Cannot extract features.")
             return None

        try:
            hd_last_token_avg, hd_last_layer_avg = extract_features_for_text(
                text_to_process=llm_answer,
                title=question,
                model=target_model,
                tokenizer=target_tokenizer,
                device=device,
                model_config=target_model.config
            )

            if hd_last_token_avg is None or hd_last_layer_avg is None:
                print(f"Warning [MIND Feature Extraction]: Failed to extract one or both features for Q: '{question[:50]}...' A: '{llm_answer[:50]}...'")
                return None

            combined_features = hd_last_token_avg + hd_last_layer_avg

            if len(combined_features) != self.input_size:
                print(f"CRITICAL ERROR [MIND Feature Extraction]: Feature dimension mismatch! Expected {self.input_size}, got {len(combined_features)}. Check feature extraction logic and classifier training.")
                return None

            return combined_features

        except Exception as e:
            print(f"Error during MIND feature extraction for QA: {e}")
            print(f"  Question: {question[:100]}...")
            print(f"  Answer: {llm_answer[:100]}...")
            import traceback; traceback.print_exc()
            return None


    @torch.no_grad()
    def calculate_score(self,
                        question: str,
                        llm_answer: str,
                        target_model, # Pass the model being evaluated
                        target_tokenizer # Pass its tokenizer
                        ) -> Optional[float]:

        if self.classifier is None:
            print("ERROR [MINDWrapper]: Classifier not loaded. Cannot calculate score.")
            return None

        features = self._extract_mind_features_for_qa(
            question=question,
            llm_answer=llm_answer,
            target_model=target_model,
            target_tokenizer=target_tokenizer,
            device=target_model.device # Use the device the target model is on
        )

        if features is None:
            print(f"MIND score calculation failed for Q: '{question[:50]}...' due to feature extraction error.")
            return None

        try:
            feature_tensor = torch.tensor([features], dtype=torch.float32).to(self.device)
        except Exception as tensor_e:
             print(f"Error converting MIND features to tensor: {tensor_e}")
             return None

        try:
            logits = self.classifier(feature_tensor) # [batch_size=1, num_classes=2]
        except Exception as mlp_e:
            print(f"Error during MIND MLP forward pass: {mlp_e}")
            return None

        try:
            probs = F.softmax(logits, dim=1)
            hallucination_prob = probs[0, 1].item()

            if np.isnan(hallucination_prob) or np.isinf(hallucination_prob):
                 print(f"Warning: MIND hallucination probability is NaN or Inf. Logits: {logits.tolist()}")
                 return None

            return float(hallucination_prob) # Score is the probability of being hallucinated

        except Exception as prob_e:
            print(f"Error calculating softmax/probability for MIND score: {prob_e}")
            return None

    def cleanup(self):
        print("Cleaning up MINDWrapper classifier...")
        del self.classifier
        self.classifier = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared after MINDWrapper cleanup.")