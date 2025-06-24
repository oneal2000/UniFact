import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Dict, Any

class LLMFactVerifier:
    def __init__(self, model_name: str, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        
        print(f"Initializing LLM FV with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set LLM FV tokenizer pad_token to eos_token: {self.tokenizer.pad_token}")

        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=model_dtype
        )
        self.model.eval()

        self.generation_config = {
            "num_beams": 1,
            "do_sample": False,
            "max_new_tokens": 10,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True
        }

    def _generate(self, prompt: str, generation_config: Optional[Dict[str, Any]] = None) -> str:
        if generation_config is None:
            generation_config = self.generation_config

        messages = [{'role': 'user', 'content': prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(next(self.model.parameters()).device)
        attention_mask = torch.ones_like(input_ids)

        with torch.cuda.amp.autocast(enabled=(input_ids.device.type == 'cuda' and torch.is_autocast_enabled())):
            output = self.model.generate(input_ids, attention_mask=attention_mask, **generation_config)
        generated_ids = output.sequences[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _parse_score(self, response_text: str) -> Optional[float]:
        try:
            float_match = re.search(r"(?:^|[^\d])([0-9]*\.?[0-9]+)(?:[^\d]|$)", response_text)
            if float_match:
                score = float(float_match.group(1))
                if 0 <= score <= 1: return score
        except: pass
        
        response_lower = response_text.lower()
        percent_match = re.search(r"(\d+)%", response_lower)
        if percent_match:
            try:
                score = float(percent_match.group(1)) / 100
                if 0 <= score <= 1: return score
            except: pass
        return None

    def verify(self, question: str, answer: str, retrieved_passages: List[str], max_retries: int = 3) -> Optional[float]:
        if not isinstance(retrieved_passages, list) or not all(isinstance(p, str) for p in retrieved_passages):
            retrieved_passages = []
            print(f"Warning: Invalid retrieved_passages for LLM FV for Q '{question[:30]}...'. Using empty passages.")

        prompt = f"""Based on the following information, determine if the generated answer contains hallucinations (inaccurate or fabricated information):

Question: {question}

Generated Answer: {answer}

Reference Documents:
{' '.join(retrieved_passages)}

Please carefully analyze whether the generated answer is consistent with the information in the reference documents. If the answer contains information not mentioned in or contradicting the reference documents, it should be considered as containing hallucinations.

Rate the hallucination level on a scale from 0 to 1:
- 0: Completely reliable (no hallucination)
- 1: Completely unreliable (severe hallucination)

Output only a number between 0 and 1, no explanation needed."""

        for attempt in range(max_retries):
            try:
                current_gen_config = self.generation_config.copy()
                if attempt > 0:
                    print(f"Retrying LLM FV... Attempt {attempt + 1}/{max_retries}")
                    current_gen_config['do_sample'] = True
                    current_gen_config['temperature'] = 0.7 + (attempt * 0.05)
                    current_gen_config['top_p'] = 0.9
                
                response = self._generate(prompt, current_gen_config).strip()
                score = self._parse_score(response)
                if score is not None:
                    return score
            except Exception as e:
                print(f"Error during LLM FV on attempt {attempt + 1} for Q '{question[:30]}...': {e}")
        return None

    def cleanup(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 