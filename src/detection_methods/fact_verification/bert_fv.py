import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Optional, Dict, Any

class BertFVModel(nn.Module):
    """BERT-based fact verification model."""
    def __init__(self, model_name_or_path: str, num_labels: int = 2, dropout_rate: float = 0.1):
        super(BertFVModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path)
        
        effective_dropout_rate = dropout_rate if dropout_rate is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(effective_dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, token_type_ids: torch.Tensor = None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        
        return logits

class BertFVDataset(Dataset):
    """Dataset class for BERT fact verification."""
    def __init__(self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        question = str(item.get('question', ''))
        answer = str(item.get('generated_answer', ''))
        external_passage = str(item.get('external_passage', ''))

        text_segment1 = external_passage
        text_segment2 = f"{question}{self.tokenizer.sep_token}{answer}"

        encoding = self.tokenizer.encode_plus(
            text_segment1,
            text_segment2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation='longest_first',
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'labels': torch.tensor(item.get('label', -1), dtype=torch.long)
        }

class BertFactVerifier:
    """Main fact verification class using BERT."""
    def __init__(self, model_dir: str, device: str = "auto", max_length: int = 512):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        self.max_length = max_length
        
        print(f"Loading BERT FV model from: {model_dir}")
        
        bert_model_state_path = os.path.join(model_dir, "best_bert_fv_model_state.bin")
        if not os.path.exists(bert_model_state_path):
            bert_model_state_path = os.path.join(model_dir, "bert_fv_model_state_final_epoch.bin")
        if not os.path.exists(bert_model_state_path):
            raise FileNotFoundError(f"BERT FV Model state file not found in {model_dir}")

        original_bert_base_name = None
        training_summary_path = os.path.join(model_dir, "best_training_summary.json")
        if os.path.exists(training_summary_path):
            with open(training_summary_path, 'r') as f:
                summary = json.load(f)
                original_bert_base_name = summary.get('args', {}).get('bert_model_name')
                if original_bert_base_name:
                    print(f"Using original BERT base '{original_bert_base_name}' for classifier")
        if not original_bert_base_name:
            raise ValueError(f"Could not determine original BERT base name from {training_summary_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = BertFVModel(model_name_or_path=original_bert_base_name, num_labels=2)
        self.model.load_state_dict(torch.load(bert_model_state_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def verify(self, question: str, answer: str, retrieved_passages: List[str]) -> Optional[float]:
        if not isinstance(retrieved_passages, list):
            retrieved_passages = []
            print(f"Warning: Invalid retrieved_passages for BERT FV for Q '{question[:30]}...'. Using empty passages.")

        text_segment1 = " ".join(retrieved_passages)
        text_segment2 = f"{question}{self.tokenizer.sep_token}{answer}"

        encoding = self.tokenizer.encode_plus(
            text_segment1, text_segment2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation='longest_first',
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)

        try:
            with torch.no_grad():
                self.model.eval()
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                probs = torch.softmax(logits, dim=1)
                score = probs[0, 1].item()
                return score
        except Exception as e:
            print(f"Error during BERT FV for Q '{question[:30]}...': {e}")
            return None

    def cleanup(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 