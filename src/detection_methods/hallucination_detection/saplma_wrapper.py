import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Any
import gc

class SaplmaProbeMLP(nn.Module):
    def __init__(self, input_size: int, dropout_rate: float = 0.2):
        super().__init__()
        if input_size <= 0:
            raise ValueError(f"Invalid input_size for MLP: {input_size}")
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.layers(x)

class SAPLMAWrapper:
    def __init__(self, probe_model_path: str, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.probe_model_path = probe_model_path
        self.probe: Optional[SaplmaProbeMLP] = None
        self.input_size: Optional[int] = None
        self.dropout_rate: float = 0.2

        print(f"Initializing SAPLMAWrapper on device: {self.device}")
        print(f"Loading SAPLMA probe from: {self.probe_model_path}")

        try:
            if not os.path.exists(self.probe_model_path):
                raise FileNotFoundError(f"SAPLMA probe checkpoint not found at {self.probe_model_path}")

            checkpoint = torch.load(self.probe_model_path, map_location='cpu')

            if 'input_size' not in checkpoint:
                raise KeyError("Probe checkpoint must contain 'input_size' key.")
            self.input_size = checkpoint['input_size']
            if not isinstance(self.input_size, int) or self.input_size <= 0:
                raise ValueError(f"Invalid 'input_size' ({self.input_size}) found in probe checkpoint.")

            self.dropout_rate = checkpoint.get('dropout_rate', 0.2) 

            print(f"  Probe Input Size: {self.input_size}")
            print(f"  Probe Dropout Rate: {self.dropout_rate}")

            self.probe = SaplmaProbeMLP(input_size=self.input_size, dropout_rate=self.dropout_rate)
            self.probe.load_state_dict(checkpoint['model_state_dict'])
            self.probe.to(self.device)
            self.probe.eval()

            print("SAPLMA probe loaded successfully.")

        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            raise
        except KeyError as e:
            print(f"ERROR: Missing key {e} in SAPLMA probe checkpoint: {self.probe_model_path}")
            raise ValueError(f"Invalid probe checkpoint format: Missing key {e}") from e
        except Exception as e:
            print(f"ERROR: Failed to load SAPLMA probe from {self.probe_model_path}: {e}")
            import traceback; traceback.print_exc()
            self.probe = None
            raise

    @torch.no_grad()
    def _extract_saplma_embedding(self,
                                  statement: str,
                                  target_model,
                                  target_tokenizer,
                                  target_device,
                                  layer_index: int = -1) -> Optional[torch.Tensor]:
        if not statement:
            return None

        try:
            inputs = target_tokenizer(statement, return_tensors="pt", truncation=True, padding=False)

            if inputs.input_ids.shape[1] == 0:
                return None

            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            outputs = target_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            if not hidden_states or len(hidden_states) <= abs(layer_index):
                 return None

            target_layer_states = hidden_states[layer_index]
            last_token_hidden_state = target_layer_states[0, -1, :] # [hidden_size]

            return last_token_hidden_state.to(self.device)

        except Exception as e:
            print(f"\nError during SAPLMA embedding extraction for statement '{statement[:50]}...': {e}")
            return None

    @torch.no_grad()
    def calculate_score(self,
                        question: str,
                        llm_answer: str,
                        target_model,
                        target_tokenizer,
                        layer_index: int = -1
                       ) -> Optional[float]:
        if self.probe is None:
            print("ERROR [SAPLMAWrapper]: Probe not loaded. Cannot calculate score.")
            return None
        if not llm_answer or llm_answer.startswith("Error:"):
            return None

        statement = f"{question} {llm_answer}"

        embedding = self._extract_saplma_embedding(
            statement=statement,
            target_model=target_model,
            target_tokenizer=target_tokenizer,
            target_device=target_model.device,
            layer_index=layer_index
        )

        if embedding is None:
            print(f"SAPLMA score calculation failed for Q: '{question[:50]}...' due to embedding extraction error.")
            return None 
        
        if embedding.shape[0] != self.input_size:
             print(f"ERROR [SAPLMAWrapper]: Embedding dimension mismatch! Probe expects {self.input_size}, got {embedding.shape[0]}.")
             return None

        embedding_tensor = embedding.unsqueeze(0).to(dtype=torch.float32, device=self.device)

        try:
            logits = self.probe(embedding_tensor)
        except Exception as mlp_e:
            print(f"Error during SAPLMA probe forward pass: {mlp_e}")
            return None

        try:
            probs = F.softmax(logits, dim=1)
            inaccurate_prob = probs[0, 0].item()

            if np.isnan(inaccurate_prob) or np.isinf(inaccurate_prob):
                print(f"Warning: SAPLMA inaccurate probability is NaN or Inf. Logits: {logits.tolist()}")
                return None

            return float(inaccurate_prob)

        except Exception as prob_e:
            print(f"Error calculating softmax/probability for SAPLMA score: {prob_e}")
            return None

    def cleanup(self):
        print("Cleaning up SAPLMAWrapper probe...")
        del self.probe
        self.probe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()