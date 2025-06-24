import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
import gc

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers library not found.")
    print("Please install it: pip install sentence-transformers")
    print("SEU detection method will be unavailable.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    class SentenceTransformer:
        def __init__(self, model_name_or_path, device=None): pass
        def encode(self, sentences, convert_to_tensor=False, normalize_embeddings=False): return None

DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

class SemanticEmbeddingUncertaintyCalculator:
    def __init__(self, model_name_or_path: Optional[str] = None, device: str = 'auto'):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers library is required for SEU but not found.")

        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name_or_path if model_name_or_path else DEFAULT_EMBEDDING_MODEL
        print(f"Initializing SEUCalculator with embedding model: {self.model_name} on {self.device}")

        try:
            self.embedding_model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Sentence transformer model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"FATAL Error loading sentence transformer model '{self.model_name}': {e}")
            print("SEU calculation will not be possible.")
            self.embedding_model = None
            raise e

    @torch.no_grad()
    def _get_embeddings(self, sentences: List[str]) -> Optional[torch.Tensor]:
        if not self.embedding_model or not sentences:
            return None
        try:
            embeddings = self.embedding_model.encode(
                sentences,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.device
            )
            return embeddings
        except Exception as e:
            print(f"Error during sentence encoding for SEU: {e}")
            return None

    def calculate_score(self, main_answer: str, sample_answers: List[str]) -> Optional[float]:
        if not self.embedding_model:
            print("Error: SEU embedding model not available.")
            return None

        valid_samples = [s for s in sample_answers if isinstance(s, str) and s.strip() and not s.startswith("Error:")]
        all_answers = []
        if isinstance(main_answer, str) and main_answer.strip() and not main_answer.startswith("Error:"):
            all_answers.append(main_answer)
        all_answers.extend(valid_samples)

        M = len(all_answers)

        # Need at least two answers to calculate pairwise similarity
        if M < 2:
            print(f"Warning: Need at least 2 valid answers for SEU, found {M}. Returning None.")
            return None

        embeddings = self._get_embeddings(all_answers)

        if embeddings is None or embeddings.shape[0] != M:
            print("Error: Failed to get valid embeddings for all answers for SEU.")
            return None

        total_similarity = 0.0
        num_pairs = 0
        try:
            cosine_matrix = torch.matmul(embeddings, embeddings.T)

            rows, cols = torch.triu_indices(M, M, offset=1, device=embeddings.device)
            pairwise_similarities = cosine_matrix[rows, cols]

            valid_similarities = pairwise_similarities[~torch.isnan(pairwise_similarities) & ~torch.isinf(pairwise_similarities)]

            if valid_similarities.numel() == 0:
                 print("Warning: No valid pairwise similarities found for SEU calculation.")
                 return None

            average_similarity = torch.mean(valid_similarities).item()
            num_pairs = valid_similarities.numel()

            seu_score = 1.0 - average_similarity

            seu_score = max(0.0, seu_score)

            return float(seu_score)

        except Exception as e:
            print(f"Error during SEU pairwise similarity calculation: {e}")
            import traceback
            traceback.print_exc()
            return None


    def cleanup(self):
        print(f"Cleaning up SEUCalculator model '{self.model_name}'...")
        del self.embedding_model
        self.embedding_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared after SEU model cleanup.")
