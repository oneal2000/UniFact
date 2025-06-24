import torch
import numpy as np
from typing import List, Dict, Any, Optional
import gc
from itertools import combinations
import math

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers library not found.")
    print("Please install it: pip install sentence-transformers")
    print("SIndex detection method will be unavailable.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    class SentenceTransformer:
        def __init__(self, model_name_or_path, device=None): pass
        def encode(self, sentences, convert_to_tensor=False, normalize_embeddings=False): return None

try:
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn library not found.")
    print("Please install it: pip install scikit-learn")
    print("SIndex detection method will be unavailable.")
    SKLEARN_AVAILABLE = False
    class AgglomerativeClustering:
        def __init__(self, n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=None): pass
        def fit_predict(self, X): return None


DEFAULT_SINDEX_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

class SIndexCalculator:
    def __init__(self,
                 model_name_or_path: Optional[str] = None,
                 device: str = 'auto',
                 clustering_threshold: float = 0.95):
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not SKLEARN_AVAILABLE:
            raise ImportError("sentence-transformers and scikit-learn libraries are required for SIndex.")

        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_name = model_name_or_path if model_name_or_path else DEFAULT_SINDEX_EMBEDDING_MODEL

        self.distance_threshold = 1.0 - clustering_threshold
        if not (0 < self.distance_threshold <= 1):
             print(f"Warning: Calculated distance threshold {self.distance_threshold} is outside (0, 1]. Clamping to (0, 1]. Original cosine threshold was {clustering_threshold}")
             self.distance_threshold = max(1e-6, min(1.0, self.distance_threshold)) # Clamp to avoid issues, ensure > 0


        print(f"Initializing SIndexCalculator with embedding model: {self.model_name} on {self.device}")
        print(f"Using distance threshold: {self.distance_threshold:.4f} (Cosine Similarity Threshold: {clustering_threshold:.4f})")

        try:
            self.embedding_model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Sentence transformer model '{self.model_name}' loaded successfully for SIndex.")
        except Exception as e:
            print(f"FATAL Error loading sentence transformer model '{self.model_name}' for SIndex: {e}")
            self.embedding_model = None
            raise e

    @torch.no_grad()
    def _get_embeddings(self, sentences: List[str], question: str) -> Optional[np.ndarray]:
        if not self.embedding_model or not sentences:
            return None
        try:
            contextual_sentences = [f"{question} [SEP] {s}" for s in sentences]

            embeddings = self.embedding_model.encode(
                contextual_sentences,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=self.device
            )
            return embeddings.cpu().numpy()
        except Exception as e:
            print(f"Error during SIndex sentence encoding: {e}")
            return None

    def _compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        similarity = np.dot(emb1, emb2)
        return float(np.clip(similarity, -1.0, 1.0))

    def _cluster_sequences(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        if embeddings is None or len(embeddings) < 1:
             print("Warning: Cannot cluster with zero embeddings.")
             return None
        if len(embeddings) == 1:
             return np.array([0]) # Single item forms a single cluster

        try:
            clustering_model = AgglomerativeClustering(
                n_clusters=None,
                metric='cosine',
                linkage='average',
                distance_threshold=self.distance_threshold
            )
            labels = clustering_model.fit_predict(embeddings)
            return labels
        except Exception as e:
            print(f"Error during SIndex Agglomerative Clustering: {e}")
            return None

    def calculate_score(self, question: str, main_answer: str, sample_answers: List[str]) -> Optional[float]:
        if not self.embedding_model:
            print("Error: SIndex embedding model not available.")
            return None

        valid_samples = [s for s in sample_answers if isinstance(s, str) and s.strip() and not s.startswith("Error:")]
        all_answers = []
        if isinstance(main_answer, str) and main_answer.strip() and not main_answer.startswith("Error:"):
            all_answers.append(main_answer)
        all_answers.extend(valid_samples)

        P = len(all_answers)

        if P < 1:
            print(f"Warning: No valid answers provided for SIndex calculation (Q: {question[:50]}...). Returning None.")
            return None
        if P == 1:
            print(f"Warning: Only 1 valid answer provided for SIndex calculation (Q: {question[:50]}...). SINdex is 0.")
            return 0.0

        # 1. Get Embeddings
        embeddings = self._get_embeddings(all_answers, question)
        if embeddings is None or embeddings.shape[0] != P:
            print("Error: Failed to get valid embeddings for SIndex.")
            return None

        # 2. Cluster Embeddings
        cluster_labels = self._cluster_sequences(embeddings)
        if cluster_labels is None:
            print("Error: Failed to cluster embeddings for SIndex.")
            return None

        # 3. Calculate SINdex
        try:
            unique_labels = sorted(list(set(cluster_labels)))
            k = len(unique_labels)

            adjusted_proportions = []

            for label in unique_labels:
                cluster_indices = np.where(cluster_labels == label)[0]
                cluster_size = len(cluster_indices)
                p_i = cluster_size / P

                avg_cos_sim_Ci = 1.0
                if cluster_size > 1:
                    total_cos_sim = 0
                    num_pairs = 0
                    cluster_embeddings = embeddings[cluster_indices]
                    for i, j in combinations(range(cluster_size), 2):
                        sim = self._compute_cosine_similarity(cluster_embeddings[i], cluster_embeddings[j])
                        total_cos_sim += sim
                        num_pairs += 1
                    if num_pairs > 0:
                         avg_cos_sim_Ci = total_cos_sim / num_pairs
                    else:
                         avg_cos_sim_Ci = 1.0

                p_prime_i = p_i * avg_cos_sim_Ci
                p_prime_i = max(0.0, p_prime_i)
                adjusted_proportions.append(p_prime_i)

            sum_p_prime = sum(adjusted_proportions)
            if sum_p_prime <= 1e-9:
                 print(f"Warning: Sum of adjusted proportions is near zero ({sum_p_prime}) for SIndex (QID: {question[:50]}...). Returning None.")
                 return None

            normalized_p_prime = [p / sum_p_prime for p in adjusted_proportions]

            sindex_score = 0.0
            for p_norm in normalized_p_prime:
                if p_norm > 1e-9:
                    sindex_score -= p_norm * math.log2(p_norm)

            if np.isnan(sindex_score) or np.isinf(sindex_score):
                 print(f"Warning: SINdex calculation resulted in NaN or Inf. Returning None.")
                 return None

            return float(sindex_score)

        except Exception as e:
            print(f"Error during SINdex calculation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cleanup(self):
        print(f"Cleaning up SIndexCalculator model '{self.model_name}'...")
        del self.embedding_model
        self.embedding_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared after SIndex model cleanup.")