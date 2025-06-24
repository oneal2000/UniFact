import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional
from collections import defaultdict
import itertools
import math
import gc

try:
    from scipy.special import logsumexp
except ImportError:
    print("Warning: scipy.special.logsumexp not found. Using manual implementation (less numerically stable for extreme values).")
    def logsumexp(a, axis=None):
        a = np.asarray(a)
        a_max = np.amax(a, axis=axis, keepdims=True)
        if a_max.ndim > 0:
            a_max[~np.isfinite(a_max)] = 0
        elif not np.isfinite(a_max):
            a_max = 0
        tmp = np.exp(a - a_max)
        with np.errstate(divide='ignore'):
            s = np.sum(tmp, axis=axis, keepdims=False)
            out = np.log(s)
        if not isinstance(a_max, float):
             a_max = a_max.squeeze(axis=axis)
        out += a_max
        return out


class SemanticEntropyCalculator:
    def __init__(self, nli_model_name_or_path: str, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.nli_model_name_or_path = nli_model_name_or_path
        print(f"Initializing SemanticEntropyCalculator with NLI model: {nli_model_name_or_path} on {self.device}")

        try:
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name_or_path)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name_or_path)
            self.nli_model.to(self.device)
            self.nli_model.eval()
            self.entailment_id = 2
            if hasattr(self.nli_model.config, 'label2id') and 'entailment' in self.nli_model.config.label2id:
                 self.entailment_id = self.nli_model.config.label2id['entailment']
                 print(f"Detected NLI entailment label ID: {self.entailment_id}")
            else:
                 print(f"Warning: Could not detect NLI entailment label ID. Assuming ID: {self.entailment_id}. Check NLI model config if issues arise.")

        except Exception as e:
            print(f"FATAL Error loading NLI model/tokenizer '{nli_model_name_or_path}': {e}")
            print("Semantic Entropy calculation will not be possible.")
            self.nli_model = None
            self.nli_tokenizer = None

        print("SemanticEntropyCalculator initialized.")

    @torch.no_grad()
    def _check_entailment(self, premise: str, hypothesis: str) -> bool:
        if not self.nli_model or not self.nli_tokenizer:
            return False

        try:
            input_text = f"{premise}{self.nli_tokenizer.sep_token}{hypothesis}"
            inputs = self.nli_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(self.device)

            if inputs['input_ids'].shape[1] == 0:
                 print(f"Warning: NLI input truncated to zero length. Premise: '{premise[:50]}...', Hypothesis: '{hypothesis[:50]}...'. Treating as non-entailment.")
                 return False

            outputs = self.nli_model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
            return predicted_class_id == self.entailment_id
        except Exception as e:
            print(f"Error during NLI entailment check: {e}")
            print(f"Premise (start): {premise[:100]}...")
            print(f"Hypothesis (start): {hypothesis[:100]}...")
            return False

    def _get_clusters(self, question: str, answers: List[str]) -> List[int]:
        if not self.nli_model or not answers:
            return list(range(len(answers))) # Return trivial clustering if no NLI or no answers

        num_answers = len(answers)
        contextual_answers = [f"{question} {ans}" for ans in answers if isinstance(ans, str) and ans]
        valid_indices = [i for i, ans in enumerate(answers) if isinstance(ans, str) and ans]
        if not contextual_answers:
             return [-1] * num_answers # No valid answers to cluster

        clusters = [[valid_indices[0]]]
        cluster_assignments = [-1] * num_answers
        if valid_indices:
            cluster_assignments[valid_indices[0]] = 0

        for m_idx, original_idx in enumerate(valid_indices[1:], start=1):
            current_answer_context = contextual_answers[m_idx]
            assigned_to_cluster = False
            for cluster_id, cluster_indices in enumerate(clusters):
                representative_original_idx = cluster_indices[0]
                representative_m_idx = valid_indices.index(representative_original_idx)
                representative_answer_context = contextual_answers[representative_m_idx]

                rep_entails_curr = self._check_entailment(representative_answer_context, current_answer_context)
                curr_entails_rep = self._check_entailment(current_answer_context, representative_answer_context)

                if rep_entails_curr and curr_entails_rep:
                    clusters[cluster_id].append(original_idx)
                    cluster_assignments[original_idx] = cluster_id
                    assigned_to_cluster = True
                    break

            if not assigned_to_cluster:
                new_cluster_id = len(clusters)
                clusters.append([original_idx])
                cluster_assignments[original_idx] = new_cluster_id

        return cluster_assignments


    def calculate_score(self,
                        question: str,
                        answers: List[str],
                        sequence_log_probs: List[Optional[float]]
                       ) -> Optional[float]:
        if not self.nli_model:
            print("Error: NLI model not available for Semantic Entropy calculation.")
            return None

        if len(answers) != len(sequence_log_probs):
            print(f"Error: Mismatch between number of answers ({len(answers)}) and log probabilities ({len(sequence_log_probs)}). Cannot calculate SE.")
            return None

        valid_indices = [i for i, logp in enumerate(sequence_log_probs) if logp is not None and isinstance(answers[i], str) and answers[i]]
        if not valid_indices:
            print("Warning: No valid answers with log probabilities found. Cannot calculate SE.")
            return None
        if len(valid_indices) < 2:
             print("Warning: Need at least two valid answers with log probabilities for meaningful SE calculation. Returning 0.0 entropy.")
             return 0.0

        valid_answers = [answers[i] for i in valid_indices]
        valid_log_probs = [sequence_log_probs[i] for i in valid_indices]

        # 1. Cluster the valid answers
        cluster_assignments_valid = self._get_clusters(question, valid_answers)

        # 2. Group log probabilities by cluster ID
        log_probs_by_cluster = defaultdict(list)
        for i, cluster_id in enumerate(cluster_assignments_valid):
            if cluster_id != -1: # Should not happen if we pre-filtered, but check anyway
                 log_probs_by_cluster[cluster_id].append(valid_log_probs[i])

        if not log_probs_by_cluster:
             print("Warning: Clustering resulted in no valid clusters. Cannot calculate SE.")
             return None

        # 3. Calculate cluster probabilities using LogSumExp
        cluster_log_probs = []
        for cluster_id in sorted(log_probs_by_cluster.keys()):
            cluster_log_prob = logsumexp(log_probs_by_cluster[cluster_id])
            cluster_log_probs.append(cluster_log_prob)

        total_log_prob_sum = logsumexp(cluster_log_probs)
        normalized_log_probs = np.array(cluster_log_probs) - total_log_prob_sum
        normalized_probs = np.exp(normalized_log_probs)

        if not np.isclose(np.sum(normalized_probs), 1.0, atol=1e-5):
             print(f"Warning: Normalized cluster probabilities do not sum to 1 (sum={np.sum(normalized_probs)}). Check calculations.")
             normalized_probs /= np.sum(normalized_probs)


        # 4. Calculate Semantic Entropy: - sum(p * log2(p))
        semantic_entropy = 0.0
        for p in normalized_probs:
            if p > 1e-9: # Avoid log(0) issues
                semantic_entropy -= p * math.log2(p)

        if np.isnan(semantic_entropy) or np.isinf(semantic_entropy):
             print(f"Warning: Semantic Entropy calculation resulted in NaN or Inf. Returning None.")
             return None

        return float(semantic_entropy)

    def cleanup(self):
        print("Cleaning up SemanticEntropyCalculator NLI model...")
        del self.nli_model
        del self.nli_tokenizer
        self.nli_model = None
        self.nli_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared after NLI model cleanup.")