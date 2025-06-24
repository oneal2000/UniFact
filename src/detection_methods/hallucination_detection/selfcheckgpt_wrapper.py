import torch
import spacy
import numpy as np
from typing import List, Dict, Any, Optional
import gc 

try:
    from selfcheckgpt.modeling_selfcheck import (
        SelfCheckMQAG,
        SelfCheckBERTScore,
        SelfCheckNgram,
        SelfCheckNLI,
    )
    SELFCHECK_AVAILABLE = True
except ImportError:
    print("Warning: selfcheckgpt library not found or specific modules missing.")
    print("Please install it: pip install selfcheckgpt")
    print("SelfCheck detection methods will be unavailable.")
    SELFCHECK_AVAILABLE = False
    class SelfCheckMQAG: pass
    class SelfCheckBERTScore: pass
    class SelfCheckNgram: pass
    class SelfCheckNLI: pass


class SelfCheckGPTWrapper:
    def __init__(self, device: str = 'auto'):
        if not SELFCHECK_AVAILABLE:
             raise ImportError("selfcheckgpt library is required but not found.")

        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Initializing SelfCheckGPT Wrapper on device: {self.device}")

        self.selfcheck_mqag: Optional[SelfCheckMQAG] = None
        self.selfcheck_bertscore: Optional[SelfCheckBERTScore] = None
        self.selfcheck_ngram_unigram: Optional[SelfCheckNgram] = None
        self.selfcheck_nli: Optional[SelfCheckNLI] = None

        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("SpaCy model loaded.")
        except OSError:
            print("Spacy 'en_core_web_sm' model not found. Please run:")
            print("python -m spacy download en_core_web_sm")
            print("Warning: Proceeding without SpaCy sentence splitting capability.")

        print("SelfCheckGPTWrapper initialization complete (Models will be loaded on demand).")

    def _load_model(self, model_type: str):
        try:
            if model_type == 'mqag' and self.selfcheck_mqag is None:
                print("Initializing SelfCheckMQAG...")
                self.selfcheck_mqag = SelfCheckMQAG(device=self.device)
            elif model_type == 'bertscore' and self.selfcheck_bertscore is None:
                print("Initializing SelfCheckBERTScore...")
                self.selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
            elif model_type == 'ngram' and self.selfcheck_ngram_unigram is None:
                print("Initializing SelfCheckNgram (Unigram)...")
                self.selfcheck_ngram_unigram = SelfCheckNgram(n=1)
            elif model_type == 'nli' and self.selfcheck_nli is None:
                print("Initializing SelfCheckNLI...")
                self.selfcheck_nli = SelfCheckNLI(device=self.device)
        except Exception as e:
            print(f"Error initializing SelfCheck model '{model_type}': {e}")
            raise

    def sentence_split(self, text: str) -> List[str]:
        if not self.nlp:
            print("Warning: SpaCy model not loaded. Treating text as a single sentence.")
            return [text.strip()] if text.strip() else []
        try:
             doc = self.nlp(text)
             return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception as e:
             print(f"Error during SpaCy sentence splitting: {e}. Treating text as single sentence.")
             return [text.strip()] if text.strip() else []


    def detect(self,
               main_answer: str,
               sample_answers: List[str],
               question: Optional[str] = None,
               methods_to_run: Optional[List[str]] = None,
               split_sentences: bool = True,
               aggregation_strategy: str = "max"
               ) -> Dict[str, Any]:
        if not main_answer or not sample_answers:
            print("Warning: main_answer or sample_answers are empty. Skipping detection.")
            return {"error": "Empty main_answer or sample_answers"}

        sentences = []
        if split_sentences:
            print("Splitting main_answer into sentences...")
            sentences = self.sentence_split(main_answer)
            if not sentences:
                print(f"Warning: Could not split main answer into sentences or result is empty: '{main_answer}'")
                return {"error": "Sentence splitting resulted in no sentences."}
        else:
            print("Treating main_answer as a single sentence.")
            if main_answer.strip(): # Ensure it's not just whitespace
                sentences = [main_answer.strip()]
            else:
                print("Warning: main_answer is empty or whitespace only.")
                return {"error": "Main answer is empty."}

        all_methods = ['mqag', 'bertscore', 'ngram', 'nli']
        if methods_to_run is None:
            methods_to_run = all_methods # Run all if None is specified
        else:
            methods_to_run = [m for m in methods_to_run if m in all_methods]
            if not methods_to_run:
                print("Warning: No valid methods specified to run.")
                return {'sentences': sentences}

        results = {'sentences': sentences} # Always include sentences

        def _aggregate_scores(scores, strategy="max"):
            """Aggregate a list of scores into a single value"""
            if scores is None or not scores:
                return None
            
            # Filter out None values
            valid_scores = [s for s in scores if s is not None and not (isinstance(s, float) and (np.isnan(s) or np.isinf(s)))]
            if not valid_scores:
                return None
                
            if strategy == "max":
                return max(valid_scores)
            elif strategy == "avg":
                return sum(valid_scores) / len(valid_scores)
            else:
                return max(valid_scores)  # Default to max

        def assign_result(method_key, score_value, num_sent):
            if score_value is None:
                 results[method_key] = None
            elif isinstance(score_value, np.ndarray):
                 if np.isnan(score_value).any() or np.isinf(score_value).any():
                      processed_scores = []
                      for v in score_value.tolist():
                          if np.isnan(v):
                              processed_scores.append(None)
                          elif np.isinf(v):
                              # Replace infinity with a large finite value
                              processed_scores.append(1e9 if v > 0 else -1e9)
                          else:
                              processed_scores.append(v)
                      # Aggregate the processed scores
                      results[method_key] = _aggregate_scores(processed_scores, aggregation_strategy)
                 else:
                      # Aggregate the numpy array scores
                      results[method_key] = _aggregate_scores(score_value.tolist(), aggregation_strategy)
            elif isinstance(score_value, list) or isinstance(score_value, tuple):
                 # Aggregate the list/tuple scores
                 results[method_key] = _aggregate_scores(list(score_value), aggregation_strategy)
            else:
                 # Single score value
                 results[method_key] = score_value


        if 'mqag' in methods_to_run:
            score = None
            try:
                self._load_model('mqag')
                score = self.selfcheck_mqag.predict(
                    sentences=sentences, passage=main_answer, sampled_passages=sample_answers,
                    num_questions_per_sent=5, scoring_method='bayes_with_alpha', beta1=0.8, beta2=0.8
                )
            except Exception as e: print(f"Error running SelfCheckMQAG: {e}")
            assign_result('mqag', score, len(sentences))


        if 'bertscore' in methods_to_run:
            score = None
            try:
                self._load_model('bertscore')
                score = self.selfcheck_bertscore.predict(
                    sentences=sentences, sampled_passages=sample_answers
                )
            except Exception as e: print(f"Error running SelfCheckBERTScore: {e}")
            assign_result('bertscore', score, len(sentences))

        if 'ngram' in methods_to_run:
            avg_score, max_score = None, None
            try:
                self._load_model('ngram')
                ngram_scores_dict = self.selfcheck_ngram_unigram.predict(
                    sentences=sentences, passage=main_answer, sampled_passages=sample_answers
                )
                if ngram_scores_dict and 'sent_level' in ngram_scores_dict:
                    if 'avg_neg_logprob' in ngram_scores_dict['sent_level']:
                         avg_score = np.array(ngram_scores_dict['sent_level']['avg_neg_logprob'])
                    if 'max_neg_logprob' in ngram_scores_dict['sent_level']:
                         max_score = np.array(ngram_scores_dict['sent_level']['max_neg_logprob'])
            except Exception as e: print(f"Error running SelfCheckNgram (Unigram): {e}")
            assign_result('ngram_unigram_avg', avg_score, len(sentences))
            assign_result('ngram_unigram_max', max_score, len(sentences))


        if 'nli' in methods_to_run:
            score = None
            try:
                self._load_model('nli')
                score = self.selfcheck_nli.predict(
                    sentences=sentences, sampled_passages=sample_answers
                )
            except Exception as e: print(f"Error running SelfCheckNLI: {e}")
            assign_result('nli', score, len(sentences))

        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results[key] = value.tolist()

        return results

    def cleanup(self):
        print("Cleaning up SelfCheckGPT models...")
        if hasattr(self, 'selfcheck_mqag') and self.selfcheck_mqag: del self.selfcheck_mqag
        if hasattr(self, 'selfcheck_bertscore') and self.selfcheck_bertscore: del self.selfcheck_bertscore
        if hasattr(self, 'selfcheck_ngram_unigram') and self.selfcheck_ngram_unigram: del self.selfcheck_ngram_unigram
        if hasattr(self, 'selfcheck_nli') and self.selfcheck_nli: del self.selfcheck_nli

        self.selfcheck_mqag = None
        self.selfcheck_bertscore = None
        self.selfcheck_ngram_unigram = None
        self.selfcheck_nli = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")