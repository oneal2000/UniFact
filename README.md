# AutoEval-HD-FV: A Comprehensive Evaluation Framework for Hallucination Detection and Fact Verification

AutoEval-HD-FV is a unified evaluation framework designed to systematically assess and benchmark hallucination detection and fact verification methods for Large Language Models (LLMs).

## 🚀 Quick Start Tutorial

### Installation

```bash
# Create and activate environment
conda create -n autoeval-hd-fv python=3.10
conda activate autoeval-hd-fv

# Install dependencies
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
python -m spacy download en_core_web_sm
```

### Download the dataset

Follow the instructions in the [Dataset Setup](#📋-dataset-setup) to download the dataset you want to evaluate on.

### Step 1: Run a Simple Baseline (LNPP)

Let's start with the simplest method - LNPP:

```bash
# Generate detection scores using LNPP method
python -m scripts.hallucination_detection \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name "triviaqa" \
    --dataset_type "total" \
    --detect_methods lnpp
```

This will create detection results in: `results/<model_name>/<dataset_name>/<dataset_type>/<method_name>.json`

### Step 2: Evaluate the Results

```bash
# Generate AUROC evaluation report
python -m scripts.evaluation \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --judge_model_name_or_path Qwen/Qwen2.5-32B-Instruct \
    --dataset_name "triviaqa" \
    --dataset_type "total"
```

You will get the results in `results/<model_name>/<dataset_name>/<dataset_type>/evaluation_summary.json`

```json
{
  "lnpp": {
    "method": "lnpp",
    "total_items": 500,
    "valid_scores": 500,
    "auroc": 0.747575,
    "positive_cases": 200,
    "negative_cases": 300
  },
}
```

## 🔧 How to Evaluate Your Own Detection Method

### For Hallucination Detection Methods

#### Step 1: Generate Your Detection Scores

Your method should produce a JSON file with the following format(similar to the example above):

```json
[
  {
    "qid": "triviaqa_tc_2",
    "question": "Who was the man behind The Chipmunks?",
    "main_answer": "Ross Bagdasarian Sr., also known as David Seville, was the man behind The Chipmunks. He was an American singer, songwriter,",
    "detection_score": 0.09336186038951079
  },
  {
    "qid": "triviaqa_tc_13", 
    "question": "What star sign is Jamie Lee Curtis?",
    "main_answer": "Jamie Lee Curtis was born on November 22, 1958. Her star sign is Sagittarius.",
    "detection_score": 0.08400685215989749
  }
]
```

**Important Notes:**
- `detection_score`: Higher values should indicate higher likelihood of hallucination
- `qid`: Must match the question IDs from our datasets
- `main_answer`: Should match the LLM's responses

<details>
<summary>Don't know how to generate the required data? Click to expand</summary>

#### Using Our Data Loader
```python
from src.data_loader import load_structured_qa_dataset

# Load any supported dataset
data_map = load_structured_qa_dataset("triviaqa", "data")
questions = data_map["total"]

# Each question has the format:
# {
#   "qid": "triviaqa_tc_2", 
#   "question": "Who was the man behind The Chipmunks?",
#   "golden_answer": "David Seville",
#   "golden_passages": [],
#   "type": "total"
# }
```

#### Using Our Answer Generator
```python
from src.answer_generator import AnswerGenerator

# Initialize generator for your target LLM
generator = AnswerGenerator("meta-llama/Llama-3.1-8B-Instruct")

# Generate answers for each question
for question_data in questions:
    result = generator.generate(
        question_data,
        max_new_tokens=30,
        main_temperature=0.8
    )
    
    # result contains:
    # {
    #   "main_answer": "Ross Bagdasarian Sr., also known as David Seville, ...",
    #   "sample_answers": [...],  # if needed
    #   "pp_pe_metrics": {...}    # includes lnpp, lnpe scores
    # }
```

</details>

#### Step 2: Place Your Results File

Save your results as: `results/<model_name>/<dataset_name>/<dataset_type>/<your_method_name>.json`

#### Step 3: Run Evaluation

```bash
python -m scripts.evaluation \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --judge_model_name_or_path Qwen/Qwen2.5-32B-Instruct \
    --dataset_name "triviaqa" \
    --dataset_type "total" \
    --detection_methods <your_method_name>
```

### For Fact Verification Methods

#### Step 1: Get Basic Results for Input

First, generate the basic question-answer pairs:

```bash
python -m scripts.hallucination_detection \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name "triviaqa" \
    --dataset_type "total" \
    --detect_methods lnpp
```

This creates `basic_results.json` with the format:

```json
[
  {
    "qid": "triviaqa_tc_2",
    "question": "Who was the man behind The Chipmunks?",
    "main_answer": "Ross Bagdasarian Sr., also known as David Seville, was the man behind The Chipmunks. He was an American singer, songwriter,"
  },
  {
    "qid": "triviaqa_tc_13",
    "question": "What star sign is Jamie Lee Curtis?", 
    "main_answer": "Jamie Lee Curtis was born on November 22, 1958. Her star sign is Sagittarius."
  }
]
```

#### Step 2: Generate Your Fact Verification Scores

Process the `basic_results.json` with your method and produce:

```json
[
  {
    "qid": "triviaqa_tc_2",
    "question": "Who was the man behind The Chipmunks?",
    "main_answer": "Ross Bagdasarian Sr., also known as David Seville, was the man behind The Chipmunks. He was an American singer, songwriter,",
    "detection_score": 0.15
  },
  {
    "qid": "triviaqa_tc_13",
    "question": "What star sign is Jamie Lee Curtis?",
    "main_answer": "Jamie Lee Curtis was born on November 22, 1958. Her star sign is Sagittarius.",
    "detection_score": 0.23
  }
]
```

#### Step 3: Evaluate Your Method

Save as `fv_<your_method_name>.json` and run evaluation:

```bash
python -m scripts.evaluation \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --judge_model_name_or_path Qwen/Qwen2.5-32B-Instruct \
    --dataset_name "triviaqa" \
    --dataset_type "total" \
    --detection_methods your_method_name
```

## 🧱 Built-in Baseline Methods

The framework includes some baseline methods for comparison. Here's how to use them:

### Training-Free Methods (Ready to Use)

```bash
# Run multiple training-free methods at once
python -m scripts.hallucination_detection \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name "triviaqa" \
    --dataset_type "total" \
    --detect_methods lnpp lnpe ptrue semantic_entropy seu sindex
```

**Available Methods:**
- `lnpp`
- `lnpe`
- [`SelfCheckGPT`](http://arxiv.org/abs/2303.08896)
  - `mqag`
  - `bertscore`
  - `ngram`
  - `nli`
- [`ptrue`](http://arxiv.org/abs/2207.05221)
- [`semantic_entropy`](https://www.nature.com/articles/s41586-024-07421-0)
- [`seu`](http://arxiv.org/abs/2410.22685)
- [`sindex`](http://arxiv.org/abs/2503.05980)

### Training-Required Methods

<details>
<summary>Click to expand setup instructions</summary>

#### [`EUBHD`](http://arxiv.org/abs/2311.13230) Method
```bash
# 1. Generate token frequency statistics
python -m scripts.eubhd.count --tokenizer meta-llama/Llama-3.1-8B-Instruct

# 2. Run evaluation
python -m scripts.hallucination_detection \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name "triviaqa" \
    --dataset_type "total" \
    --detect_methods eubhd \
    --eubhd_idf_path eubhd_idf/token_idf_Llama-3.1-8B-Instruct.pkl
```

#### [`SAPLMA`](https://arxiv.org/abs/2304.13734v2) Method
```bash
# 1. Extract features
python -m scripts.saplma.extract_features \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --input_dir_path training_data/SAPLMA \
    --output_dir_path saplma/Llama-3.1-8B-Instruct_-1/data

# 2. Train probe
python -m scripts.saplma.train_probe \
    --embedding_dir_path saplma/Llama-3.1-8B-Instruct_-1/data \
    --output_probe_path saplma/Llama-3.1-8B-Instruct_-1/probe.pt

# 3. Run evaluation
python -m scripts.hallucination_detection \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name "triviaqa" \
    --dataset_type "total" \
    --detect_methods saplma \
    --saplma_probe_path saplma/Llama-3.1-8B-Instruct_-1/probe.pt
```

#### [`MIND`](http://arxiv.org/abs/2403.06448) Method
```bash
# 1. Generate training data
python -m scripts.mind.generate_data \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --wiki_data_dir training_data/MIND \
    --output_dir mind/Llama-3.1-8B-Instruct/text_data

# 2. Extract features  
python -m scripts.mind.extract_features \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --generated_data_dir mind/Llama-3.1-8B-Instruct/text_data \
    --output_feature_dir mind/Llama-3.1-8B-Instruct/feature_data

# 3. Train classifier
python -m scripts.mind.train_mind \
    --feature_dir mind/Llama-3.1-8B-Instruct/feature_data \
    --output_classifier_dir mind/Llama-3.1-8B-Instruct/classifier

# 4. Run evaluation
python -m scripts.hallucination_detection \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name "triviaqa" \
    --dataset_type "total" \
    --detect_methods mind \
    --mind_classifier_path mind/Llama-3.1-8B-Instruct/classifier/mind_classifier_best.pt
```

</details>

### Built-in Fact Verification Methods

<details>
<summary>Click to expand setup instructions</summary>

#### Setup Requirements
```bash
# Download Wikipedia dump for retrieval
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
cd data/dpr && gzip -d psgs_w100.tsv.gz && cd ../..

# Setup Elasticsearch
cd data
wget -O elasticsearch-8.15.0.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz
tar zxvf elasticsearch-8.15.0.tar.gz && rm elasticsearch-8.15.0.tar.gz
cd elasticsearch-8.15.0 && nohup bin/elasticsearch & && cd ../..
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki
```

#### Train BERT Classifier

1. **Generate training data:**

```bash
python -m scripts.bert.generate_data \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --judge_model_name_or_path Qwen/Qwen2.5-32B-Instruct \
    --data_path bert/training_data.json
```

2. **Train the classifier:**

```bash
python -m scripts.bert.train_bert \
    --output_dir "bert_classifier" \
    --retrieval_type "question_only"

python -m scripts.bert.train_bert \
    --output_dir "bert_classifier" \
    --retrieval_type "question_answer"
```

#### Running Fact Verification

```bash
python -m scripts.fact_verification \
    --target_llm_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name "popqa" \
    --dataset_type "total" \
    --fv_llm_model_name Qwen/Qwen2.5-32B-Instruct \
    --bert_fv_q_model_dir bert_classifier/fv_model_question_only \
    --bert_fv_qa_model_dir bert_classifier/fv_model_question_answer
```

</details>

## 📊 Understanding Evaluation Results

### AUROC Score Interpretation

The framework uses AUROC (Area Under ROC Curve) as the primary metric:

- **Score Range**: 0.0 to 1.0  
- **Higher is Better**: Higher AUROC indicates better hallucination detection
- **Benchmark**: 0.5 = random performance, 1.0 = perfect detection

### Sample Results Comparison

```json
{
  "lnpp": {
    "method": "lnpp",
    "total_items": 100,
    "valid_scores": 100,
    "auroc": 0.7234,
    "positive_cases": 45,
    "negative_cases": 55
  },
  "my_new_method": {
    "method": "my_new_method", 
    "total_items": 100,
    "valid_scores": 100,
    "auroc": 0.8156,
    "positive_cases": 45,
    "negative_cases": 55
  }
}
```

In this example, `my_new_method` (AUROC: 0.8156) outperforms the baseline `lnpp` method (AUROC: 0.7234).

## 📋 Dataset Setup

AutoEval-HD-FV supports five major QA datasets:

### 2WikiMultihopQA

1. Download the dataset from the [official repository](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1)
2. Extract and move the folder to `data/2wikimultihopqa`

### HotpotQA

```bash
mkdir -p data/hotpotqa
wget -P data/hotpotqa/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

### PopQA

```bash
mkdir -p data/popqa
wget -P data/popqa https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/main/data/popQA.tsv
```

### TriviaQA

```bash
mkdir -p data/triviaqa
wget -P data/triviaqa https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz
tar -xvzf data/triviaqa/triviaqa-unfiltered.tar.gz -C data/triviaqa
```

### Natural Questions (NQ)

Download the `NQ-open.efficientqa.dev.1.1.jsonl` file from the [Google Research repository](https://github.com/google-research-datasets/natural-questions) and place it in `data/nq/`.