import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

from src.detection_methods.fact_verification.bert_fv import BertFVModel, BertFVDataset
from transformers import AutoTokenizer

def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler._LRScheduler
):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(data_loader, desc="Training Epoch", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, avg_loss

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs_positive_class = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs_positive_class.extend(probs)
            
    avg_loss = total_loss / len(data_loader)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': -1.0,
        'auroc': -1.0,
        'f1': -1.0
    }

    if len(np.unique(all_labels)) > 1:
        try:
            metrics['auroc'] = roc_auc_score(all_labels, all_probs_positive_class)
            binary_preds = [1 if p > 0.5 else 0 for p in all_probs_positive_class]
            metrics['f1'] = f1_score(all_labels, binary_preds, zero_division=0)
            metrics['accuracy'] = accuracy_score(all_labels, binary_preds)
        except ValueError as e:
            print(f"Could not compute some metrics: {e}")
    else:
        print("Warning: Only one class present in validation labels. AUROC is undefined.")
        try:
            binary_preds = [1 if p > 0.5 else 0 for p in all_probs_positive_class]
            metrics['accuracy'] = accuracy_score(all_labels, binary_preds)
            metrics['f1'] = f1_score(all_labels, binary_preds, zero_division=0)
        except Exception:
            pass
            
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train BERT Fact Verification Model")
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased", help="Pretrained BERT model name or path")
    parser.add_argument("--data_path", type=str, default="bert/training_data.json", help="Path to the training data JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--retrieval_type", type=str, choices=['question_only', 'question_answer'], default='question_only')
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--warmup_steps_factor", type=float, default=0.1)
    parser.add_argument("--validation_split_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from '{args.data_path}'...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        full_dataset_json = json.load(f)
    
    valid_dataset_json = [item for item in full_dataset_json if item.get('label') in [0, 1]]
    passage_key = 'question_only_passages' if args.retrieval_type == 'question_only' else 'qa_combined_passages'
    
    invalid_items = [i for i, item in enumerate(valid_dataset_json) if passage_key not in item]
    if invalid_items:
        print(f"Warning: {len(invalid_items)} items missing required passage key '{passage_key}'. These will be filtered out.")
        valid_dataset_json = [item for i, item in enumerate(valid_dataset_json) if i not in invalid_items]

    print(f"Loaded {len(full_dataset_json)} total items, {len(valid_dataset_json)} have valid labels and required passages.")

    if not valid_dataset_json:
        print("No valid data items found for training. Exiting.")
        return

    if args.validation_split_ratio > 0:
        train_data, val_data = train_test_split(
            valid_dataset_json,
            test_size=args.validation_split_ratio,
            random_state=args.seed,
            stratify=[item['label'] for item in valid_dataset_json]
        )
    else:
        train_data = valid_dataset_json
        val_data = []

    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
    train_dataset = BertFVDataset(
        train_data,
        tokenizer,
        args.max_seq_length,
        retrieval_type=args.retrieval_type
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if val_data:
        val_dataset = BertFVDataset(
            val_data,
            tokenizer,
            args.max_seq_length,
            retrieval_type=args.retrieval_type
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = BertFVModel(
        model_name_or_path=args.bert_model_name,
        num_labels=2,
        dropout_rate=args.dropout_rate
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_steps_factor)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)

    best_val_auroc = -1.0
    training_history = {'train_loss': [], 'train_acc': [], 'val_metrics': []}

    print("\n--- Starting Training ---")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        train_acc, train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scheduler
        )
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        if val_loader:
            val_metrics = evaluate_model(model, val_loader, loss_fn, device)
            training_history['val_metrics'].append(val_metrics)
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Val AUROC: {val_metrics['auroc']:.4f}, Val F1: {val_metrics['f1']:.4f}")

            if val_metrics['auroc'] > best_val_auroc:
                best_val_auroc = val_metrics['auroc']
                print(f"New best validation AUROC: {best_val_auroc:.4f}. Saving model...")
                
                model_save_dir = os.path.join(args.output_dir, f"fv_model_{args.retrieval_type}")
                os.makedirs(model_save_dir, exist_ok=True)
                
                torch.save(model.state_dict(), os.path.join(model_save_dir, "best_bert_fv_model_state.bin"))
                tokenizer.save_pretrained(model_save_dir)
                
                training_summary = {
                    "args": vars(args),
                    "best_epoch": epoch + 1,
                    "best_metrics": val_metrics
                }
                with open(os.path.join(model_save_dir, "best_training_summary.json"), 'w') as f:
                    json.dump(training_summary, f, indent=4)

    with open(os.path.join(model_save_dir, "full_training_history.json"), 'w') as f:
        json.dump(training_history, f, indent=4)

    print("\n--- Training Complete ---")
    if val_loader:
        print(f"Best Validation AUROC: {best_val_auroc:.4f}")
    print(f"Model and logs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
