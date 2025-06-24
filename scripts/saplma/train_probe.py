import os
import sys
import argparse
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import random
import gc

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.detection_methods.hallucination_detection.saplma_wrapper import SaplmaProbeMLP

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seed set to {seed}")

class SaplmaEmbeddingDataset(Dataset):
    def __init__(self, embedding_data_list):
        self.data = embedding_data_list
        self.labels = [item['label'] for item in self.data]
        self.num_class_0 = self.labels.count(0)
        self.num_class_1 = self.labels.count(1)
        print(f"Dataset created with {len(self.data)} samples. Class 0: {self.num_class_0}, Class 1: {self.num_class_1}")

        if len(self.data) > 0:
            first_embedding = self.data[0].get('embedding')
            if first_embedding is not None:
                self.feature_dim = len(first_embedding)
                print(f"Determined Embedding dimension: {self.feature_dim}")
            else:
                 raise ValueError("Could not determine embedding dimension from the first data sample.")
        else:
            self.feature_dim = 0
            print("Warning: Dataset is empty.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        embedding = torch.tensor(item['embedding'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)
        return {"embedding": embedding, "label": label}

    def get_class_counts(self):
        return self.num_class_0, self.num_class_1

def load_saplma_embeddings(embedding_dir_path: str) -> list:
    all_embeddings_data = []
    json_files = glob.glob(os.path.join(embedding_dir_path, '*.json'))

    if not json_files:
        raise FileNotFoundError(f"No .json files found in directory: {embedding_dir_path}")

    print(f"Found {len(json_files)} JSON embedding files to load.")
    for file_path in tqdm(json_files, desc="Loading JSON files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Basic validation for the first item if list is not empty
                    if data and ('embedding' not in data[0] or 'label' not in data[0]):
                         print(f"Warning: Skipping file {os.path.basename(file_path)} - first item missing 'embedding' or 'label' key.")
                         continue
                    all_embeddings_data.extend(data)
                else:
                    print(f"Warning: Skipping file {os.path.basename(file_path)} - expected a list of objects, got {type(data)}.")
        except json.JSONDecodeError:
            print(f"Warning: Skipping file {os.path.basename(file_path)} due to JSON decoding error.")
        except Exception as e:
            print(f"Warning: Skipping file {os.path.basename(file_path)} due to error: {e}")

    print(f"Loaded a total of {len(all_embeddings_data)} embedding entries.")
    if not all_embeddings_data:
        raise ValueError("No valid embedding data loaded from the specified directory.")
    return all_embeddings_data

def evaluate_probe(model: SaplmaProbeMLP, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs_class1 = []

    metrics = {
        "loss": None, "accuracy": None, "auroc": None, "support": 0
    }

    if len(dataloader.dataset) == 0:
        print("Warning: Empty dataset for evaluation.")
        return metrics

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False, ncols=80):
            embeddings = batch["embedding"].to(device)
            labels = batch["label"].to(device)

            logits = model(embeddings)
            if torch.isnan(logits).any():
                print(f"Warning: NaN detected in model logits during evaluation. Skipping batch.")
                continue

            loss = loss_fn(logits, labels)
            if not torch.isnan(loss):
                total_loss += loss.item()
            else:
                print("Warning: NaN detected in loss during evaluation.")

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs_class1.extend(probs[:, 1].cpu().numpy()) # Prob of class 1

    if not all_labels:
        print("Warning: No valid batches processed during evaluation.")
        return metrics

    metrics["support"] = len(all_labels)
    metrics["loss"] = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    metrics["accuracy"] = accuracy_score(all_labels, all_preds)

    unique_labels = np.unique(all_labels)
    if len(unique_labels) > 1:
        try:
            valid_indices = [i for i, p in enumerate(all_probs_class1) if not np.isnan(p)]
            if len(valid_indices) < len(all_labels):
                print(f"Warning: Found {len(all_labels) - len(valid_indices)} NaN probabilities during AUC calculation. Filtering.")
            filtered_labels = np.array(all_labels)[valid_indices]
            filtered_probs = np.array(all_probs_class1)[valid_indices]

            if len(np.unique(filtered_labels)) > 1 and len(filtered_probs) > 0:
                metrics["auroc"] = roc_auc_score(filtered_labels, filtered_probs)
            else:
                print("Warning: Not enough valid samples or only one class after filtering NaNs for AUC calculation.")
                metrics["auroc"] = None

        except ValueError as e:
            print(f"Warning: Could not calculate AUC metrics: {e}")
            metrics["auroc"] = None
        except Exception as e_auc:
            print(f"Warning: Unexpected error calculating AUC: {e_auc}")
            metrics["auroc"] = None
    else:
        print(f"Warning: Only one class ({unique_labels}) present. AUC metrics undefined.")
        metrics["auroc"] = None

    return metrics


def run_probe_training(args: argparse.Namespace):
    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"Using device: {device}")

    print("Loading and preparing SAPLMA embedding data...")
    try:
        all_data = load_saplma_embeddings(args.embedding_dir_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    if args.validation_split_ratio <= 0 or args.validation_split_ratio >= 1:
         print("Warning: Invalid validation_split_ratio. Using 0.1 (10%) for validation.")
         val_ratio = 0.1
    else:
         val_ratio = args.validation_split_ratio

    try:
        train_data, valid_data = train_test_split(
            all_data,
            test_size=val_ratio,
            random_state=args.seed,
            stratify=[d['label'] for d in all_data] # Stratify to keep class balance
        )
        print(f"Data split: {len(train_data)} training samples, {len(valid_data)} validation samples.")
    except ValueError as e:
         print(f"Error during train/validation split (possibly too few samples for stratification?): {e}")
         print("Attempting split without stratification.")
         try:
              train_data, valid_data = train_test_split(
                   all_data, test_size=val_ratio, random_state=args.seed
              )
              print(f"Data split (unstratified): {len(train_data)} training samples, {len(valid_data)} validation samples.")
         except Exception as split_e:
              print(f"FATAL: Could not split data: {split_e}")
              return

    try:
        train_dataset = SaplmaEmbeddingDataset(train_data)
        valid_dataset = SaplmaEmbeddingDataset(valid_data)
    except ValueError as e:
         print(f"FATAL: Error creating dataset: {e}")
         return

    if len(train_dataset) == 0 or train_dataset.feature_dim <= 0:
        print("Error: Training dataset is empty or feature dimension is invalid. Exiting.")
        return
    if len(valid_dataset) == 0:
         print("Warning: Validation dataset is empty.")

    input_size = train_dataset.feature_dim

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory='cuda' in device.type)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory='cuda' in device.type) if len(valid_dataset) > 0 else None

    model = SaplmaProbeMLP(input_size=input_size, dropout_rate=args.dropout).to(device)

    c0, c1 = train_dataset.get_class_counts()
    if args.use_class_weights and c0 > 0 and c1 > 0:
        total = c0 + c1
        weight_c0 = total / (2 * c0)
        weight_c1 = total / (2 * c1)
        class_weights = torch.tensor([weight_c0, weight_c1], dtype=torch.float32).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted loss. Weights [Class 0, Class 1]: {class_weights.cpu().numpy()}")
    else:
        if args.use_class_weights: print("Warning: Cannot use class weights (one class missing or count is zero). Using unweighted loss.")
        loss_fn = nn.CrossEntropyLoss()
        print("Using unweighted CrossEntropyLoss.")


    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_valid_metric_value = -1.0
    metric_to_optimize = "accuracy"
    best_epoch = -1
    output_probe_dir = os.path.dirname(args.output_probe_path)
    if output_probe_dir: os.makedirs(output_probe_dir, exist_ok=True)

    print(f"\n--- Starting SAPLMA Probe Training for {args.epochs} Epochs ---")
    print(f"Optimizing for validation {metric_to_optimize}")
    print(f"Probe will be saved to: {args.output_probe_path}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_epoch = 0.0
        train_preds_epoch = []
        train_labels_epoch = []
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False, ncols=100)

        for batch in train_iter:
            embeddings = batch["embedding"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(embeddings)
            loss = loss_fn(logits, labels)

            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered in epoch {epoch}, step {train_iter.n}. Skipping backward pass for this batch.")
                continue

            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_preds_epoch.extend(preds.cpu().numpy())
            train_labels_epoch.extend(labels.cpu().numpy())
            train_iter.set_postfix(loss=f"{train_loss_epoch / (train_iter.n + 1):.4f}")

        avg_train_loss = train_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0.0
        train_accuracy = accuracy_score(train_labels_epoch, train_preds_epoch) if train_labels_epoch else 0.0
        print(f"Epoch {epoch} [Train] Avg Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        valid_accuracy = -1.0
        if valid_loader:
            valid_metrics = evaluate_probe(model, valid_loader, loss_fn, device)
            valid_loss = valid_metrics["loss"]
            valid_accuracy = valid_metrics["accuracy"]
            valid_auroc = valid_metrics["auroc"]

            print(f"Epoch {epoch} [Valid] Loss: {valid_loss:.4f}, Acc: {valid_accuracy:.4f}", end="")
            if valid_auroc is not None: print(f", AUROC: {valid_auroc:.4f}", end="")
            print()

            current_metric_value = valid_metrics.get(metric_to_optimize)
            if current_metric_value is not None and current_metric_value > best_valid_metric_value:
                best_valid_metric_value = current_metric_value
                best_epoch = epoch
                print(f"  >> New best validation {metric_to_optimize}: {best_valid_metric_value:.4f}! Saving model...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    f'best_valid_{metric_to_optimize}': best_valid_metric_value,
                    'input_size': input_size,
                    'dropout_rate': args.dropout,
                }, args.output_probe_path)
        else:
             if epoch == args.epochs:
                 print("No validation set. Saving model from last epoch.")
                 torch.save({
                     'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'input_size': input_size,
                     'dropout_rate': args.dropout,
                 }, args.output_probe_path)


        gc.collect()

    print(f"\n--- SAPLMA Probe Training Finished ---")
    if best_epoch != -1:
         print(f"Best Validation {metric_to_optimize}: {best_valid_metric_value:.4f} at Epoch {best_epoch}")
         print(f"Best probe saved to: {args.output_probe_path}")
    elif not valid_loader:
         print(f"Probe from last epoch saved to: {args.output_probe_path}")
    else:
         print("No best model saved (validation metric might not have improved or validation failed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAPLMA Probe MLP Classifier from Generated Embeddings")
    parser.add_argument("--embedding_dir_path", type=str, required=True,
                        help="Directory containing the generated JSON embedding files (output from generate_saplma_embeddings.py).")
    parser.add_argument("--output_probe_path", type=str, required=True,
                        help="Path where the trained PyTorch probe model (.pt file) will be saved.")
    parser.add_argument("--validation_split_ratio", type=float, default=0.1,
                        help="Fraction of data to use for validation (e.g., 0.1 for 10%).")
    # Training Hyperparameters
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--wd", default=0.01, type=float, help="Weight decay")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate for MLP")
    parser.add_argument("--use_class_weights", action='store_true', help="Use inverse frequency class weighting for loss.")
    parser.add_argument("--device", default="auto", type=str, help="Device for training ('cuda', 'cpu', 'auto')")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", default=0, type=int, help="Number of workers for DataLoader (set > 0 for multiprocessing)")

    args = parser.parse_args()
    run_probe_training(args)