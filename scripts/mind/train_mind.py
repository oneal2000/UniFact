import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import random
import gc


from src.detection_methods.hallucination_detection.mind_wrapper import MindMLP

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seed set to {seed}")

class MindFeatureDataset(Dataset):
    def __init__(self, combined_features_list):
        self.data = combined_features_list
        self.labels = [item['label'] for item in self.data]
        self.num_class_0 = self.labels.count(0)
        self.num_class_1 = self.labels.count(1)
        print(f"Dataset created with {len(self.data)} samples. Class 0 (Fact): {self.num_class_0}, Class 1 (Hallu): {self.num_class_1}")
        if len(self.data) > 0:
             self.feature_dim = len(self.data[0]['hd'])
             print(f"Feature dimension: {self.feature_dim}")
        else:
             self.feature_dim = 0
             print("Warning: Dataset is empty.")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor(item['hd'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)
        return {"input": features, "label": label}

    def get_class_counts(self):
        return self.num_class_0, self.num_class_1

def load_and_prepare_data(feature_dir: str, split: str = "train") -> list:
    file1 = os.path.join(feature_dir, f"last_token_mean_{split}.json")
    file2 = os.path.join(feature_dir, f"last_mean_{split}.json")

    print(f"Loading features for split '{split}' from:")
    print(f"  - {file1}")
    print(f"  - {file2}")

    try:
        with open(file1, 'r', encoding='utf-8') as f: data1 = json.load(f)
        with open(file2, 'r', encoding='utf-8') as f: data2 = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Feature file not found for split '{split}'. {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from feature file for split '{split}'. {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred loading data for split '{split}': {e}")
        return []

    if not data1 or not data2:
        print(f"Warning: One or both feature files for split '{split}' are empty.")
        return []


    if len(data1) != len(data2):
        print(f"Warning: Mismatch in entry count for '{split}' ({len(data1)} vs {len(data2)}). Using minimum.")
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]

    combined_data = []
    skipped_original = 0
    processed_hallu = 0

    for item1, item2 in zip(data1, data2):
        right1 = item1.get("right")
        right2 = item2.get("right")
        hallu1_list = item1.get("hallu", [])
        hallu2_list = item2.get("hallu", [])

        if not isinstance(right1, list) or not right1 or \
           not isinstance(right2, list) or not right2:
            skipped_original += 1
            continue

        combined_right = right1 + right2
        combined_data.append({"hd": combined_right, "label": 0})

        if isinstance(hallu1_list, list) and isinstance(hallu2_list, list) and len(hallu1_list) == len(hallu2_list):
            for h1, h2 in zip(hallu1_list, hallu2_list):
                if isinstance(h1, list) and h1 and isinstance(h2, list) and h2:
                    combined_hallu = h1 + h2
                    combined_data.append({"hd": combined_hallu, "label": 1})
                    processed_hallu += 1

    if skipped_original > 0: print(f"Skipped {skipped_original} original items due to missing/invalid features.")
    print(f"Prepared {len(combined_data)} total samples for '{split}' ({processed_hallu} hallucinated samples added).")

    return combined_data

def evaluate_model(model: MindMLP, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device) -> dict:
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
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)

            logits = model(inputs)
            if torch.isnan(logits).any():
                 print(f"Warning: NaN detected in model logits during evaluation. Input stats: mean={inputs.mean().item()}, std={inputs.std().item()}")
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
            all_probs_class1.extend(probs[:, 1].cpu().numpy())

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

        except ValueError as e:
            print(f"Warning: Could not calculate AUC metrics: {e}")
        except Exception as e_auc:
            print(f"Warning: Unexpected error calculating AUC: {e_auc}")

    return metrics


def run_training(args: argparse.Namespace):
    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"Using device: {device}")

    print("Loading and preparing data...")
    train_data_combined = load_and_prepare_data(args.feature_dir, "train")
    valid_data_combined = load_and_prepare_data(args.feature_dir, "valid")
    test_data_combined = load_and_prepare_data(args.feature_dir, "test")

    if not train_data_combined and not valid_data_combined:
        print("Error: Both Training and Validation raw data failed to load. Exiting.")
        return
    if not test_data_combined:
         print("Error: Test data (used for validation checkpoints) failed to load. Exiting.")
         return

    rtrain_data = train_data_combined + valid_data_combined
    validation_data = test_data_combined
    print(f"Using {len(rtrain_data)} samples for training (train+valid), {len(validation_data)} for validation (test).")

    if not rtrain_data:
        print("Error: Combined training dataset (train+valid) is empty. Exiting.")
        return


    train_dataset = MindFeatureDataset(rtrain_data)
    valid_dataset = MindFeatureDataset(validation_data)

    if len(train_dataset) == 0 or len(valid_dataset) == 0:
         print("Error: Training or validation dataset is empty after initialization. Exiting.")
         return

    input_size = train_dataset.feature_dim
    if input_size <= 0:
        print(f"Error: Determined MLP input size is invalid ({input_size}). Check feature files.")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory='cuda' in device.type)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory='cuda' in device.type)

    model = MindMLP(input_size=input_size, dropout_rate=args.dropout).to(device)

    c0, c1 = train_dataset.get_class_counts()
    if c0 > 0 and c1 > 0:
        total = c0 + c1
        weight_c0 = total / (2 * c0)
        weight_c1 = total / (2 * c1)
        class_weights = torch.tensor([weight_c0, weight_c1], dtype=torch.float32).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted loss. Weights [0, 1]: {class_weights.cpu().numpy()}")
    else:
        print("Warning: Training data has only one class or is empty. Using unweighted loss.")
        loss_fn = nn.CrossEntropyLoss()


    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_valid_metric = -1.0
    metric_to_optimize = "accuracy"
    best_epoch = -1
    output_model_dir = os.path.join(args.output_classifier_dir)
    os.makedirs(output_model_dir, exist_ok=True)
    print(f"\n--- Starting Training for {args.epochs} Epochs ---")
    print(f"Optimizing for validation {metric_to_optimize}")
    print(f"MLP Classifier will be saved to: {output_model_dir}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_epoch = 0.0
        train_preds_epoch = []
        train_labels_epoch = []
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False, ncols=100)

        for batch in train_iter:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(inputs)
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

        if not train_labels_epoch:
             print(f"Epoch {epoch} [Train] Warning: No valid batches processed.")
             avg_train_loss = np.nan
             train_accuracy = np.nan
        else:
             avg_train_loss = train_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0.0
             train_accuracy = accuracy_score(train_labels_epoch, train_preds_epoch)
        print(f"Epoch {epoch} [Train] Avg Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        # --- Validation ---
        valid_metrics = evaluate_model(model, valid_loader, loss_fn, device)
        valid_loss = valid_metrics["loss"]
        valid_accuracy = valid_metrics["accuracy"]
        valid_auroc = valid_metrics["auroc"]

        print(f"Epoch {epoch} [Valid] Loss: {valid_loss:.4f}, Acc: {valid_accuracy:.4f}", end="")
        if valid_auroc is not None: print(f", AUROC: {valid_auroc:.4f}", end="")
        print()

        # --- Save Best Model based on chosen metric ---
        current_metric_value = valid_metrics.get(metric_to_optimize)
        if current_metric_value is not None and current_metric_value > best_valid_metric:
            best_valid_metric = current_metric_value
            best_epoch = epoch
            save_path = os.path.join(output_model_dir, "mind_classifier_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                f'best_valid_{metric_to_optimize}': best_valid_metric,
                'valid_loss': valid_loss,
                'input_size': input_size,
            }, save_path)
            print(f"  >> New best validation {metric_to_optimize}: {best_valid_metric:.4f}! Saved model to {save_path}")

        # Save last model checkpoint
        last_save_path = os.path.join(output_model_dir, "mind_classifier_last.pt")
        torch.save({
             'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'valid_accuracy': valid_accuracy,
             'valid_loss': valid_loss,
             'valid_auroc': valid_auroc,
             'input_size': input_size,
         }, last_save_path)

        gc.collect()

    print(f"\n--- Training Finished ---")
    print(f"Best Validation {metric_to_optimize}: {best_valid_metric:.4f} at Epoch {best_epoch}")
    print(f"Best model saved to: {os.path.join(output_model_dir, 'mind_classifier_best.pt')}")
    print(f"Last model saved to: {os.path.join(output_model_dir, 'mind_classifier_last.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIND MLP Classifier from Extracted Features")
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory containing the feature files (last_token_mean_*.json, last_mean_*.json)")
    parser.add_argument("--output_classifier_dir", type=str, required=True,
                        help="Directory where the trained MLP classifier (.pt file) will be saved.")
    # Training Hyperparameters
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--wd", default=1e-5, type=float, help="Weight decay")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate for MLP")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str,
                        help="Device for training (e.g., cuda:0, cpu)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers for DataLoader")

    args = parser.parse_args()
    run_training(args)