"""
train.py
--------
Command-line interface for single-model training with:
- Mandatory CUDA GPU enforcement
- Dynamic VRAM allocation and OOM retry mechanism (especially for EfficientNet-B7)
- Mixed Precision (AMP) acceleration
- Comprehensive logging and result persistence (loss curves, accuracy curves, confusion matrix, metrics JSON)
"""

import argparse
from pathlib import Path
import sys
import time
import torch
import torch.nn as nn

from data_setup import create_dataloaders, resolve_data_paths
from engine import train
from models import get_model
from utils import (
    count_parameters,
    plot_accuracy_curves,
    plot_confusion_matrix,
    plot_loss_curves,
    save_metrics,
    save_model,
)


def run_training(
    model_name: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    data_dir: str = "data",
    results_dir: str = "results",
    use_amp: bool = True,
    seed: int = 42,
    num_workers: int = 2
) -> dict:
    """
    Main training routine for a specified model architecture.
    """
    # 1. Strict GPU Requirement Check
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is strictly required for this project but is NOT available. "
            "Please ensure NVIDIA GPU drivers and CUDA PyTorch are properly installed."
        )

    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    print(f"\n{'='*70}")
    print(f"Starting Training: {model_name.upper()}")
    print(f"Target Device: {device_name} | Total VRAM: {total_vram_gb:.2f} GB")
    print(f"Hyperparameters: Epochs={epochs}, LR={lr}, Mixed Precision={use_amp}")
    print(f"{'='*70}")

    # 2. VRAM-aware Batch Size adjustment
    current_batch_size = batch_size
    if "b7" in model_name.lower():
        if total_vram_gb < 12.0 and current_batch_size > 8:
            print(f"[VRAM Info] Total VRAM ({total_vram_gb:.2f} GB) < 12 GB. "
                  f"Automatically adjusting batch_size from {current_batch_size} to 8 for EfficientNet-B7.")
            current_batch_size = 8

    # 3. Path & Data Resolution
    train_path, val_path, labels_file = resolve_data_paths(data_dir)
    model_results_dir = Path(results_dir) / model_name.lower()
    model_results_dir.mkdir(parents=True, exist_ok=True)

    max_retries = 2
    attempt = 0

    while attempt <= max_retries:
        try:
            torch.cuda.empty_cache()
            print(f"\n[Attempt {attempt + 1}/{max_retries + 1}] Initializing model and DataLoader (Batch Size: {current_batch_size})...")
            
            # Model & Transforms
            model, transform = get_model(name=model_name, num_classes=10, seed=seed)
            model = model.to(device)

            train_loader, val_loader, class_names = create_dataloaders(
                train_dir=train_path,
                val_dir=val_path,
                transform=transform,
                batch_size=current_batch_size,
                num_workers=num_workers,
                labels_file=labels_file,
                pin_memory=True
            )

            total_params, trainable_params = count_parameters(model)
            frozen_params = total_params - trainable_params
            print(f"Parameters: Total={total_params:,} | Trainable={trainable_params:,} | Frozen={frozen_params:,}")

            # Loss & Optimizer
            loss_fn = nn.CrossEntropyLoss()
            # Only optimize trainable parameters
            trainable_layer_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params=trainable_layer_params, lr=lr)

            # Training Execution
            results, final_preds, final_targets, total_time = train(
                model=model,
                train_dataloader=train_loader,
                test_dataloader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=epochs,
                device=device,
                use_amp=use_amp,
                verbose=True
            )

            # Metrics
            final_train_loss = results["train_loss"][-1]
            final_train_acc = results["train_acc"][-1]
            final_test_loss = results["test_loss"][-1]
            final_test_acc = results["test_acc"][-1]

            metrics = {
                "model": model_name,
                "status": "completed",
                "total_params": total_params,
                "trainable_params": trainable_params,
                "frozen_params": frozen_params,
                "training_time_seconds": round(total_time, 2),
                "final_train_loss": round(final_train_loss, 4),
                "final_train_accuracy": round(final_train_acc * 100, 2),
                "final_test_loss": round(final_test_loss, 4),
                "final_test_accuracy": round(final_test_acc * 100, 2),
                "epochs": epochs,
                "batch_size": current_batch_size,
                "learning_rate": lr,
                "device": device_name,
                "history": results
            }

            # Save Visualizations and Artifacts
            print("\nSaving plots and metrics...")
            save_metrics(metrics, model_results_dir / "metrics.json")
            plot_loss_curves(results, model_results_dir / "loss_curve.png", title=f"{model_name} - Loss Curve")
            plot_accuracy_curves(results, model_results_dir / "accuracy_curve.png", title=f"{model_name} - Accuracy Curve")
            plot_confusion_matrix(final_targets, final_preds, class_names, model_results_dir / "confusion_matrix.png", title=f"{model_name} - Confusion Matrix")
            save_model(model, model_results_dir, f"{model_name}.pth")

            print(f"[SUCCESS] Training finished successfully in {total_time:.2f}s!")
            print(f"[SUCCESS] Final Test Accuracy: {final_test_acc * 100:.2f}% | Final Test Loss: {final_test_loss:.4f}")
            print(f"[SUCCESS] Artifacts saved to: {model_results_dir}\n")

            return metrics

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"\n[CUDA OOM Error]: {e}")
                attempt += 1
                torch.cuda.empty_cache()
                if attempt <= max_retries:
                    current_batch_size = max(1, current_batch_size // 2)
                    print(f"Retrying with halved batch size: {current_batch_size}...")
                    time.sleep(2)
                else:
                    print(f"\n[FAILED]: Model {model_name} exceeded available VRAM after {max_retries} retries.")
                    failure_metrics = {
                        "model": model_name,
                        "status": "failed",
                        "error": "VRAM insufficient (CUDA OOM)",
                        "batch_size": current_batch_size,
                        "epochs": epochs
                    }
                    save_metrics(failure_metrics, model_results_dir / "metrics.json")
                    return failure_metrics
            else:
                raise e


def parse_args():
    parser = argparse.ArgumentParser(description="Transfer Learning & Baseline Model Trainer")
    parser.add_argument("--model", type=str, required=True,
                        choices=["tinyvgg", "resnet18", "efficientnet_b0", "efficientnet_b7", "mobilenet_v3_small"],
                        help="Target architecture name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        use_amp=not args.no_amp,
        seed=args.seed,
        num_workers=args.num_workers
    )