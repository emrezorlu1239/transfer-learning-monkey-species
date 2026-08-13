"""
utils.py
--------
Utility helper functions for:
- Parameter counting (total vs trainable)
- Checkpoint saving
- Metric persistence (JSON)
- Visualization (loss curves, accuracy curves, confusion matrix)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Counts the total and trainable parameters of a PyTorch model.
    
    Args:
        model: Target PyTorch model.
        
    Returns:
        Tuple of (total_parameters, trainable_parameters).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_model(model: nn.Module, target_dir: Union[str, Path], model_name: str) -> Path:
    """
    Saves PyTorch model state_dict to specified target directory.
    
    Args:
        model: Target model to save.
        target_dir: Target directory path.
        model_name: Filename (e.g. 'resnet18.pth').
        
    Returns:
        Path to saved model file.
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    if not model_name.endswith(".pth") and not model_name.endswith(".pt"):
        model_name += ".pth"
        
    model_save_path = target_dir_path / model_name
    torch.save(obj=model.state_dict(), f=model_save_path)
    return model_save_path


def save_metrics(metrics: Dict, save_path: Union[str, Path]) -> None:
    """
    Saves metrics dictionary to a formatted JSON file.
    
    Args:
        metrics: Dictionary containing training metrics and parameters.
        save_path: Destination path for JSON file.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def plot_loss_curves(
    results: Dict[str, List[float]],
    save_path: Union[str, Path],
    title: Optional[str] = "Loss Curve"
) -> None:
    """
    Plots training and validation loss curves and saves to image.
    
    Args:
        results: Dictionary containing 'train_loss' and 'test_loss'.
        save_path: Destination file path for plot.
        title: Plot title.
    """
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(epochs, train_loss, label="Train Loss", marker="o", linewidth=2, color="#e74c3c")
    plt.plot(epochs, test_loss, label="Validation Loss", marker="s", linewidth=2, color="#3498db")
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_accuracy_curves(
    results: Dict[str, List[float]],
    save_path: Union[str, Path],
    title: Optional[str] = "Accuracy Curve"
) -> None:
    """
    Plots training and validation accuracy curves and saves to image.
    
    Args:
        results: Dictionary containing 'train_acc' and 'test_acc'.
        save_path: Destination file path for plot.
        title: Plot title.
    """
    train_acc = [acc * 100 for acc in results["train_acc"]]
    test_acc = [acc * 100 for acc in results["test_acc"]]
    epochs = range(1, len(train_acc) + 1)
    
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(epochs, train_acc, label="Train Accuracy (%)", marker="o", linewidth=2, color="#2ecc71")
    plt.plot(epochs, test_acc, label="Validation Accuracy (%)", marker="s", linewidth=2, color="#9b59b6")
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim([0, 105])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    classes: List[str],
    save_path: Union[str, Path],
    title: Optional[str] = "Confusion Matrix"
) -> None:
    """
    Generates and saves a styled Confusion Matrix heatmap.
    
    Args:
        y_true: Ground truth target indices.
        y_pred: Model prediction indices.
        classes: List of class label names.
        save_path: Destination file path for heatmap.
        title: Title of confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        annot_kws={"size": 10, "weight": "bold"}
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=14)
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
    plt.ylabel("True Label", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()