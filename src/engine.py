"""
engine.py
---------
Core training and evaluation engine supporting:
- Mixed precision training (AMP + GradScaler)
- Robust per-epoch metrics calculation
- Inference mode evaluation with prediction tracking for confusion matrices
"""

import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None
) -> Tuple[float, float]:
    """
    Performs a single training epoch step.
    
    Args:
        model: Target PyTorch model.
        dataloader: Training DataLoader.
        loss_fn: Loss criterion.
        optimizer: PyTorch optimizer.
        device: Target compute device ('cuda' or 'cpu').
        use_amp: Whether to use Automatic Mixed Precision (AMP).
        scaler: GradScaler instance for AMP.
        
    Returns:
        Tuple of (average_train_loss, average_train_accuracy).
    """
    model.train()
    train_loss, train_acc = 0.0, 0.0
    total_samples = 0
    
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        batch_size = X.size(0)
        total_samples += batch_size
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
        train_loss += loss.item() * batch_size
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()
        
    avg_loss = train_loss / total_samples
    avg_acc = train_acc / total_samples
    return avg_loss, avg_acc


def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float, List[int], List[int]]:
    """
    Performs a single validation/test epoch step.
    
    Args:
        model: Target PyTorch model.
        dataloader: Validation DataLoader.
        loss_fn: Loss criterion.
        device: Target compute device ('cuda' or 'cpu').
        
    Returns:
        Tuple of (average_test_loss, average_test_accuracy, all_predictions, all_targets).
    """
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            batch_size = X.size(0)
            total_samples += batch_size
            
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item() * batch_size
            
            test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
            test_acc += (test_pred_labels == y).sum().item()
            
            all_preds.extend(test_pred_labels.cpu().tolist())
            all_targets.extend(y.cpu().tolist())
            
    avg_loss = test_loss / total_samples
    avg_acc = test_acc / total_samples
    return avg_loss, avg_acc, all_preds, all_targets


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    use_amp: bool = False,
    verbose: bool = True
) -> Tuple[Dict[str, List[float]], List[int], List[int], float]:
    """
    Trains and evaluates a PyTorch model across given epochs.
    
    Args:
        model: Target PyTorch model.
        train_dataloader: Training DataLoader.
        test_dataloader: Validation DataLoader.
        optimizer: Optimizer.
        loss_fn: Loss function.
        epochs: Number of training epochs.
        device: Compute device ('cuda' or 'cpu').
        use_amp: Whether to use AMP.
        verbose: If True, prints per-epoch progress.
        
    Returns:
        Tuple of (results_dict, final_test_preds, final_test_targets, total_training_time_seconds).
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None
    
    start_time = time.time()
    final_preds: List[int] = []
    final_targets: List[int] = []
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler
        )
        
        test_loss, test_acc, preds, targets = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        
        epoch_time = time.time() - epoch_start
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        final_preds = preds
        final_targets = targets
        
        if verbose:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% | "
                f"Time: {epoch_time:.2f}s"
            )
            
    total_time = time.time() - start_time
    return results, final_preds, final_targets, total_time