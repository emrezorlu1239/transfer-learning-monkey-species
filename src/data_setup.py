"""
data_setup.py
-------------
Utilities for data loading, dataset discovery, label parsing, and DataLoader creation.
Supports custom transforms as well as model-specific torchvision transforms.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def parse_label_mapping(labels_file_path: Union[str, Path]) -> Dict[str, str]:
    """
    Parses monkey_labels.txt to map label folder names (e.g., 'n0') to human-readable common names.
    
    Args:
        labels_file_path: Path to monkey_labels.txt.
        
    Returns:
        Dictionary mapping folder label (e.g. 'n0') to common name (e.g. 'mantled_howler').
    """
    path = Path(labels_file_path)
    mapping = {}
    if not path.exists():
        return mapping
        
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith("Label"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            label_code = parts[0]
            common_name = parts[2].replace(" ", "_").strip()
            mapping[label_code] = common_name
            
    return mapping


def resolve_data_paths(data_dir: Union[str, Path] = "data") -> Tuple[Path, Path, Path]:
    """
    Resolves training and validation paths accounting for nested folder structures.
    
    Args:
        data_dir: Root data directory path.
        
    Returns:
        Tuple of (train_path, val_path, labels_file_path)
    """
    root = Path(data_dir)
    
    # Handle possible nested structures (e.g. data/training/training vs data/training)
    train_dir = root / "training" / "training"
    if not train_dir.exists():
        train_dir = root / "training"
        
    val_dir = root / "validation" / "validation"
    if not val_dir.exists():
        val_dir = root / "validation"
        
    labels_file = root / "monkey_labels.txt"
    return train_dir, val_dir, labels_file


def create_dataloaders(
    train_dir: Union[str, Path],
    val_dir: Union[str, Path],
    transform: transforms.Compose,
    batch_size: int = 32,
    num_workers: int = 2,
    labels_file: Optional[Union[str, Path]] = None,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Creates PyTorch DataLoaders for training and validation datasets.
    
    Args:
        train_dir: Path to training data directory.
        val_dir: Path to validation data directory.
        transform: Image transform pipeline (torchvision.transforms or model weights.transforms()).
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.
        labels_file: Optional path to monkey_labels.txt for mapping folder names to common names.
        pin_memory: If True, copies Tensors into CUDA pinned memory before returning them.
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, class_names)
    """
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    
    # Load datasets using ImageFolder
    train_data = datasets.ImageFolder(root=str(train_dir), transform=transform)
    val_data = datasets.ImageFolder(root=str(val_dir), transform=transform)
    
    raw_classes = train_data.classes  # ['n0', 'n1', ..., 'n9']
    
    # Resolve human readable class names if labels file is available
    if labels_file and Path(labels_file).exists():
        mapping = parse_label_mapping(labels_file)
        class_names = [mapping.get(c, c) for c in raw_classes]
    else:
        class_names = raw_classes
        
    # Create DataLoaders
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_dataloader, val_dataloader, class_names