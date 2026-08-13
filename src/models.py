"""
models.py
---------
Model factory providing 5 architectures for transfer learning and baseline comparison:
1. TinyVGG (Trained from scratch - baseline)
2. ResNet18 (Torchvision Pretrained, feature extractor frozen)
3. EfficientNet-B0 (Torchvision Pretrained, feature extractor frozen)
4. EfficientNet-B7 (Torchvision Pretrained, feature extractor frozen)
5. MobileNetV3-Small (Torchvision Pretrained, feature extractor frozen)

Each model comes paired with its official/recommended torchvision transform pipeline.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models, transforms


class TinyVGG(nn.Module):
    """
    TinyVGG Architecture replicating the baseline convolutional neural network
    for 64x64 3-channel RGB image classification.
    """
    def __init__(self, in_channels: int = 3, hidden_units: int = 10, out_shape: int = 10) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16, out_features=out_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


def get_model(
    name: str,
    num_classes: int = 10,
    seed: int = 42
) -> Tuple[nn.Module, transforms.Compose]:
    """
    Factory function returning the instantiated model and its corresponding transform.
    
    Args:
        name: Name of architecture ('tinyvgg', 'resnet18', 'efficientnet_b0', 'efficientnet_b7', 'mobilenet_v3_small').
        num_classes: Number of output classes.
        seed: Random seed for reproducibility of new classifier layer weights.
        
    Returns:
        Tuple of (model, transform)
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    model_name_clean = name.lower().strip()

    if model_name_clean == "tinyvgg":
        model = TinyVGG(in_channels=3, hidden_units=16, out_shape=num_classes)
        # Custom 64x64 transform with normalization
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return model, transform

    elif model_name_clean in ["resnet18", "resnet_18"]:
        weights = models.ResNet18_Weights.DEFAULT
        transform = weights.transforms()
        model = models.resnet18(weights=weights)
        
        # Freeze all backbone layers
        for param in model.parameters():
            param.requires_grad = False
            
        # Recreate classifier head
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=num_classes)
        return model, transform

    elif model_name_clean in ["efficientnet_b0", "effnet_b0", "efficientnetb0"]:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        transform = weights.transforms()
        model = models.efficientnet_b0(weights=weights)
        
        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False
            
        # Recreate classifier head
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )
        return model, transform

    elif model_name_clean in ["efficientnet_b7", "effnet_b7", "efficientnetb7"]:
        weights = models.EfficientNet_B7_Weights.DEFAULT
        transform = weights.transforms()
        model = models.efficientnet_b7(weights=weights)
        
        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False
            
        # Recreate classifier head
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )
        return model, transform

    elif model_name_clean in ["mobilenet_v3_small", "mobilenetv3_small", "mobilenet"]:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        transform = weights.transforms()
        model = models.mobilenet_v3_small(weights=weights)
        
        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False
            
        # Recreate final classifier layer
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features=in_features, out_features=num_classes)
        return model, transform

    else:
        valid_models = ["tinyvgg", "resnet18", "efficientnet_b0", "efficientnet_b7", "mobilenet_v3_small"]
        raise ValueError(f"Unknown model name '{name}'. Supported architectures: {valid_models}")