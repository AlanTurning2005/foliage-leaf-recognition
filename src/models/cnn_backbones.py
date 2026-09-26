import torch
import torch.nn as nn
from torchvision import models


class CustomCNNBackbone(nn.Module):
    """
    CNN Backbone tùy biến theo kiến trúc bài báo Boi M. Quach et al.:
    5 lớp Conv2D (16 -> 16 -> 32 -> 32 -> 32) + BatchNorm + MaxPool
    Nén ảnh không gian về embedding 100 chiều.
    """
    def __init__(self, embedding_dim: int = 100):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 & 2: 16 filters 3x3
            nn.Conv2d(3, 16, kernel_size=3, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 16, kernel_size=3, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3, 4, 5: 32 filters 5x5
            nn.Conv2d(16, 32, kernel_size=5, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 32, kernel_size=5, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 32, kernel_size=5, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class LeafClassifier(nn.Module):
    """
    Module phân loại tổng thể:
    Kết hợp Backbone (ResNet hoặc Custom CNN) với Classification Head 60 classes.
    """
    def __init__(
        self, 
        backbone_name: str = "resnet18", 
        num_classes: int = 60, 
        pretrained: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            in_features = base_model.fc.in_features
            # Tách backbone (bỏ lớp fc cuối)
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            self.classifier = nn.Linear(in_features, num_classes)

        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            in_features = base_model.fc.in_features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            self.classifier = nn.Linear(in_features, num_classes)

        elif backbone_name == "custom_cnn":
            # Embedding 100 chiều
            self.backbone = CustomCNNBackbone(embedding_dim=100)
            self.classifier = nn.Linear(100, num_classes)
        else:
            raise ValueError(f"Không hỗ trợ backbone: {backbone_name}")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất vector đặc trưng (Embedding) phục vụ SVM hoặc phân tích t-SNE"""
        features = self.backbone(x)
        return torch.flatten(features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)

    