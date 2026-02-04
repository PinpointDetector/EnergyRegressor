import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNProjectionNetwork(nn.Module):
    """CNN for processing a single 2D projection"""

    def __init__(self, conv_dims: Tuple=(8, 16, 32), feature_dim: int=128, kernel_size: int=3, padding=1, dropout: float=0.5):
        super(CNNProjectionNetwork, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i, dim in enumerate(conv_dims):
            in_ch = 1 if i == 0 else conv_dims[i - 1]
            self.conv_layers.append(
                nn.Conv2d(in_ch, dim, kernel_size=kernel_size, padding=padding)
            )

        self.batch_norms = nn.ModuleList(
        [nn.BatchNorm2d(dim) for dim in conv_dims]
        )

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout2d = nn.Dropout2d(dropout)
        self.dropout = nn.Dropout(dropout)

        # Feature extraction
        # self.feature_fc = nn.Linear(32 * 4 * 4, feature_dim)
        self.feature_fc = nn.Linear(conv_dims[-1] * 4 * 4, feature_dim)


    def forward(self, x):

        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            
            if i == len(self.conv_layers) - 1:
                # Use adaptive pooling for last layer
                x = self.adaptive_pool(F.relu(bn(conv(x))))
                x = self.dropout2d(x)
            else:
                x = self.pool(F.relu(bn(conv(x))))
                x = self.dropout2d(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.feature_fc(x))
        x = self.dropout(x)

        return x


class ClassifierProjectionCNN(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        conv_dims: Tuple=(8, 16, 32),
        fc_dims: Tuple=(128,),
        kernel_size: int=3, 
        padding=1, 
        dropout: float=0.5,
        num_classes: int = 3,
    ):
        super(ClassifierProjectionCNN,
         self).__init__()
        self.zx_cnn = CNNProjectionNetwork(conv_dims=conv_dims, feature_dim=feature_dim, kernel_size=kernel_size, padding=padding, dropout=dropout)
        self.zy_cnn = CNNProjectionNetwork(conv_dims=conv_dims, feature_dim=feature_dim, kernel_size=kernel_size, padding=padding, dropout=dropout)


        self.classifier = nn.Sequential()
        for i, dim in enumerate(fc_dims):
            if i == 0:
                self.classifier.add_module(f"fc_{i}", nn.Linear(feature_dim * 2, dim))
            else:
                self.classifier.add_module(f"fc_{i}", nn.Linear(fc_dims[i - 1], dim))
            self.classifier.add_module(f"relu_{i}", nn.ReLU())
            self.classifier.add_module(f"dropout_{i}", nn.Dropout(dropout))

        self.classifier.add_module("output", nn.Linear(fc_dims[-1], num_classes))

        
    def forward(self, x):
        zx_proj, zy_proj = x

        zx_features = self.zx_cnn(zx_proj)
        zy_features = self.zy_cnn(zy_proj)

        combined_features = torch.cat([zx_features, zy_features], dim=1)

        output = self.classifier(combined_features)

        return output


class RegressionCNN(nn.Module):
    def __init__(self, 
        feature_dim: int = 128,
        conv_dims: Tuple=(8, 16, 32),
        fc_dims: Tuple=(128,),
        kernel_size: int=3, 
        padding=1, 
        dropout: float=0.5,
        num_targets: int = 3,):

        super(RegressionCNN, self).__init__()
        self.zx_cnn = CNNProjectionNetwork(conv_dims=conv_dims, feature_dim=feature_dim, kernel_size=kernel_size, padding=padding, dropout=dropout)
        self.zy_cnn = CNNProjectionNetwork(conv_dims=conv_dims, feature_dim=feature_dim, kernel_size=kernel_size, padding=padding, dropout=dropout)


        self.regressor = nn.Sequential()
        for i, dim in enumerate(fc_dims):
            if i == 0:
                self.regressor.add_module(f"fc_{i}", nn.Linear(feature_dim * 2, dim))
            else:
                self.regressor.add_module(f"fc_{i}", nn.Linear(fc_dims[i - 1], dim))
            self.regressor.add_module(f"relu_{i}", nn.ReLU())
            self.regressor.add_module(f"dropout_{i}", nn.Dropout(dropout))

        self.regressor.add_module("output", nn.Linear(fc_dims[-1], num_targets))

    def forward(self, x):

        zx_proj, zy_proj = x

        zx_features = self.zx_cnn(zx_proj)
        zy_features = self.zy_cnn(zy_proj)

        combined_features = torch.cat([zx_features, zy_features], dim=1)
        
        output = self.regressor(combined_features)

        return output.squeeze(-1)  # Return shape (batch,) instead of (batch, 1) for single target regression
