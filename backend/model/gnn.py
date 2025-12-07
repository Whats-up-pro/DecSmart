import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

class HiFiGAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, heads=4, dropout=0.2):
        super(HiFiGAT, self).__init__()
        self.dropout = dropout
        
        # Embedding Layer for Opcodes (Optional, if input is indices)
        # But currently preprocess returns a Bag-of-Words vector (float), so we use Linear projection
        self.feature_proj = nn.Linear(num_node_features, hidden_channels)
        
        # First GAT layer
        # Input: Projected features
        self.conv1 = GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        
        # Second GAT layer
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)

    def forward(self, x, edge_index, batch):
        # x: Node feature matrix [num_nodes, num_node_features]
        
        # 0. Feature Projection
        x = self.feature_proj(x)
        x = F.relu(x)

        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        # 2. Readout layer (Global Pooling)
        # Combine mean and max pooling for better representation
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = x_mean + x_max  # Or concatenate

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
