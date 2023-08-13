import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

# Logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        out = torch.sigmoid(out)
        return out

# Feed Forward Neural Network (MLP)
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, nodes_per_layer, output_size):
        super(FeedForwardNN, self).__init__()
        self.input_layer = nn.Linear(input_size, nodes_per_layer[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(nodes_per_layer)-1):
            layer = nn.Linear(nodes_per_layer[i], nodes_per_layer[i+1])
            self.hidden_layers.append(layer)
        self.output_layer = nn.Linear(nodes_per_layer[-1], output_size)

    def forward(self, x):
        out = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            out = F.relu(layer(out))
        out = self.output_layer(out)
        return out
