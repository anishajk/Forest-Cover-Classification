from models import FeedForwardNN
from data import dataloaders
from train import train_model
from performance import *

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

data = pd.read_csv("/home/kambhamettu.s/assignments/ml_project/data/processed_forest_cover.csv")
train_data = data.sample(frac=0.7,random_state=200)
val_ = data.drop(train_data.index)
val_data = val_.sample(frac = 0.7, random_state=200)
test_data = val_.drop(val_data.index)

print("train: ", train_data.shape)
print("test: ", test_data.shape)
print("vald: ", val_data.shape)

# Defining Hyperparameters
LEARNING_RATE = 0.001
MOMENTUM = 0.9
BATCH_SIZE = 128
NUM_EPOCHS = 100
NUM_CLASSES = int(train_data['Cover_Type'].nunique())
INPUT_SIZE = 44
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nodes_per_layer = [256, 128]
output_size = NUM_CLASSES

# Creating Train, Validation & Test Dataloaders
train_loader, validation_loader, test_loader = dataloaders(train_data, val_data, test_data, BATCH_SIZE)

# Defining Loss
criterion = nn.CrossEntropyLoss()

# Definig the model
model = FeedForwardNN(INPUT_SIZE, nodes_per_layer, output_size)

# Definig the optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Training
best_model, losses, accuracies, f1_scores = train_model(model, train_loader, validation_loader, 
                                                                        criterion, optimizer, NUM_EPOCHS, DEVICE)

# Getting Performance
perf_df = performance(model, best_model, NUM_CLASSES, losses, accuracies, f1_scores, 'MLP')
perf_df.to_csv("/home/kambhamettu.s/assignments/ml_project/results/performance/MLP.csv", index=False)