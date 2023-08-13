from models import LogisticRegression
from data import dataloaders
from train import train_model
from performance import performance
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

# Defining Hyperparameters
LEARNING_RATE = 0.002
MOMENTUM = 0.9
BATCH_SIZE = 128
NUM_EPOCHS = 50
NUM_CLASSES = int(train_data['Cover_Type'].nunique())
INPUT_SIZE = 44
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Creating Train, Validation & Test Dataloaders
train_loader, validation_loader, test_loader = dataloaders(train_data, val_data, test_data, BATCH_SIZE)

# Defining Loss
criterion = nn.CrossEntropyLoss()

# Definig the model
model = LogisticRegression(INPUT_SIZE, NUM_CLASSES)

# Definig the optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# Training
best_model, losses, accuracies, f1_scores = train_model(model, train_loader, validation_loader, 
                                                                        criterion, optimizer, NUM_EPOCHS, DEVICE)

# Getting Performance
perf_df = performance(model, best_model, NUM_CLASSES, losses, accuracies, f1_scores, 'LR')
perf_df.to_csv("/home/kambhamettu.s/assignments/ml_project/results/performance/LR.csv", index=False)