from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import numpy as np

# Custom pytorch Dataset class
class ForestDataset(Dataset):
    """
    Custom dataset class to load the data and apply transforms.
    """
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        x = torch.tensor(sample[:-1], dtype=torch.float32)
        y = torch.tensor(int(sample[-1]), dtype=torch.long)
        return x, y

def dataloaders(train, validation, test, batch_size):
    # Modifying the range of the df
    shift_values = lambda x: x - 1
    train['Cover_Type'] = train['Cover_Type'].map(shift_values)
    validation['Cover_Type'] = validation['Cover_Type'].map(shift_values)
    test['Cover_Type'] = test['Cover_Type'].map(shift_values)

    train_dataset = ForestDataset(train)
    validation_dataset = ForestDataset(validation)
    test_dataset = ForestDataset(test)

    # Creating Train, Validation & Test Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def datasets(data):
    X = data.drop('Cover_Type', axis=1)
    y = data['Cover_Type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    train_data = [X_train, y_train]
    # val_data = [X_val, y_val]
    test_data = [X_test, y_test]

    return train_data, test_data



