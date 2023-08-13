from data import datasets
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Loading the Data
data_df = pd.read_csv("/home/kambhamettu.s/assignments/ml_project/data/processed_forest_cover.csv")
# train_data, val_data, test_data = datasets(data_df)
train_data, test_data = datasets(data_df)

# Loading train and test splits
X_train = train_data[0]
y_train = train_data[1]

# X_val = val_data[0]
# y_val = val_data[1]

X_test = test_data[0]
y_test = test_data[1]

# Define the model parameters
n_estimators = 100
max_depth = 10

# Define Random Forest Classifier
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

# Fitting the model
model.fit(X_train, y_train)

# Metrics
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f'Train Accuracy: {train_accuracy:.4f}')
print(f"Train f1-score: {f1_score(y_train, y_train_pred, average='macro')}")

# Test the model
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f"Test f1-score: {f1_score(y_test, y_test_pred, average='macro')}")