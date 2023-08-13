from data import datasets
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Loading the Data
data_df = pd.read_csv("/home/kambhamettu.s/assignments/ml_project/data/processed_forest_cover.csv")
train_data, test_data = datasets(data_df)

# Loading train and test splits
X_train = train_data[0]
y_train = train_data[1]

X_test = test_data[0]
y_test = test_data[1]

# Define the model parameters
K = 5

# Defining the KNN
model = KNeighborsClassifier(n_neighbors=K)

# Fitting the KNN model
model.fit(X_train, y_train)

# Metrics
train_accuracy=model.score(X_train,y_train)
y_train_pred = model.predict(X_train)
print("Train Accuracy : {}".format(train_accuracy))
print(f"Train f1-score: {f1_score(y_train, y_train_pred, average='macro')}")


test_accuracy=model.score(X_test,y_test)
y_test_pred = model.predict(X_test)
print("Test Accuracy : {}".format(test_accuracy))
print(f"Test f1-score: {f1_score(y_test, y_test_pred, average='macro')}")

