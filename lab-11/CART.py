import pandas as pd
import numpy as np

# Define the dataset
data = pd.read_csv("data_dt_CART.csv")

# Define the DecisionTree class
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.argmax(np.bincount(y))

        num_samples, num_features = X.shape
        best_gini = 1.0
        best_feature = None
        best_split = None

        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                left_indices = np.where(X[:, feature] <= value)[0]
                right_indices = np.where(X[:, feature] > value)[0]

                left_gini = self._calculate_gini(y[left_indices])
                right_gini = self._calculate_gini(y[right_indices])
                gini = (len(left_indices) / num_samples) * left_gini + (len(right_indices) / num_samples) * right_gini

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_split = value

        if best_gini == 1.0:
            return np.argmax(np.bincount(y))

        left_indices = np.where(X[:, best_feature] <= best_split)[0]
        right_indices = np.where(X[:, best_feature] > best_split)[0]

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_split, left_tree, right_tree)

    def _calculate_gini(self, y):
        if len(y) == 0:
            return 0
        p0 = np.sum(y == 0) / len(y)
        p1 = np.sum(y == 1) / len(y)
        return 1 - p0**2 - p1**2

    def predict(self, X):
        predictions = []
        for x in X:
            node = self.tree
            while isinstance(node, tuple):
                feature, split, left, right = node
                if x[feature] <= split:
                    node = left
                else:
                    node = right
            predictions.append(node)
        return np.array(predictions)

# Prepare data using only "Temperature" and "Humidity"
X = data[["Temperature", "Humidity"]].values
y = (data["Decision"] == "Yes").astype(int).values

# Build the CART decision tree
tree = DecisionTree(max_depth=4)
tree.fit(X, y)

samples = [
    (75, 80),  # Temperature: 75, Humidity: 80
    (70, 70),  # Temperature: 70, Humidity: 70
    (83, 78),  # Temperature: 83, Humidity: 78
    (68, 80),  # Temperature: 68, Humidity: 80
    (81, 75),  # Temperature: 81, Humidity: 75
]

for sample in samples:
    prediction = tree.predict(np.array(sample).reshape(1, -1))
    if prediction[0] == 1:
        result = "Yes"
    else:
        result = "No"
    print(f"Prediction for the sample {sample}: {result}")