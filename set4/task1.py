import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from collections import Counter

# Set a random seed for reproducible results
np.random.seed(42)

# --- 1. Normalization Technique: Standard Scaler ---
class MyStandardScaler:
    """
    Transforms features by removing the mean and scaling to unit variance.
    """
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-9 # Add epsilon for stability

    def transform(self, X):
        return (X - self.mean_) / self.std_
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# --- 2. Classification Metric: Macro F1 Score ---
def calculate_macro_f1(y_true, y_pred, n_classes):
    """
    Calculates the Macro F1 score from scratch.
    """
    f1_scores = []
    epsilon = 1e-9 # To avoid division by zero

    # Calculate F1 score for each class
    for c in range(n_classes):
        # True Positives
        tp = np.sum((y_pred == c) & (y_true == c))
        # False Positives
        fp = np.sum((y_pred == c) & (y_true != c))
        # False Negatives
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        f1_scores.append(f1)

    # Macro F1 is the unweighted average
    return np.mean(f1_scores)

# --- 3. Model: Decision Tree Classifier ---

class Node:
    """
    Helper class to represent a single node in the Decision Tree.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature       # Index of feature to split on
        self.threshold = threshold   # Threshold value for the split
        self.left = left             # Left child node
        self.right = right           # Right child node
        self.value = value           # The class label (if it's a leaf node)

    def is_leaf_node(self):
        return self.value is not None

class MyDecisionTreeClassifier:
    """
    Implements a Decision Tree Classifier from scratch using Gini Impurity.
    """
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes_ = 0

    def _gini(self, y):
        """Calculates the Gini Impurity for a set of labels."""
        # Count occurrences of each class
        _, counts = np.unique(y, return_counts=True)
        # Calculate probabilities
        probabilities = counts / len(y)
        # Gini = 1 - sum(p_i^2)
        return 1.0 - np.sum(probabilities**2)

    def _best_split(self, X, y):
        """Finds the best feature and threshold to split the data."""
        n_samples, n_features = X.shape
        best_gini = 1.0
        best_split = (None, None) # (feature_idx, threshold)
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for thr in thresholds:
                # Split the data
                left_indices = np.where(X[:, feature_idx] <= thr)[0]
                right_indices = np.where(X[:, feature_idx] > thr)[0]

                # Skip invalid splits
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                # Calculate weighted Gini impurity
                y_left, y_right = y[left_indices], y[right_indices]
                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)
                
                weighted_gini = (len(y_left) / n_samples) * gini_left + \
                                (len(y_right) / n_samples) * gini_right
                
                # Update best split if this one is better
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = (feature_idx, thr)
        
        return best_split

    def _grow_tree(self, X, y, depth=0):
        """Recursively builds the tree."""
        n_samples, _ = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split):
            
            # Create a leaf node
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # Find the best split
        feature_idx, threshold = self._best_split(X, y)
        
        # If no good split is found (e.g., all features are identical)
        if feature_idx is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # Split the data
        left_indices = np.where(X[:, feature_idx] <= threshold)[0]
        right_indices = np.where(X[:, feature_idx] > threshold)[0]
        
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # Recursively grow children
        left_child = self._grow_tree(X_left, y_left, depth + 1)
        right_child = self._grow_tree(X_right, y_right, depth + 1)

        return Node(feature=feature_idx, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.root = self._grow_tree(X, y)

    def _traverse_tree(self, x, node):
        """Helper function to predict a single sample."""
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

# --- 4. Main Execution ---
print("--- Task 1: Decision Tree Classifier (From Scratch) ---")

# 1. Load the IRIS dataset
iris = load_iris()
X = iris.data
y = iris.target
n_classes = len(np.unique(y))
print(f"Dataset: IRIS (3 classes)")

# 2. Split the dataset (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 3. Normalize the data
# Note: Decision Trees are not sensitive to feature scaling,
# but we include it to follow the prompt's general rule.
scaler = MyStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create and Train the model
model = MyDecisionTreeClassifier(max_depth=5)
model.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred = model.predict(X_test_scaled)

# 6. Evaluate the model (Macro F1 Score)
macro_f1 = calculate_macro_f1(y_test, y_pred, n_classes)
print(f"Test Set Macro F1 Score: {macro_f1:.4f}")