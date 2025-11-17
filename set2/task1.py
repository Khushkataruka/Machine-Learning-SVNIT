import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

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

        # Calculate Precision and Recall for class c
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        # Calculate F1 score for class c
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        
        f1_scores.append(f1)

    # Macro F1 is the unweighted average of F1 scores
    return np.mean(f1_scores)

# --- 3. Model: Logistic Regression ---
class MyLogisticRegression:
    """
    Implements Logistic Regression from scratch using Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # Sigmoid function: 1 / (1 + e^-z)
        # Clipped to avoid overflow errors with np.exp
        z_clipped = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z_clipped))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 1. Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 2. Gradient Descent
        for _ in range(self.n_iterations):
            # Calculate linear model: z = X * weights + bias
            z = X @ self.weights + self.bias
            
            # Calculate hypothesis (predictions): h = sigmoid(z)
            h = self._sigmoid(z)
            
            # Calculate gradient of the Log Loss function
            # dw = (1/m) * X^T * (h - y)
            dw = (1 / n_samples) * X.T @ (h - y)
            # db = (1/m) * sum(h - y)
            db = (1 / n_samples) * np.sum(h - y)
            
            # 3. Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        # Predicts the probability of class 1
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        # Predicts the class label
        probabilities = self.predict_proba(X)
        # Return 1 if probability >= 0.5, else 0
        return (probabilities >= threshold).astype(int)

# --- 4. Main Execution ---
print("--- Task 1: Logistic Regression (From Scratch) ---")

# 1. Load and prepare the IRIS dataset for binary classification
iris = load_iris()
# We'll only use the first 100 samples (class 0 and 1)
X = iris.data[:100]
y = iris.target[:100]
n_classes = len(np.unique(y)) # This will be 2
print(f"Dataset: IRIS (Classes 0 vs 1)")

# 2. Split the dataset (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 3. Normalize the data (Required for Gradient Descent)
scaler = MyStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create and Train the model
model = MyLogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred = model.predict(X_test_scaled)

# 6. Evaluate the model (Macro F1 Score)
macro_f1 = calculate_macro_f1(y_test, y_pred, n_classes)
print(f"Test Set Macro F1 Score: {macro_f1:.4f}")