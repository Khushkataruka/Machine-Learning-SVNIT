import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
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

# --- 3. Oversampling Technique: SMOTE ---
class MySMOTE:
    """
    Implements SMOTE (Synthetic Minority Over-sampling TEchnique) from scratch.
    """
    def __init__(self, k_neighbors=5):
        self.k = k_neighbors

    def _find_k_nearest(self, sample, X_minority):
        """Finds k-nearest neighbors for a sample within the minority set."""
        distances = np.linalg.norm(X_minority - sample, axis=1)
        k_nearest_indices = np.argsort(distances)[1:self.k + 1]
        return X_minority[k_nearest_indices]

    def fit_resample(self, X, y):
        counts = Counter(y)
        minority_class = min(counts, key=counts.get)
        majority_class = max(counts, key=counts.get)
        
        n_minority = counts[minority_class]
        n_majority = counts[majority_class]
        n_to_synthesize = n_majority - n_minority
        
        if n_to_synthesize <= 0:
            return X, y # Already balanced
            
        X_minority = X[y == minority_class]
        new_samples = []
        
        for _ in range(n_to_synthesize):
            random_idx = np.random.randint(0, n_minority)
            sample = X_minority[random_idx]
            neighbors = self._find_k_nearest(sample, X_minority)
            neighbor = neighbors[np.random.randint(0, self.k)]
            
            # Create a new sample on the line segment
            diff = neighbor - sample
            gap = np.random.rand()
            new_sample = sample + gap * diff
            new_samples.append(new_sample)
            
        # Combine original data with new synthetic samples
        X_resampled = np.vstack((X, np.array(new_samples)))
        y_resampled = np.hstack((y, np.full(n_to_synthesize, minority_class)))
        
        return X_resampled, y_resampled

# --- 4. Model: Linear Regression (used as a Classifier) ---
class LinearRegressionClassifier:
    """
    Implements Linear Regression trained via Gradient Descent.
    The model is trained to minimize MSE against 0/1 labels.
    A threshold is applied during prediction to classify.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent (minimizing MSE)
        for _ in range(self.n_iterations):
            # 1. Predict continuous value
            y_hat = X @ self.weights + self.bias
            
            # 2. Calculate error
            error = y_hat - y
            
            # 3. Calculate gradients of MSE
            # dw = (2/m) * X^T * (y_hat - y)
            dw = (2 / n_samples) * X.T @ error
            # db = (2/m) * sum(y_hat - y)
            db = (2 / n_samples) * np.sum(error)
            
            # 4. Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X, threshold=0.5):
        # Predict continuous value
        y_hat_continuous = X @ self.weights + self.bias
        
        # Apply threshold to get class
        return (y_hat_continuous >= threshold).astype(int)

# --- 5. Main Execution ---
print("--- Task 2 (Modified): SMOTE + Linear Regression (as Classifier) ---")

# 1. Create a synthetic IMBALANCED dataset
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, weights=[0.9, 0.1], 
                           flip_y=0, random_state=42)
n_classes = len(np.unique(y))
print(f"Original dataset shape: {X.shape}")
print(f"Original class distribution: {Counter(y)}")

# 2. Split the dataset (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 3. Normalize the data
scaler = MyStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model 1: Trained on IMBALANCED data (Baseline) ---
print("\n--- Model 1: Training on Imbalanced Data ---")
model_1 = LinearRegressionClassifier(learning_rate=0.01, n_iterations=1000)
model_1.fit(X_train_scaled, y_train)
y_pred_1 = model_1.predict(X_test_scaled)
macro_f1_1 = calculate_macro_f1(y_test, y_pred_1, n_classes)
print(f"Test Set Macro F1 Score (Imbalanced): {macro_f1_1:.4f}")

# --- Model 2: Trained on OVERSAMPLED data (SMOTE) ---
print("\n--- Model 2: Training on SMOTE Oversampled Data ---")
# 4. Apply SMOTE to the training data
smote = MySMOTE(k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"Oversampled training data shape: {X_train_resampled.shape}")
print(f"Oversampled class distribution: {Counter(y_train_resampled)}")

# 5. Train the model on the new, balanced data
model_2 = LinearRegressionClassifier(learning_rate=0.01, n_iterations=1000)
model_2.fit(X_train_resampled, y_train_resampled)

# 6. Evaluate on the ORIGINAL, unbalanced test set
y_pred_2 = model_2.predict(X_test_scaled)
macro_f1_2 = calculate_macro_f1(y_test, y_pred_2, n_classes)
print(f"Test Set Macro F1 Score (SMOTE): {macro_f1_2:.4f}")

# --- 7. Comparison ---
print("\n--- Comparison ---")
print(f"Imbalanced Model F1: {macro_f1_1:.4f}")
print(f"SMOTE Model F1:      {macro_f1_2:.4f}")