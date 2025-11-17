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

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        f1_scores.append(f1)

    # Macro F1 is the unweighted average
    return np.mean(f1_scores)

# --- 3. Model: AdaBoost ---

class MyDecisionStump:
    """
    A simple Decision Stump (a 1-level Decision Tree).
    This is the "weak learner" for AdaBoost.
    """
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None # The "amount of say" for this stump

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        min_weighted_error = float('inf')

        # Loop through all features
        for j in range(n_features):
            unique_values = np.unique(X[:, j])
            
            # Loop through all unique values as potential thresholds
            for threshold in unique_values:
                # Try a split
                p = 1
                predictions = np.ones(n_samples)
                predictions[X[:, j] < threshold] = -1

                # Calculate weighted error
                weighted_error = np.sum(sample_weights[y != predictions])

                # Check if flipping polarity is better
                if weighted_error > 0.5:
                    p = -1
                    weighted_error = 1 - weighted_error

                # Store the best split
                if weighted_error < min_weighted_error:
                    min_weighted_error = weighted_error
                    self.polarity = p
                    self.feature_idx = j
                    self.threshold = threshold
        
        # Calculate alpha (amount of say)
        # Add epsilon to avoid division by zero if error is 0
        self.alpha = 0.5 * np.log((1.0 - min_weighted_error + 1e-10) / (min_weighted_error + 1e-10))

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        
        if self.polarity == 1:
            predictions[X[:, self.feature_idx] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_idx] > self.threshold] = -1
            
        return predictions

class MyAdaBoost:
    """
    Implements the AdaBoost algorithm from scratch.
    """
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.estimators_ = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        # AdaBoost requires labels to be {-1, 1}
        y_ = np.where(y <= 0, -1, 1)
        
        # 1. Initialize sample weights
        w = np.full(n_samples, (1 / n_samples))
        
        self.estimators_ = []
        for _ in range(self.n_estimators):
            # 2. Train a weak learner (stump)
            stump = MyDecisionStump()
            stump.fit(X, y_, w)
            
            # 3. Make predictions
            y_pred = stump.predict(X)
            
            # 4. Update sample weights
            # w_i = w_i * exp(-alpha * y_i * h_i(x))
            w *= np.exp(-stump.alpha * y_ * y_pred)
            
            # 5. Normalize weights
            w /= np.sum(w)
            
            # Save the trained stump
            self.estimators_.append(stump)

    def predict(self, X):
        # Get predictions from all stumps
        stump_preds = [stump.alpha * stump.predict(X) for stump in self.estimators_]
        
        # Sum the weighted predictions
        y_pred = np.sum(stump_preds, axis=0)
        
        # Return the sign (the final class)
        return np.sign(y_pred).astype(int)


# --- 4. Main Execution ---
print("\n--- Task 2: AdaBoost (From Scratch) ---")

# 1. Load IRIS data and make it binary
iris = load_iris()
# We'll use class 1 and class 2
X = iris.data[50:]
y = iris.target[50:]
# Map labels from {1, 2} to {0, 1} for F1 score,
# and AdaBoost will internally map {0, 1} to {-1, 1}
y = np.where(y == 1, 0, 1) 
n_classes = len(np.unique(y))
print(f"Dataset: IRIS (Classes 1 vs 2)")

# 2. Split the dataset (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 3. Normalize the data
# Note: Decision trees are not sensitive to scaling, but we follow the prompt.
scaler = MyStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create and Train the model
model = MyAdaBoost(n_estimators=10)
model.fit(X_train_scaled, y_train)

# 5. Make predictions
# Model predicts in {-1, 1}, so we map back to {0, 1}
y_pred_raw = model.predict(X_test_scaled)
y_pred = np.where(y_pred_raw == -1, 0, 1)

# 6. Evaluate the model (Macro F1 Score)
macro_f1 = calculate_macro_f1(y_test, y_pred, n_classes)
print(f"Test Set Macro F1 Score: {macro_f1:.4f}")