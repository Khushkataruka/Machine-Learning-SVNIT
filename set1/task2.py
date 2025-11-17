import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

# Set a random seed for reproducible results
np.random.seed(42)

# --- 1. Helper Function: Train/Test Split ---
def custom_train_test_split(X, y, test_size=0.20):
    """
    Manually splits data into train and test sets (80:20 ratio).
    """
    n_samples = X.shape[0]
    shuffled_indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_size))
    
    train_indices = shuffled_indices[:split_idx]
    test_indices = shuffled_indices[split_idx:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

# --- 2. Evaluation Metric: Macro F1 Score ---
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

# --- 3. Model: Bagging Classifier ---
class MyBaggingClassifier:
    """
    A from-scratch implementation of the Bagging (Bootstrap Aggregating)
    ensemble method.
    """
    def __init__(self, base_estimator, n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators_ = [] # This will store our trained models

    def _bootstrap_sample(self, X, y):
        """
        Creates a single bootstrap sample (sampling with replacement).
        """
        n_samples = X.shape[0]
        # Get indices by sampling WITH replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """
        Trains the ensemble of estimators on bootstrap samples.
        """
        self.estimators_ = []
        
        for _ in range(self.n_estimators):
            # 1. Create a bootstrap sample
            X_boot, y_boot = self._bootstrap_sample(X, y)
            
            # 2. Create a new, fresh instance of the base estimator
            estimator = clone(self.base_estimator)
            
            # 3. Fit this new estimator on the bootstrap sample
            estimator.fit(X_boot, y_boot)
            
            # 4. Add the trained estimator to our list
            self.estimators_.append(estimator)
            
    def predict(self, X):
        """
        Aggregates predictions from all estimators using a majority vote.
        """
        # 1. Get predictions from every single estimator
        all_predictions_list = [estimator.predict(X) for estimator in self.estimators_]
        
        # 2. Stack them into a 2D numpy array: (n_estimators, n_samples)
        all_predictions = np.array(all_predictions_list)
        
        # 3. Transpose to (n_samples, n_estimators)
        predictions_per_sample = all_predictions.T
        
        # 4. Perform the majority vote (aggregation) for each sample (row)
        # np.argmax(np.bincount(row)) is a fast way to find the mode
        final_predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), 
            axis=1, 
            arr=predictions_per_sample
        )
        
        return final_predictions

# --- 4. Main Execution ---

# Load the IRIS dataset
iris = load_iris()
X = iris.data
y = iris.target
n_classes = len(np.unique(y))

# Split the dataset (80:20 ratio) using our custom function
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.20)

print(f"Total dataset shape: {X.shape}")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}\n")

# --- 5. Create and Train the Model ---
# Define the base estimator we want to bag
base_tree = DecisionTreeClassifier(random_state=42)

# Initialize our custom Bagging model
my_model = MyBaggingClassifier(
    base_estimator=base_tree,
    n_estimators=100  # We'll create an ensemble of 100 trees
)

print("Training the custom Bagging classifier...")
my_model.fit(X_train, y_train)
print("Training complete.\n")

# --- 6. Evaluate the Model ---
y_pred = my_model.predict(X_test)

# Calculate the required metric: Macro F1 Score
macro_f1 = calculate_macro_f1(y_test, y_pred, n_classes)

# We can also check accuracy
accuracy = np.mean(y_pred == y_test)

print("--- Evaluation Results (From Scratch Model) ---")
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
print(f"Test Set Macro F1 Score: {macro_f1:.4f}")