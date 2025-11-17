import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Set a random seed for reproducible results
np.random.seed(42)

# --- 1. Normalization Technique: Standard Scaler ---
class MyStandardScaler:
    """
    Transforms features by removing the mean and scaling to unit variance.
    x_scaled = (x - mean(x)) / std(x)
    """
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-9 # Add epsilon for stability

    def transform(self, X):
        return (X - self.mean_) / self.std_
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# --- 2. Regression Metric: Mean Squared Error (MSE) ---
def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared Error from scratch.
    """
    return np.mean((y_true - y_pred) ** 2)

# --- 3. Regression Metric: Mean Absolute Error (MAE) ---
def mean_absolute_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Error from scratch.
    """
    return np.mean(np.abs(y_true - y_pred))

# --- 4. Model: Linear Regression (OLS) ---
class MyLinearRegression:
    """
    Implements Linear Regression using the Ordinary Least Squares (OLS)
    closed-form solution: beta = (X_T * X)^-1 * X_T * y
    """
    def __init__(self):
        self.beta_ = None # Will store coefficients [bias, w1, w2, ...]

    def _add_intercept(self, X):
        # Adds a column of ones to X for the intercept (bias) term
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    def fit(self, X, y):
        # Add the intercept (bias) term
        X_b = self._add_intercept(X)
        
        # Calculate coefficients using the OLS formula
        try:
            # beta = (X^T * X)^-1 * (X^T * y)
            self.beta_ = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        except np.linalg.LinAlgError:
            print("Error: Singular matrix. Cannot compute inverse.")
            self.beta_ = np.zeros(X_b.shape[1])

    def predict(self, X):
        # Add the intercept term
        X_b = self._add_intercept(X)
        # Predict: y_hat = X_b * beta
        return X_b @ self.beta_

# --- 5. Main Execution ---
print("--- Task 1: Linear Regression (From Scratch) ---")

# 1. Create a synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
print(f"Dataset: Synthetic Regression")

# 2. Split the dataset (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 3. Normalize the data (as requested)
# Note: OLS is not sensitive to feature scaling, but we apply it
# as per the prompt's general rule ("In most cases, normalization is required").
scaler = MyStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create and Train the model
model = MyLinearRegression()
model.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred = model.predict(X_test_scaled)

# 6. Evaluate the model (MSE and MAE)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Test Set Mean Squared Error (MSE):   {mse:.4f}")
print(f"Test Set Mean Absolute Error (MAE):  {mae:.4f}")