import numpy as np

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

# --- 2. Normalization Technique 1: Min-Max Scaler ---
class MyMinMaxScaler:
    """
    Transforms features by scaling each feature to a [0, 1] range.
    x_scaled = (x - min(x)) / (max(x) - min(x))
    """
    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        # Add epsilon to avoid division by zero
        self.range_ = self.max_ - self.min_ + 1e-9 
        
    def transform(self, X):
        return (X - self.min_) / self.range_
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# --- 3. Normalization Technique 2: Standard Scaler ---
class MyStandardScaler:
    """
    Transforms features by removing the mean and scaling to unit variance.
    x_scaled = (x - mean(x)) / std(x)
    """
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-9 # Add epsilon

    def transform(self, X):
        return (X - self.mean_) / self.std_
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# --- 4. Evaluation Metric: Mean Squared Error (MSE) ---
def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared Error.
    """
    return np.mean((y_true - y_pred) ** 2)

# --- 5. Model: Linear Regression (Best Fit Line) ---
class MyLinearRegression:
    """
    Implements Linear Regression using the Ordinary Least Squares (OLS)
    closed-form solution: beta = (X_T * X)^-1 * X_T * y
    """
    def __init__(self):
        self.beta_ = None # Will store coefficients (weights)

    def _add_intercept(self, X):
        # Adds a column of ones to X for the intercept term
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    def fit(self, X, y):
        # Add the intercept (bias) term
        X_b = self._add_intercept(X)
        
        # Calculate coefficients using the OLS formula
        try:
            # (X^T * X)
            XtX = X_b.T @ X_b
            # (X^T * X)^-1
            XtX_inv = np.linalg.inv(XtX)
            # X^T * y
            Xty = X_b.T @ y
            # beta = (X^T * X)^-1 * (X^T * y)
            self.beta_ = XtX_inv @ Xty
        except np.linalg.LinAlgError:
            print("Error: Singular matrix. Cannot compute inverse.")
            self.beta_ = np.zeros(X_b.shape[1])

    def predict(self, X):
        # Add the intercept term
        X_b = self._add_intercept(X)
        # Predict: y_hat = X_b * beta
        return X_b @ self.beta_

# --- 6. Main Execution ---

# Generate some simple data for regression
# y = 5x + 3 + noise
X = 10 * np.random.rand(100, 1)
y = 5 * X + 3 + np.random.randn(100, 1)

# Split the dataset (80:20 ratio)
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.20)

# --- Model 1: Min-Max Normalization ---
print("--- Model 1: Min-Max Scaler ---")
min_max_scaler = MyMinMaxScaler()
X_train_scaled_1 = min_max_scaler.fit_transform(X_train)
X_test_scaled_1 = min_max_scaler.transform(X_test)

model_1 = MyLinearRegression()
model_1.fit(X_train_scaled_1, y_train)
y_pred_1 = model_1.predict(X_test_scaled_1)
mse_1 = mean_squared_error(y_test, y_pred_1)
print(f"MSE with Min-Max Scaling: {mse_1:.4f}\n")


# --- Model 2: Standardization (Z-score) ---
print("--- Model 2: Standard Scaler ---")
std_scaler = MyStandardScaler()
X_train_scaled_2 = std_scaler.fit_transform(X_train)
X_test_scaled_2 = std_scaler.transform(X_test)

model_2 = MyLinearRegression()
model_2.fit(X_train_scaled_2, y_train)
y_pred_2 = model_2.predict(X_test_scaled_2)
mse_2 = mean_squared_error(y_test, y_pred_2)
print(f"MSE with Standard Scaling: {mse_2:.4f}\n")


# --- Model 3: No Normalization (Baseline) ---
print("--- Model 3: No Normalization ---")
model_3 = MyLinearRegression()
model_3.fit(X_train, y_train)
y_pred_3 = model_3.predict(X_test)
mse_3 = mean_squared_error(y_test, y_pred_3)
print(f"MSE with No Scaling: {mse_3:.4f}\n")


# --- 7. Comparison ---
print("--- Comparison of MSEs ---")
print(f"Min-Max MSE:    {mse_1:.4f}")
print(f"Standard MSE:   {mse_2:.4f}")
print(f"No Scaling MSE: {mse_3:.4f}")