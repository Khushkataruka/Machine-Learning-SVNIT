import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

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

# --- 2. Regression Metric: Mean Squared Error (MSE) ---
def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared Error from scratch.
    """
    return np.mean((y_true - y_pred) ** 2)

# --- 3. Model: Linear Regression with NAG ---
class MyLinearRegression_NAG:
    """
    Implements Linear Regression from scratch using
    Nesterov Accelerated Gradient (NAG) descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, gamma=0.9):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.gamma = gamma  # Momentum parameter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 1. Initialize weights, bias, and momentum vector (v)
        self.weights = np.zeros(n_features)
        self.bias = 0
        v_weights = np.zeros(n_features)
        v_bias = 0

        # 2. NAG Gradient Descent
        for _ in range(self.n_iterations):
            
            # 1. Calculate the "lookahead" weights and bias
            weights_lookahead = self.weights - self.gamma * v_weights
            bias_lookahead = self.bias - self.gamma * v_bias
            
            # 2. Calculate predictions at the "lookahead" position
            y_hat_lookahead = X @ weights_lookahead + bias_lookahead
            
            # 3. Calculate gradient at the "lookahead" position
            # Derivative of MSE: (2/m) * X^T * (y_hat - y)
            dw = (2 / n_samples) * X.T @ (y_hat_lookahead - y)
            db = (2 / n_samples) * np.sum(y_hat_lookahead - y)

            # 4. Update momentum vector (v)
            # v = gamma * v + alpha * gradient_at_lookahead
            v_weights = (self.gamma * v_weights) + (self.learning_rate * dw)
            v_bias = (self.gamma * v_bias) + (self.learning_rate * db)
            
            # 5. Update weights (using the momentum vector)
            self.weights -= v_weights
            self.bias -= v_bias

    def predict(self, X):
        # Predict: y_hat = X * weights + bias
        return X @ self.weights + self.bias

# --- 4. Main Execution ---
print("\n--- Task 2: Linear Regression with NAG (From Scratch) ---")

# 1. Create a synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
print(f"Dataset: Synthetic Regression")

# 2. Split the dataset (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 3. Normalize the data (Required for Gradient Descent)
scaler = MyStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create and Train the model
model = MyLinearRegression_NAG(learning_rate=0.01, n_iterations=1000, gamma=0.9)
model.fit(X_train_scaled, y_train)

# 5. Make predictions
y_pred = model.predict(X_test_scaled)

# 6. Evaluate the model (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Test Set MSE: {mse:.4f}")