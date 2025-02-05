import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut

# Load and preprocess data
def load_data(train_path, test_path):
    print("Loading data...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Filter data for labels 3 and 8, relabel as -1 and +1
    train_data = train_data[(train_data['label'] == 3) | (train_data['label'] == 8)]
    test_data = test_data[(test_data['label'] == 3) | (test_data['label'] == 8)]

    train_data['label'] = train_data['label'].replace({3: -1, 8: 1})
    test_data['label'] = test_data['label'].replace({3: -1, 8: 1})

    # Separate features and labels
    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values

    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    # Add bias term (column of ones)
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test

# Logistic loss function
def logistic_loss(t):
    return np.log(1 + np.exp(-t))

# Objective function F(beta)
def F(beta, X, y, lambda_reg):
    N = X.shape[0]
    a = beta[:-1]  # Coefficients for features
    b = beta[-1]   # Bias term
    linear_term = y * (X @ beta)
    loss = np.sum(logistic_loss(linear_term)) / N
    regularization = lambda_reg * np.linalg.norm(a) ** 2
    return loss + regularization

# Gradient of F(beta)
def gradient_F(beta, X, y, lambda_reg):
    N = X.shape[0]
    a = beta[:-1]  # Coefficients for features
    b = beta[-1]   # Bias term
    linear_term = y * (X @ beta)
    sigmoid_term = -y / (1 + np.exp(linear_term))
    grad_loss = (X.T @ sigmoid_term) / N
    grad_reg = np.hstack([2 * lambda_reg * a, 0])  # No regularization for bias
    return grad_loss + grad_reg

# Backtracking line search
def backtracking_line_search(beta, grad, X, y, lambda_reg, alpha=1.0, rho=0.5, c=0.1):
    while F(beta - alpha * grad, X, y, lambda_reg) > F(beta, X, y, lambda_reg) - c * alpha * np.linalg.norm(grad) ** 2:
        alpha *= rho
    return alpha

# Gradient descent with backtracking line search
def gradient_descent(X, y, lambda_reg, max_iter=1000, tol=1e-5):
    n_features = X.shape[1]
    beta = np.zeros(n_features)  # Initialize beta
    for i in range(max_iter):
        grad = gradient_F(beta, X, y, lambda_reg)
        alpha = backtracking_line_search(beta, grad, X, y, lambda_reg)
        beta_new = beta - alpha * grad
        if np.linalg.norm(beta_new - beta) < tol:
            print(f"Converged after {i} iterations.")
            break
        beta = beta_new
    return beta

# Leave-One-Out Cross-Validation for lambda selection
def loocv_lambda_selection(X, y, lambda_candidates):
    loo = LeaveOneOut()
    best_lambda = None
    best_error = float('inf')

    for lambda_reg in lambda_candidates:
        errors = []
        for train_idx, val_idx in loo.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            beta = gradient_descent(X_train, y_train, lambda_reg)
            y_pred = np.sign(X_val @ beta)
            errors.append(y_pred != y_val)

        avg_error = np.mean(errors)
        print(f"Lambda: {lambda_reg}, Error: {avg_error}")
        if avg_error < best_error:
            best_error = avg_error
            best_lambda = lambda_reg

    return best_lambda

# Predict labels for test data
def predict(X, beta):
    return np.sign(X @ beta)

# Evaluate success rate
def evaluate(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Main function
def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data('mnist_train.csv', 'mnist_test.csv')

    # Reduce dataset size for testing
    X_train, y_train = X_train[:100], y_train[:100]

    # Define lambda candidates for LOOCV
    lambda_candidates = [0.01, 0.1]  # Test with fewer values

    # Select best lambda using LOOCV
    print("Performing LOOCV for lambda selection...")
    best_lambda = loocv_lambda_selection(X_train, y_train, lambda_candidates)
    print(f"Best lambda: {best_lambda}")

    # Train model with best lambda
    print("Training model with best lambda...")
    beta = gradient_descent(X_train, y_train, best_lambda)

    # Predict and evaluate on test data
    print("Evaluating on test data...")
    y_pred = predict(X_test, beta)
    success_rate = evaluate(y_test, y_pred)
    print(f"Success rate on test data: {success_rate * 100:.2f}%")

if __name__ == "__main__":
    main()
