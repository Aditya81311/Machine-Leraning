import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function (log loss)
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    print(f"Cost:{cost}")
    return cost

# Gradient descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Example dataset (binary classification)
# Features: study hours, sleep hours
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])   # inputs
y = np.array([0, 0, 1, 1])                       # labels

# Add bias term (column of ones)
X = np.c_[np.ones(X.shape[0]), X]

# Initialize parameters
theta = np.zeros(X.shape[1])

# Run gradient descent
alpha = 0.1
iterations = 1000
theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

print("Final parameters (theta):", theta)
print("Final cost:", cost_history[-1])

# Predictions
pred_probs = sigmoid(X @ theta)
print(pred_probs)
predictions = (pred_probs >= 0.5).astype(int)

print("Predicted probabilities:", pred_probs)
print("Predicted classes:", predictions)


