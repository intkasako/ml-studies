import numpy as np
import matplotlib.pyplot as plt

# Dataset: credit approval (small dataset — prone to overfitting)
# Features: income_score (0-5), credit_history_score (0-3)
# Target: 1 = approved, 0 = rejected

X = np.array([
    [2.5, 1.2], [3.1, 1.8], [1.0, 0.5], [3.8, 2.1], [2.0, 0.8],
    [0.5, 0.2], [4.5, 3.0], [1.5, 1.0], [3.5, 2.5], [0.8, 0.3],
    [4.0, 2.8], [2.8, 1.5], [1.2, 0.6], [3.2, 2.0], [0.3, 0.1]
])
y = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0])

# Feature scaling 
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

m, n = X_scaled.shape  
# m = num of examples, n = num of features


# Sigmoid function 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Cost function WITHOUT regularization 
def cost(X, y, w, b):
    linear = (X @ w) + b
    return -(1/m) * np.sum(y * np.log(sigmoid(linear)) +
                           (1-y) * np.log(1-sigmoid(linear)))
    


#  Cost function with L2 regularization 
def cost_regularized(X, y, w, b, lambda_):
    return cost(X, y, w, b)  + (lambda_ / (2 * m)) * np.sum(w ** 2)


# Gradient descent WITHOUT regularization 
def gradient_descent(X, y, w, b, alpha, iterations):
    costs = []
    for i in range(iterations):
        linear = X @ w + b
        predicted = sigmoid(linear) 
        erros = predicted - y
         
        dw = (1/m) * (X.T) @ (erros)
        db = (1/m) * np.sum(erros)
        
        w = w - alpha * dw
        b = b - alpha * db
        
        costs.append(cost(X, y, w, b)) 
        
    return w, b, costs


# Gradient descent with L2 regularization 
def gradient_descent_regularized(X, y, w, b, alpha, lambda_, iterations):
    costs = []
    for i in range(iterations):
        linear = X @ w + b
        predicted = sigmoid(linear) 
        erros = predicted - y
         
        dw = (1/m) * (X.T) @ (erros) + (lambda_ / m) * w
        db = (1/m) * np.sum(erros)
        
        w = w - alpha * dw
        b = b - alpha * db
        
        costs.append(cost_regularized(X, y, w, b, lambda_)) 
        
    return w, b, costs


# Training 
alpha = 0.1
iterations = 1000
lambda_ = 0.1  

w_init = np.zeros(n)
b_init = 0.0

# Train both models
w_noreg, b_noreg, costs_noreg = gradient_descent(
    X_scaled, y, w_init.copy(), b_init, alpha, iterations
)
w_reg, b_reg, costs_reg = gradient_descent_regularized(
    X_scaled, y, w_init.copy(), b_init, alpha, lambda_, iterations
)

print(f"Without regularization — w: {w_noreg}, b: {b_noreg:.4f}")
print(f"With L2 regularization — w: {w_reg}, b: {b_reg:.4f}")

# Q: What happens to w when lambda_ is very large (e.g. 100)?
# A: TODO

# Q: What happens when lambda_ = 0?
# A: TODO

# Q: Why don't we regularize the bias b?
# A: TODO


# --- Plot: decision boundaries side by side ---
def plot_decision_boundary(ax, X_scaled, y, w, b, X_mean, X_std, title):
    # Build a grid over the original (unscaled) feature space
    x0_range = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
    x1_range = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200)
    xx0, xx1 = np.meshgrid(x0_range, x1_range)

    # Scale the grid using training mean and std
    grid = np.c_[xx0.ravel(), xx1.ravel()]
    grid_scaled = (grid - X_mean) / X_std

    # TODO: compute probabilities for each point in the grid using sigmoid
    # probs = ...

    # TODO: reshape probs to xx0.shape and plot contour at 0.5
    # ax.contourf(xx0, xx1, probs.reshape(xx0.shape), levels=[0, 0.5, 1], ...)
    # ax.contour(xx0, xx1, probs.reshape(xx0.shape), levels=[0.5], ...)

    # Plot the data points (approved = green, rejected = red)
    approved = y == 1
    rejected = y == 0
    ax.scatter(X[approved, 0], X[approved, 1], color='green', label='Approved (y=1)', zorder=3)
    ax.scatter(X[rejected, 0], X[rejected, 1], color='red', label='Rejected (y=0)', zorder=3)
    ax.set_xlabel('Income Score')
    ax.set_ylabel('Credit History Score')
    ax.set_title(title)
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(axes[0], X_scaled, y, w_noreg, b_noreg, X_mean, X_std,
                       "No Regularization")
plot_decision_boundary(axes[1], X_scaled, y, w_reg, b_reg, X_mean, X_std,
                       f"L2 Regularization (λ={lambda_})")
plt.tight_layout()
plt.show()

# --- Bonus: plot cost curves for both models ---
# TODO (optional): plot costs_noreg and costs_reg on the same graph
# This shows how regularization affects the training loss
