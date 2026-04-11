
import numpy as np

# simple dataset: tumor size (X) and whether it's malignant (y)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# initialize weights and bias
w = np.array([0.1]) # only one feature (tumor size), so weight is a 1D array
b = 0
alpha = 0.01 # learning rate

def sigmoid(z):
    predicted = 1 / (1 + np.exp(-z)) #sigmoid function to convert linear output to probability
    return predicted

def log_loss_cost(predicted, y, m):
    cost = -(1/m) * np.sum(y * np.log(predicted) + 
                           (1 - y) * np.log(1 - predicted)) #log loss cost function
    return cost

def update_weights(X, predicted, y, w, b, alpha, m):
    dw = (1/m) * (X.T @ (predicted - y)) #gradient of weights
    db = (1/m) * np.sum(predicted - y) #gradient of bias
    w = w - alpha * dw #update weights
    b = b - alpha * db #update bias
    return w, b

for i in range (1001):
    z = X @ w + b #linear combination of inputs and weights
    predicted = sigmoid(z) #predicted probabilities
    cost = log_loss_cost(predicted, y, len(X)) #calculate cost
    w, b = update_weights(X, predicted, y, w, b, alpha, len(X)) #update weights and bias
    if i % 100 == 0:
        print(f"Cost at iteration {i}: {cost}")
        print(f"Weights: {w}, Bias: {b}")
        
x_test = np.array([6.5]) #test with a new tumor size
z = x_test @ w + b
probability = sigmoid(z)
print(f"Predicted probability of being malignant for tumor size {x_test[0]}: {probability}")
