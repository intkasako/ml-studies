# I've received a simple dataset to predict if a client will pay back or not.
# 3 features: monthly income, debt and credit score.

#Target variable: 1 if the client will pay back, 0 if not.

import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [3000,  45000, 30],
    [8000,  10000, 75],
    [2000,  48000, 20],
    [15000, 5000,  90],
    [5000,  30000, 45],
    [12000, 8000,  80],
    [1500,  50000, 15],
    [9000,  12000, 70],
    [4000,  35000, 35],
    [18000, 2000,  95]
])
X_original = X.copy()
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

#feature scaling (z-score normalization) to improve convergence of gradient descent
training_mean_x = X.mean(axis=0)
training_std_x = X.std(axis=0)

X = (X - training_mean_x) / training_std_x

#initial wheights and bias
w = np.array([0.1, 0.1, 0.1])
b = 0

#learning rate
alpha = 0.001 

def sigmoid(z):
    predicted = 1 / (1 + np.exp(-z)) #sigmoid function to convert linear output to probability
    return predicted

def log_loss_cost(predicted, y, m):
    cost = -(1/m) * np.sum(y * np.log(predicted) + 
                           (1 - y) * np.log(1 - predicted)) 
    return cost

def update_weights(x, predicted, y, w, b, alpha, m):
    dw = (1/m) * (x.T @ (predicted - y)) #gradient of weights
    db = (1/m) * np.sum(predicted - y) #gradient of bias
    w = w - alpha * dw #update weights
    b = b - alpha * db #update bias
    return w, b

for i in range (1001):
    z = X @ w + b # linear function
    predicted = sigmoid(z)
    cost = log_loss_cost(predicted, y, len(X)) #calculate cost
    w, b = update_weights(X, predicted, y, w, b, alpha, len(X)) #update weights and bias
    
    if i % 100 == 0:
        print(f"Cost at iteration {i}: {cost}")
        print(f"Weights: {w}, Bias: {b}")

# new client data for prediction
new_client_x = np.array([7000, 20000, 60])
new_client_x = (new_client_x - training_mean_x) / training_std_x #feature scaling using training data mean and std
z = new_client_x @ w + b
probability = sigmoid(z)
print(f"Predicted probability of the new client paying back: {probability}")

