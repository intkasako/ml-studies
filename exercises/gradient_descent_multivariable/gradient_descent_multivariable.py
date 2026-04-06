import numpy as np

x = np.array([[1,2,4], #x 10 examples, 3 features
              [3,4,5],
              [6,7,8],
              [8, 11, 7],
              [9, 10, 11],
              [10, 13, 14],
              [15, 15, 17],
              [17, 19, 20],
              [21, 29, 23],
              [24, 30, 26]])

w = np.array([0, 0, 0]) #weights for the features
y = x @ np.array([2, 5, 1]) + np.random.randn(10) * 0.5 #target values with some noise 

learning_rate = 0.0001

x = (x - x.mean(axis=0)) / x.std(axis=0) #feature scaling (standardization) to improve convergence of gradient descent
y = (y - y.mean()) / y.std() #target variable scaling (standardization) to improve convergence of gradient descent

def update_weights(x, w, learning_rate, erros, len_x):
    w = w - learning_rate * ((x.T) @ erros) / len_x #update weights using gradient descent
    return w

def gradient_descent(x, y, w, learning_rate,iterations):
    for i in range (iterations):
        predicted = np.dot(x, w) #predicted values
        erros = predicted - y #errors
        cost = np.sum(erros ** 2) / (2 * len(x)) #cost function
        w = update_weights(x, w, learning_rate, erros, len(x)) #update weights using gradient descent
        
        if(i % 100 == 0): #print weights every 100 iterations
            print(f"Updated weights: {w}")
            print(f"Cost: {cost}")
            
gradient_descent(x, y, w, learning_rate, iterations=1000)

