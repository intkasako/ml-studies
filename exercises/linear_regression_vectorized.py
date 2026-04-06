import numpy as np

x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 7, 7],
              [9, 10, 11]])

w = np.array([1, 3, 5])

result = np.dot(x,w)
print(result)
print(result.shape)

w_2d = np.array([[1, 3, 5]])  # shape (1, 3)
print(w_2d.shape)
print(np.dot(x, w_2d.T))     # precisa transpor — por quê?