import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


np.random.seed(42)
tf.random.set_seed(42)

# Dataset: email spam detection
# Features:
# - num_links: number of links in the email (higher = more likely spam)
# - num_exclamations: number of exclamation marks (higher = more likely spam)
# - has_promotion_word: contains words like "free", "promotion", "offer" (0 or 1)
# Target: 0 = not spam, 1 = spam

X = np.array([
    [1,  1, 0],
    [10, 5, 1],
    [0,  1, 0],
    [8,  7, 1],
    [2,  1, 0],
    [15, 3, 1],
    [1,  2, 0],
    [12, 8, 1],
    [0,  0, 0],
    [9,  6, 1]
])

y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

#feature scalling (z-score normalization)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

#neural network 

model = Sequential([
    Dense(4, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])

loss = tf.keras.losses.BinaryCrossentropy()
optimizer=tf.keras.optimizers.Adam(0.001)

model.compile(loss=loss,optimizer=optimizer)

history = model.fit(X_scaled, y, epochs=1000, verbose =0)

X_new = [[1, 5, 1]]
X_new_scaled = (X_new - X_mean) / X_std

probability = model.predict(X_new_scaled)[0][0]
print(f"Spam probability: {probability:.3f}")
print(f"Classification: {'Spam' if probability >= 0.5 else 'Not Spam'}")

#first graph: loss curve
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#second graph: predicted probability per example, sorted lowest to highest
predictions = model.predict(X_scaled, verbose=0).flatten()
sorted_indices = np.argsort(predictions)
sorted_probs = predictions[sorted_indices]
sorted_labels = y[sorted_indices]

insert_pos = np.searchsorted(sorted_probs, probability)
all_probs = np.insert(sorted_probs, insert_pos, probability)
all_labels = np.insert(sorted_labels.astype(int), insert_pos, -1)
all_xticks = [f'Ex {i}' for i in sorted_indices]
all_xticks.insert(insert_pos, 'X_new')

colors = []
for label in all_labels:
    if label == 1:
        colors.append('red')
    elif label == 0:
        colors.append('blue')
    else:
        colors.append('orange')

plt.figure()
plt.bar(range(len(all_probs)), all_probs, color=colors)
plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
plt.xticks(range(len(all_probs)), all_xticks)
plt.xlabel('Example (sorted by probability)')
plt.ylabel('Spam Probability')
plt.title('Predicted Spam Probability per Example')
plt.legend(handles=[
    plt.Rectangle((0,0),1,1, color='red', label='Spam (y=1)'),
    plt.Rectangle((0,0),1,1, color='blue', label='Not Spam (y=0)'),
    plt.Rectangle((0,0),1,1, color='orange', label='X_new (unknown)'),
    plt.Line2D([0],[0], color='black', linestyle='--', label='Decision boundary (0.5)')
])
plt.tight_layout()
plt.show()