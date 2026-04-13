from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


data = load_iris()
X, y = data.data, data.target



dataframe = pd.DataFrame(X, columns=("sepal_length", "sepal_width", "petal_length",
                            "petal_width"))
dataframe["class"] = y

pairplot = seaborn.pairplot(dataframe, hue="class")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train_scaled = (X_train-X_mean) / X_std
X_test_scaled = (X_test-X_mean) / X_std

#training session

model = Sequential([Dense(8, activation='relu'),
                    Dense(3, activation='softmax')])

loss = tensorflow.keras.losses.SparseCategoricalCrossentropy()
optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
model.fit(X_train_scaled, y_train, epochs=1000, verbose=0)
#-

print(f"dataset size {X.shape}")

#evaluating
model.evaluate(X_test_scaled, y_test)

#predictions
predict = model.predict(X_test_scaled)
class_predicted = np.argmax(predict, axis=1)
class_probability = np.max(predict, axis=1)
class_dataframe = pd.DataFrame({"class prediction": class_predicted, "class probability": class_probability, "real class": y_test})

print(class_dataframe)
plt.show()