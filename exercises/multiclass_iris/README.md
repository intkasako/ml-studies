# Multiclass Classification — Iris Dataset

Neural network trained to classify iris flowers into 3 species using TensorFlow/Keras.

## Problem

Given 4 measurements of an iris flower (sepal length, sepal width, petal length, petal width), predict which of the 3 species it belongs to: setosa, versicolor, or virginica.

## Key Learnings

Before training, a pairplot was used to explore how well each feature separates the classes. This revealed that petal length and petal width are much more informative than the sepal features, and that classes 1 and 2 have some overlap — which sets a realistic expectation for the model.

The train/test split was done before normalization to avoid data leakage. Mean and standard deviation were calculated only from the training set and then applied to both sets.

The output layer uses Softmax, which converts the raw scores into probabilities that sum to 1 — one per class. The loss function used was SparseCategoricalCrossentropy, which works directly with integer class labels.

## Architecture

| Layer | Neurons | Activation |
|-------|---------|------------|
| Input | 4 | — |
| Hidden | 8 | ReLU |
| Output | 3 | Softmax |

Optimizer: Adam (lr=0.001) — Epochs: 1000

## Results

Test accuracy of 100%. The two samples with lower confidence (around 65-72%) were from the overlap region between classes 1 and 2, which was already visible in the pairplot before training.
