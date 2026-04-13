# Multiclass Classification — Iris Dataset

Neural network trained to classify iris flowers into 3 species using TensorFlow/Keras.

## Problem

Classify iris flowers into 3 species (setosa, versicolor, virginica) based on 4 measurements: sepal length, sepal width, petal length, and petal width.

## Key Learnings

- EDA with pairplots to understand class separability before training
- Train/test split before normalization to avoid data leakage
- Manual feature normalization using mean and std from the training set only
- Keras Sequential API with ReLU hidden layers and Softmax output
- `SparseCategoricalCrossentropy` as the loss function for multiclass problems
- Model evaluation and prediction interpretation with class probabilities

## Architecture

| Layer | Neurons | Activation |
|-------|---------|------------|
| Input | 4 | — |
| Hidden | 8 | ReLU |
| Output | 3 | Softmax |

- Optimizer: Adam (lr=0.001)
- Loss: SparseCategoricalCrossentropy
- Epochs: 1000

## Results

- Test accuracy: **100%**
- The model shows lower confidence (~65-72%) on samples from classes 1 and 2 that overlap — consistent with what the pairplot revealed during EDA

## Key insight from EDA

The pairplot showed that `petal_length` and `petal_width` separate the 3 classes much better than the sepal features. Class 0 (setosa) is linearly separable, while classes 1 and 2 have some overlap — which explains the lower confidence predictions on boundary samples.
