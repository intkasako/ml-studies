# Logistic Regression — Binary Classification from Scratch

Implementation of logistic regression with gradient descent from scratch using NumPy, applied to a binary tumor classification problem.

## What it does
- Classifies tumors as benign (0) or malignant (1) based on size
- Implements sigmoid function to convert linear output to probability
- Uses log loss (binary cross-entropy) as cost function
- Runs gradient descent for 1000 iterations
- Predicts probability for new inputs

## Concepts covered
- Sigmoid activation function
- Log loss cost function
- Gradient descent for classification
- Binary decision boundary (threshold = 0.5)

## Requirements
```bash
pip install numpy
```

## Usage
```bash
python logistic_regression.py
```

## Key difference from linear regression
Linear regression outputs any number. Logistic regression applies a sigmoid function to compress the output between 0 and 1, representing the probability of belonging to a class.