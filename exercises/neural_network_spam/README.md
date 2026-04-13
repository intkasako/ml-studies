# Neural Network — Email Spam Detection

Binary classification using a neural network built with TensorFlow/Keras.

## What it does
- Classifies emails as spam (1) or not spam (0)
- Uses 3 features: number of links, exclamation marks, and presence of promotional words
- Applies Z-score feature scaling before training
- Architecture: 2 dense layers (4 neurons hidden, 1 output) with sigmoid activation

## Key learnings
- Small datasets require more epochs to converge consistently
- Feature scaling is essential even with few features at different scales
- Random seeds (`np.random.seed` + `tf.random.set_seed`) make weight initialization deterministic and results reproducible

## Graphs
![Loss Curve](images/loss_curve.png)
![Predictions](images/predictions.png)

## Requirements
```bash
pip install tensorflow numpy matplotlib
```

## Usage
```bash
python neural_network_spam.py
```