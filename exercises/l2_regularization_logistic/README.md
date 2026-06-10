
I built this exercise with Claude Code's help. I asked it to create a dataset to force overfitting so I could practice logistic regression with L2 regularization, and I also asked it to create the skeleton code to make it faster to implement (I already had the code for logistic regression without regularization, so I just needed to make the functions).

## What this exercise covers

- Binary logistic regression from scratch (numpy)
- L2 regularization applied to logistic regression
- Feature scaling (z-score normalization)
- Comparing regularized vs unregularized models
- Visualizing decision boundaries side by side

## Dataset

15 credit approval examples with 2 features:
- `income_score` (0–5)
- `credit_history_score` (0–3)

Target: `1` = approved, `0` = rejected. The dataset is intentionally small to make overfitting visible.

Smaller weights with L2 indicate a more conservative model that generalizes better to unseen data.
