# Customer Segmentation — Unsupervised Clustering

Unsupervised clustering project on the **Mall Customers** dataset to identify natural customer segments based on demographics and spending behavior.

The project compares **K-Means**, **Gaussian Mixture Models**, and **DBSCAN**, using the Elbow method and Silhouette score to select the optimal number of clusters, and produces business-oriented profiles for each segment.

## Project Structure

```
customer_segmentation/
├── data/raw/                    raw dataset (CSV)
├── notebooks/                   exploratory analysis
│   ├── 01_eda.ipynb
│   ├── 02_clustering.ipynb
│   └── 03_advanced_methods.ipynb
├── src/                         reusable pipeline code
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── evaluation.py
│   └── visualization.py
├── models/                      saved model artifacts (.pkl)
├── reports/figures/             final report-quality figures
├── main.py                      end-to-end pipeline orchestrator
├── requirements.txt
└── README.md
```

## Dataset

**Mall Customer Segmentation Data** — 200 customers, 5 columns:

| Column | Description |
|---|---|
| `CustomerID` | unique identifier (dropped during preprocessing) |
| `Gender` | Male / Female |
| `Age` | customer age in years |
| `Annual Income (k$)` | annual income in thousands of dollars |
| `Spending Score (1-100)` | mall-assigned score reflecting spending behavior |

Source: [Kaggle — Customer Segmentation Tutorial](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

Place the file at `data/raw/Mall_Customers.csv` before running the pipeline.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**End-to-end pipeline:**

```bash
python main.py
```

**Exploratory notebooks** (recommended order):

1. `notebooks/01_eda.ipynb` — distributions, pairwise relationships, initial intuitions
2. `notebooks/02_clustering.ipynb` — K-Means with elbow / silhouette analysis
3. `notebooks/03_advanced_methods.ipynb` — GMM and DBSCAN comparison

## Approach

1. **Preprocessing** — drop ID, encode `Gender`, standardize features (essential for distance-based clustering)
2. **Model selection** — sweep K from 2–10, plot inertia (elbow) and silhouette score
3. **Final clustering** — train K-Means with the chosen K
4. **Profiling** — compute per-cluster means and label segments by business meaning
5. **Comparison** — repeat with GMM (probabilistic, elliptical clusters) and DBSCAN (density-based, detects outliers)

## Results

_To be filled in after running the pipeline._

## Tech Stack

Python 3.11 · scikit-learn · pandas · numpy · matplotlib · seaborn · jupyter

---

_Studies guided by Andrew Ng's Machine Learning Specialization._
