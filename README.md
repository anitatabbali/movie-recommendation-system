# Movie Recommendation System

A user-based collaborative filtering recommender built in Python with k-nearest neighbours (KNN) and cosine similarity, trained and evaluated on the MovieLens dataset.

Built for **COMP1827 Introduction to Artificial Intelligence** at the University of Greenwich. This was a group project in which each member developed a different recommendation algorithm. This repository contains my user-based KNN collaborative filtering model and its evaluation.

## How it works

1. **Data:** the [MovieLens Latest Small](https://grouplens.org/datasets/movielens/) dataset (100,836 ratings from 610 users across 9,724 movies).
2. **Preprocessing:** ratings are merged with movie titles, then filtered to movies with at least 60 ratings to reduce sparsity, leaving 35,080 ratings.
3. **Train/test split:** 80/20 split with a fixed random seed (42) for reproducible results.
4. **User–item matrix:** training ratings are pivoted into a users × movies matrix and stored as a compressed sparse row (CSR) matrix.
5. **Model:** scikit-learn's `NearestNeighbors` with cosine similarity finds the users most similar to a target user.
6. **Recommendations:** a user's top-N list is built from the average ratings of their nearest neighbours, excluding movies they have already rated.

## Results

The model was tuned against a baseline (fewer neighbours, lower popularity threshold):

| Model     | Neighbours (k) | Popularity threshold | MAE   | RMSE  |
|-----------|----------------|----------------------|-------|-------|
| Baseline  | 5              | 40                   | 0.781 | 1.021 |
| Optimised | 20             | 60                   | 0.716 | 0.930 |

Tuning reduced RMSE by about 9%. The optimised model reaches **Precision@20 of 0.15** and **Recall@20 of 0.46**: on average it retrieves around 46% of the movies each user rated 4 stars or higher.

## Tech stack

Python · Pandas · NumPy · scikit-learn · SciPy · Matplotlib · Jupyter / Google Colab

## Running the notebook

The easiest way is Google Colab:

1. Open [Google Colab](https://colab.research.google.com/) and choose **File → Upload notebook**.
2. Upload `movie_recommender.ipynb`.
3. Choose **Runtime → Run all**. The first cells download the dataset automatically.

To run it locally instead:

```bash
pip install -r requirements.txt
jupyter notebook movie_recommender.ipynb
```

The first cells use `wget` and `unzip`, which work in Colab, macOS and Linux. On Windows, download the dataset manually from the link above and unzip it next to the notebook.

## Possible improvements

- Item-based KNN as a comparison
- Matrix factorisation methods such as SVD
- A hybrid model that also uses movie genres

## Dataset credit

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems 5(4). The dataset is not included in this repository; the notebook downloads it from GroupLens.
