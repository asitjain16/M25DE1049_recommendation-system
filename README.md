```
## Name: Asit Jain
## Roll No: M25DE1049
# Assignment 3
# Subject: MLBD

```
#  Recommender Systems — Assignment 3

A complete recommender systems project built in Python using Jupyter Notebooks.  
We go from simple content-based filtering all the way to reinforcement learning and explainability.

Everything runs on the **MovieLens ml-latest-small** dataset — 100K+ ratings from 610 users across 9,742 movies.

---

##  Quick Start

**Step 1** — Make sure you have Python 3.8+ and Jupyter installed:
```bash
pip install jupyter
```

**Step 2** — Open any notebook:
```bash
jupyter notebook assignment3.ipynb
```

**Step 3** — Click **Kernel → Restart & Run All**

That's it. Each notebook auto-downloads the dataset and installs its own dependencies.

>  **Getting a `NameError`?** That means a cell was skipped. Just do **Kernel → Restart & Run All** to run everything from the top.

---

##  Files

| File | What It Does |
|------|-------------|
| `assignment3.ipynb` |  **Main file** — Tasks 8 to 13 combined in one notebook |
| `task1_content_based_filtering.ipynb` | TF-IDF genre similarity recommendations |
| `task2_user_profile_recommender.ipynb` | User profile from rating-weighted genres |
| `task3_user_based_cf.ipynb` | User-based collaborative filtering |
| `task4_item_based_cf.ipynb` | Item-based collaborative filtering |
| `task5_svd_recommendations.ipynb` | Manual SVD matrix factorization |
| `task6_surprise_svd.ipynb` | SVD using the Surprise library |
| `task7_hybrid_recommendation.ipynb` | Hybrid recommender (CBF + CF + SVD) |
| `task8_neural_cbf.ipynb` | Neural network content-based filtering |
| `task9_rl_recommendations.ipynb` | Reinforcement learning recommender |
| `task10_feature_explanations.ipynb` | SHAP feature explanations |
| `task11_neighborhood_explanations.ipynb` | Neighborhood-based CF explanations |
| `task12_lime_explanations.ipynb` | LIME for neural network explanations |
| `task13_evaluating_explainability.ipynb` | Comparing all explainability methods |
| `ml-latest-small/` | Dataset folder (auto-downloaded) |

---

##  How It All Flows

### Part 1 — Content-Based Filtering
> *"You liked Sci-Fi? Here are more Sci-Fi movies."*

- **Task 1** — Use TF-IDF on genres → find similar movies with cosine similarity
- **Task 2** — Build a user taste profile from their ratings → recommend matching movies

### Part 2 — Collaborative Filtering
> *"People like you also liked these movies."*

- **Task 3** — Find similar users → use their ratings to predict yours (User-CF)
- **Task 4** — Find similar movies → use your past ratings to predict new ones (Item-CF)

### Part 3 — Matrix Factorization
> *"Compress the giant rating matrix into hidden patterns."*

- **Task 5** — Decompose the user-item matrix with SVD → predict missing ratings
- **Task 6** — Same idea but using the Surprise library with cross-validation

### Part 4 — Hybrid System
> *"Why pick one method when you can combine them all?"*

- **Task 7** — Weighted blend of CBF + User-CF + Item-CF + SVD → best of all worlds

### Part 5 — Learning-Based *(in assignment3.ipynb)*
> *"Let the model learn and adapt over time."*

- **Task 8** — Train a neural network with two branches (user features + movie features)
- **Task 9** — Reinforcement learning: ε-Greedy bandit, UCB bandit, Q-Learning agent
  - Compares RL exploration vs traditional models (CF, SVD)

### Part 6 — Explainability *(in assignment3.ipynb)*
> *"Why was this movie recommended?"*

- **Task 10** — SHAP: which features (genre, year, popularity) drove the prediction?
- **Task 11** — Neighborhood: "Users like you rated this 4.5★" / "Because you liked Inception"
- **Task 12** — LIME: explain a neural network's decision with a simple local model
- **Task 13** — Compare all methods on fidelity, stability, coverage, speed, and bias

---

##  Dataset Details

**MovieLens ml-latest-small** by GroupLens Research (University of Minnesota)

| File | Contents |
|------|----------|
| `ratings.csv` | 100,836 ratings (0.5 to 5.0 stars) |
| `movies.csv` | 9,742 movies with titles and genres |
| `tags.csv` | 3,683 user-generated text tags |
| `links.csv` | IMDb and TMDb IDs for each movie |

- 610 users, each with at least 20 ratings
- Ratings from March 1996 to September 2018
- 19 genre categories (Action, Comedy, Drama, Sci-Fi, etc.)

> Harper, F.M. & Konstan, J.A. (2015). The MovieLens Datasets. ACM TiiS, 5(4), 19:1–19:19.

---

##  Dependencies

Everything installs automatically when you run a notebook. Here's what gets used:

| Package | Why |
|---------|-----|
| `pandas` | Load and wrangle data |
| `numpy` | Math and arrays |
| `scikit-learn` | Models, metrics, similarity |
| `scipy` | Sparse matrices |
| `matplotlib` | Charts and plots |
| `torch` | Neural networks (Tasks 8, 12) |
| `shap` | SHAP explanations (Tasks 10, 13) |
| `lime` | LIME explanations (Tasks 12, 13) |
| `surprise` | SVD library (Task 6) |

---

##  Metrics We Use

| Metric | What It Measures |
|--------|-----------------|
| **RMSE** | How far off our predicted ratings are from actual ratings |
| **Precision@K** | Out of K recommendations, how many did the user actually like? |
| **Recall@K** | Out of all movies the user liked, how many did we recommend? |
| **Hit Rate** | Did the RL agent recommend something the user rated ≥ 4? |
| **Fidelity** | Does the explanation actually reflect what the model cares about? |
| **Stability** | Do we get the same explanation if we run it again? |
| **Coverage** | What % of recommendations can we explain? |
