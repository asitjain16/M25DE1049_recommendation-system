# Product

This is an academic assignment project (Assignment 3, MLBD course) implementing a full recommender systems pipeline on the MovieLens ml-latest-small dataset (100K+ ratings, 610 users, 9,742 movies).

The project progresses through six parts:
1. Content-Based Filtering (TF-IDF genre similarity)
2. Collaborative Filtering (User-CF, Item-CF)
3. Matrix Factorization (SVD, Surprise library)
4. Hybrid Recommender (weighted blend of CBF + CF + SVD)
5. Learning-Based (Neural CBF, Reinforcement Learning: ε-Greedy, UCB, Q-Learning)
6. Explainability (SHAP, Neighborhood-based, LIME, evaluation of explainability methods)

Each task is implemented in its own notebook (`task1_*.ipynb` through `task13_*.ipynb`). Tasks 8–13 are also combined in `assignment3.ipynb`.
