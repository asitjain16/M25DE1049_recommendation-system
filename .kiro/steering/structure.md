# Project Structure

```
/
├── assignment3.ipynb              # Main notebook: Tasks 8–13 combined
├── task1_content_based_filtering.ipynb
├── task2_user_profile_recommender.ipynb
├── task3_user_based_cf.ipynb
├── task4_item_based_cf.ipynb
├── task5_svd_recommendations.ipynb
├── task6_surprise_svd.ipynb
├── task7_hybrid_recommendation.ipynb
├── task8_neural_cbf.ipynb
├── task9_rl_recommendations.ipynb
├── task10_feature_explanations.ipynb
├── task11_neighborhood_explanations.ipynb
├── task12_lime_explanations.ipynb
├── task13_evaluating_explainability.ipynb
├── ml-latest-small/               # Dataset (auto-downloaded if missing)
│   ├── movies.csv                 # movieId, title, genres
│   ├── ratings.csv                # userId, movieId, rating, timestamp
│   ├── tags.csv                   # userId, movieId, tag, timestamp
│   └── links.csv                  # movieId, imdbId, tmdbId
└── README.md
```

## Conventions

### Notebook Structure
- Each notebook is self-contained: imports, data loading, model, evaluation, visualization
- Shared data loading and feature engineering is duplicated across notebooks (no shared module)
- `assignment3.ipynb` contains a "Shared Data Loading" section at the top that all tasks in that file depend on

### Data Patterns
- Dataset loaded from `ml-latest-small/` directory; auto-downloaded from GroupLens if absent
- Standard split: `train_test_split(..., test_size=0.2, random_state=42)`
- User-item matrix built as both `csr_matrix` (sparse) and `.toarray()` (dense) depending on use
- Movie genres one-hot encoded as `g_<GenreName>` columns
- User genre preferences stored as `user_pref_<GenreName>` columns

### Model Patterns
- PyTorch models use `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Training loops include early stopping with patience tracking
- Predictions clipped to `[0.5, 5.0]` rating range
- RL reward: `+1.0` if rating ≥ 4.0, `-1.0` otherwise

### ID Mappings
- `uid_map` / `mid_map`: original IDs → matrix indices
- `idx_to_uid` / `idx_to_mid`: reverse mappings
- `mid_to_title`: movieId → title string
