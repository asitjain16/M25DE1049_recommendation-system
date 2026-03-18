# Tech Stack

## Language & Environment
- Python 3.8+
- Jupyter Notebooks (`.ipynb`)

## Core Libraries
| Library | Usage |
|---------|-------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations, matrix math |
| `scikit-learn` | Models, metrics, TF-IDF, cosine similarity, train/test split, StandardScaler |
| `scipy` | Sparse matrices (`csr_matrix`) |
| `matplotlib` | Visualizations and plots |
| `torch` (PyTorch) | Neural network models (Tasks 8, 12) |
| `shap` | SHAP explainability (Tasks 10, 13) |
| `lime` | LIME explainability (Tasks 12, 13) |
| `surprise` | SVD via Surprise library (Task 6) |

## Dependency Installation
Each notebook auto-installs its own dependencies via:
```python
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', ...])
```

## Common Commands
```bash
# Launch Jupyter
jupyter notebook

# Open the main combined notebook
jupyter notebook assignment3.ipynb

# Open a specific task notebook
jupyter notebook task1_content_based_filtering.ipynb
```

## Running Notebooks
- Use **Kernel → Restart & Run All** to execute a notebook from scratch
- Notebooks auto-download the dataset if `ml-latest-small/` folder is missing
- No separate build or test step — execution is cell-by-cell in Jupyter

## Evaluation Metrics
- **RMSE** — rating prediction error
- **Precision@K / Recall@K** — top-K recommendation quality (default K=10, threshold=4.0)
- **Hit Rate** — RL reward signal (rating ≥ 4.0 → positive)
- **Fidelity, Stability, Coverage** — explainability evaluation metrics
