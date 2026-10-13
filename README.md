# Detecting Paper-Mill Articles with Heterogeneous Graph Neural Networks
 
Bachelor thesis · Department of Computer Engineering & Informatics, University of Patras · July 2026
Author: Aristea Tempelopoulou · Supervisor: Prof. Dimitrios Kosmopoulos
 
## Overview
Paper mills are organisations that mass-produce fraudulent scientific articles.
This project predicts whether an article is likely to be retracted, using
Machine Learning on a heterogeneous graph that combines the article text,
bibliometric statistics and the citation/authorship network.
The model is a Hetero-GraphSAGE that also supports inductive learning,
i.e. classification of articles unseen during training.
 
## Dataset
A new balanced dataset of 6,190 articles (3,095 retracted / 3,095 non-retracted)
was built from Retraction Watch and OpenAlex, with SPECTER embeddings of the abstracts.
 
The dataset is **not included** in this repository, because it contains retracted articles.
It is available upon request: aristea15.temp@gmail.com
To run the code, place `dataset_latest.csv` and `specter_embeddings_latest.pt`
in a `data/` folder at the root of the repository.
 
## Method
- **Graph:** nodes are papers and authors. Edges represent citations, authorship,
  co-authorship and bibliographic coupling.
- **Node features:** SPECTER abstract embeddings, citation percentile, citation count, temporal citation features, ORCID presence.
- **Model:** 2-layer Hetero-GraphSAGE (PyTorch Geometric), with batch normalisation, dropout, early stopping and a learning-rate scheduler.
- **Evaluation:** stratified 70/10/20 split, with measures against data leakage
  (train-only normalisation, train-only bibliographic coupling in the inductive setup).
  Metrics: accuracy, precision, recall, F1, ROC-AUC, average precision.
 
## Experiments
Set `EXPERIMENT = '...'` at the top of the notebook.
 
| Name | Description |
|------|-------------|
| `baseline` | Paper and author nodes |
| `cosine` | Extra edges from semantic (cosine) similarity |
| `university` | Added university nodes |
| `country` | Added country nodes |
| `journal` | Added journal nodes |
| `virtual` | Added virtual (global) node |
| `inductive` | Mini-batch training with NeighborLoader, testing on unseen papers |
 
## Results (test set)
 
| Experiment | Architecture | Accuracy | Precision | Recall | F1 |
|------------|--------------|----------|-----------|--------|----|
| Baseline | Paper + author nodes | 90.79% | 89.39% | 92.57% | 90.95% |
| 1 – `cosine` | Semantic similarity edges | 90.71% | 89.50% | 92.25% | 90.85% |
| 2 – `university` | University nodes | 89.90% | 88.24% | 92.08% | 90.12% |
| 3 – `country` | Country nodes | 90.06% | 87.80% | 93.05% | 90.35% |
| 4 – `journal` | Journal nodes | 95.64% | 95.79% | 95.48% | 95.63% |
| 5 – `virtual` | Virtual node | 90.63% | 89.48% | 92.08% | 90.76% |
| 6 – `inductive` | Inductive learning (unseen papers) | 89.82% | 89.31% | 90.47% | 89.89% |
 
**Key findings**
- The baseline Hetero-GraphSAGE reaches F1 = 90.95%. Semantic edges, university/country nodes and a virtual node bring no significant gain (the virtual node suffers from oversmoothing).
- Journal nodes give the highest score (F1 = 95.63%), but this result should be interpreted with caution because of the structure of the dataset.
- In the harder inductive setting, where test papers are unseen during training,
  the model still reaches F1 = 89.89% (AP = 0.957).
 
Plots (loss curves, confusion matrix, ROC and PR curves) are produced in `results/`.
 
## How to run
1. Request the dataset (see above) and place it in `data/`.
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Open `notebooks/all_experiments.ipynb`, choose `EXPERIMENT`, and run all cells.
 
## Tech stack
Python, PyTorch, PyTorch Geometric, scikit-learn, pandas, NumPy, Matplotlib.
 
## Thesis
Nemertes link:https://nemertes.library.upatras.gr/items/e0cd1b49-711f-468b-84a9-b454dd7ce93d
Full text (in Greek): [thesis/thesis.pdf](thesis/thesis.pdf)
 
## Contact
aristea15.temp@gmail.com · www.linkedin.com/in/aristea-tempelopoulou-610315436
