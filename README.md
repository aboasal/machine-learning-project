# Machine Learning Course Project

End-to-end machine learning project covering regression on numerical data, classification and clustering on images, classification on text, and a unified prediction GUI.

## Repository Structure

```text
machine-learning-project/
├── README.md                  # Setup instructions, dataset links, and run guide
├── requirements.txt           # Core Python dependencies
├── .gitignore                 # Excludes venvs, caches, IDE files, and raw datasets
├── part_a_numerical/          # Part A: Numerical Regression (Amr Ashour & Amr Adel)
├── part_b_images/             # Part B: Image Classification & Clustering (Abo Asal & Abdelrahman)
├── part_c_text/               # Part C: Text Classification (Omar Mohamed & Yasser)
├── gui/                       # Part D: Prediction GUI loading serialized models
├── models/                    # Shared serialized models and preprocessors (.joblib)
└── report/                    # Report drafts, metric logs, plots, and GUI screenshots
```

## Setup Instructions

1. Clone the repository and enter the project directory:
   ```bash
   git clone <REPOSITORY_URL>
   cd machine-learning-project
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Datasets

Raw datasets are excluded from Git via `.gitignore` and must be downloaded locally before running training pipelines:

- **Part A (Numerical Regression):** _TBD — add raw dataset name and source link here._
- **Part B (Images — Flowers Recognition):** [Flowers Recognition (Kaggle)](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) — place the 5 class folders (`daisy`, `dandelion`, `rose`, `sunflower`, `tulip`) under `flowers/` in the repository root.
- **Part C (Text Classification):** _TBD — add raw dataset name and source link here._

## How to Run

### Part A — Numerical Data (Regression)
- Navigate to `part_a_numerical/` and run the team's training script or notebook once implemented.

### Part B — Image Data (Classification & Clustering)
1. Ensure the raw Flowers Recognition images are placed under `flowers/<class_name>/`.
2. Run Abo Asal's preprocessing and Logistic Regression pipeline in `part_b_images/`:
   ```bash
   python part_b_images/train_logreg.py
   ```
3. Run Abdelrahman's KNN Classifier and K-Means clustering script/notebook using `part_b_images/flowers_splits.npz`.

### Part C — Text Data (Classification)
- Navigate to `part_c_text/` and run the team's text preprocessing and training script or notebook once implemented.

### Part D — GUI
Once all models are trained and saved in `models/`, launch the GUI (without retraining inside the GUI):
```bash
streamlit run gui/app.py
```

## Team Ownership & Shared `models/` Contract

All trained models and preprocessors must be serialized with `joblib` into `models/`:

| Part | Owners | Expected Files in `models/` |
| :--- | :--- | :--- |
| **Part A (Numerical)** | Amr Ashour & Amr Adel | `num_linreg.joblib`, `num_knn.joblib`, `num_scaler.joblib`, `num_encoders.joblib` |
| **Part B (Images — Preprocessing & LogReg)** | Abo Asal | `image_logreg.joblib`, `image_scaler.joblib`, `image_classes.joblib` |
| **Part B (Images — KNN & K-Means)** | Abdelrahman | `image_knn.joblib` |
| **Part C (Text)** | Omar Mohamed & Yasser | `text_logreg.joblib`, `text_svm.joblib`, `text_vectorizer.joblib` |
