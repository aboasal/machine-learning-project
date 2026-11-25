# Shared Serialized Models (`models/`)

All parts save their trained scikit-learn models, scalers, encoders, vectorizers, and class lists here using `joblib` so the Part D GUI can load them directly without retraining.

| Part | Expected Files |
| :--- | :--- |
| **Part A (Numerical)** | `num_linreg.joblib`, `num_knn.joblib`, `num_scaler.joblib`, `num_encoders.joblib` |
| **Part B (Images — LogReg)** | `image_logreg.joblib`, `image_scaler.joblib`, `image_classes.joblib` |
| **Part B (Images — KNN)** | `image_knn.joblib` |
| **Part C (Text)** | `text_logreg.joblib`, `text_svm.joblib`, `text_vectorizer.joblib` |
