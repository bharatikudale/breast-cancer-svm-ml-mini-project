# Breast Cancer Prediction using SVM (Mini Machine Learning Project)

This is a mini machine learning project that uses a **Support Vector Machine (SVM)** classifier
to predict whether a tumor is **malignant** or **benign** using the
**Breast Cancer Wisconsin dataset** from `scikit-learn`.

 Project Objectives

- Understand the workflow of a complete ML project:
  - Data loading
  - Train–test split
  - Feature scaling
  - Model training (SVM)
  - Model evaluation
  - Model saving
- Demonstrate supervised classification using a real medical dataset.

 Algorithm

- **Support Vector Machine (SVM)**
  - Kernel = RBF
  - Handles non-linear decision boundaries
  - Good performance on many classification tasks

 Project Structure


.
├── main.py
├── requirements.txt
├── README.md
└── models/
    ├── svm_breast_cancer.pkl  # saved SVM model 
    └── scaler.pkl             # saved StandardScaler 
