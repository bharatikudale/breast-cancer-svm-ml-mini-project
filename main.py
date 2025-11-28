# main.py
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from joblib import dump


def load_data():
    
    Load the Breast Cancer dataset from scikit-learn.

    Returns:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        feature_names (list): Names of the features
        target_names (list): Names of the classes
    
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    return X, y, feature_names, target_names


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split the data into train and test sets and apply standard scaling.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        test_size (float): Fraction of data to use for testing
        random_state (int): Random state for reproducibility

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_svm(X_train, y_train, kernel="rbf", C=1.0, gamma="scale"):
   
    Train an SVM classifier.

    Args:
        X_train (np.ndarray): Scaled training features
        y_train (np.ndarray): Training labels
        kernel (str): Kernel type ('linear', 'rbf', etc.)
        C (float): Regularization parameter
        gamma (str or float): Kernel coefficient

    Returns:
        model (SVC): Trained SVM model
  
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_names):

    Evaluate the SVM model on test data.

    Prints:
        - Accuracy
        - Classification report

    Also:
        - Plots confusion matrix
        - Plots ROC curve
   
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, target_names)

    # ROC Curve
    plot_roc_curve(y_test, y_proba)


def plot_confusion_matrix(cm, target_names):
  
    Plot a confusion matrix using matplotlib.

    Args:
        cm (np.ndarray): Confusion matrix
        target_names (list): Class names
  
    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Tick marks
    ax.set_xticks(np.arange(len(target_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)

    # Rotate x-tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    plt.show()


def plot_roc_curve(y_test, y_proba):
    
    Plot ROC curve and calculate AUC.

    Args:
        y_test (np.ndarray): True labels
        y_proba (np.ndarray): Predicted probabilities for the positive class
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def save_model(model, scaler, save_dir="models", model_filename="svm_breast_cancer.pkl", scaler_filename="scaler.pkl"):
    
    Save the trained SVM model and scaler to disk.

    Args:
        model (SVC): Trained SVM model
        scaler (StandardScaler): Fitted scaler
        save_dir (str): Directory to save model files
        model_filename (str): Name for the model file
        scaler_filename (str): Name for the scaler file
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, model_filename)
    scaler_path = os.path.join(save_dir, scaler_filename)

    dump(model, model_path)
    dump(scaler, scaler_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


def main():
    print("=== Breast Cancer Prediction using SVM ===")

    # 1. Load data
    X, y, feature_names, target_names = load_data()
    print(f"Total samples: {X.shape[0]}")
    print(f"Total features: {X.shape[1]}")

    # 2. Split and scale
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(X, y)

    # 3. Train model
    model = train_svm(X_train_scaled, y_train)

    # 4. Evaluate model
    evaluate_model(model, X_test_scaled, y_test, target_names)

    # 5. Save model and scaler
    save_model(model, scaler)


if __name__ == "__main__":
    main()
