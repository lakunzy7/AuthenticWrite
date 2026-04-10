"""
Model Training Script for AuthentiWrite
Trains an ensemble model (Random Forest + Gradient Boosting + Logistic Regression)
using 28 stylometric features from features.py and TF-IDF text features.

Run: python train_model.py
"""

import pickle
import numpy as np
import nltk
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack, csr_matrix

from features import extract_features, FEATURE_NAMES
from training_data import load_training_data

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


def build_feature_matrix(texts):
    """Extract 28 stylometric features for all texts.

    Returns:
        np.ndarray of shape (n_samples, 28)
    """
    print(f"  Extracting stylometric features from {len(texts)} texts...")
    matrix = []
    for i, text in enumerate(texts):
        matrix.append(extract_features(text))
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(texts)} done")
    return np.array(matrix)


def train_model():
    """Train and save the ensemble model."""
    # --- Load data ---
    print("Loading training data...")
    texts, labels = load_training_data()
    y = np.array(labels)
    print(f"  {len(texts)} samples: {sum(1 for l in labels if l == 0)} human, "
          f"{sum(1 for l in labels if l == 1)} AI")

    # --- Stylometric features (28) ---
    print("\nStep 1: Stylometric features...")
    X_stylo = build_feature_matrix(texts)
    print(f"  Shape: {X_stylo.shape}")

    # Scale stylometric features
    scaler = StandardScaler()
    X_stylo_scaled = scaler.fit_transform(X_stylo)

    # --- TF-IDF features ---
    print("\nStep 2: TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_tfidf = vectorizer.fit_transform(texts)
    print(f"  Shape: {X_tfidf.shape}")

    # --- Combine features ---
    X_combined = hstack([csr_matrix(X_stylo_scaled), X_tfidf])
    print(f"\nCombined feature matrix: {X_combined.shape}")

    # --- Build ensemble ---
    print("\nStep 3: Training ensemble model...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
    )

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft',
        weights=[2, 2, 1],
    )

    # --- Cross-validation ---
    print("\nStep 4: Cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(ensemble, X_combined, y, cv=cv, scoring='accuracy')
    print(f"  5-Fold CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"  Per-fold: {[f'{s:.4f}' for s in scores]}")

    # --- Train on full data ---
    print("\nStep 5: Training final model on all data...")
    ensemble.fit(X_combined, y)

    # Print classification report on training data
    y_pred = ensemble.predict(X_combined)
    print("\nTraining set classification report:")
    print(classification_report(y, y_pred, target_names=['Human', 'AI']))

    # --- Feature importance (from Random Forest) ---
    rf_model = ensemble.named_estimators_['rf']
    importances = rf_model.feature_importances_
    stylo_importances = importances[:len(FEATURE_NAMES)]
    top_features = sorted(
        zip(FEATURE_NAMES, stylo_importances),
        key=lambda x: x[1],
        reverse=True,
    )
    print("Top 10 stylometric features:")
    for name, imp in top_features[:10]:
        print(f"  {name}: {imp:.4f}")

    # --- Save artifacts ---
    print("\nSaving model artifacts...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    print("  model.pkl saved")

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("  vectorizer.pkl saved")

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("  scaler.pkl saved")

    # --- Summary ---
    print("\n" + "=" * 50)
    print("MODEL TRAINING COMPLETE")
    print("=" * 50)
    print(f"Training samples: {len(texts)}")
    print(f"Stylometric features: {len(FEATURE_NAMES)}")
    print(f"TF-IDF features: {X_tfidf.shape[1]}")
    print(f"Total features: {X_combined.shape[1]}")
    print(f"CV Accuracy: {scores.mean():.4f}")
    print(f"Artifacts: model.pkl, vectorizer.pkl, scaler.pkl")


if __name__ == '__main__':
    train_model()
