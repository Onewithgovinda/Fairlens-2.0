
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "loan_approval_dataset.csv"

df = pd.read_csv(DATA)
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)

df["loan_status"] = (
    df["loan_status"].astype(str).str.strip().str.lower()
    .map({"approved": 1, "rejected": 0})
)

df = df.dropna(subset=["loan_status"]).copy()
df["loan_status"] = df["loan_status"].astype(int)

X = df.drop(columns=["loan_status", "loan_id"])
y = df["loan_status"]

num = X.select_dtypes(include=np.number).columns.tolist()
cat = [c for c in X.columns if c not in num]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat)
])

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20,
    random_state=42, stratify=y
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

parameter_space = {
    "classifier__n_estimators": [200, 300, 500, 700],
    "classifier__max_depth": [None, 10, 15, 20, 25],
    "classifier__min_samples_split": [2, 5, 10, 15],
    "classifier__min_samples_leaf": [1, 2, 3, 5],
    "classifier__max_features": ["sqrt", "log2", 0.7],
    "classifier__criterion": ["gini", "entropy"]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_space,
    n_iter=25,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True
)

print("Starting Random Forest hyperparameter tuning...")
search.fit(X_dev, y_dev)

best_model = search.best_estimator_

pred = best_model.predict(X_test)
prob = best_model.predict_proba(X_test)[:, 1]
# Save held-out test predictions for fairness auditing
fairness_test = X_test.copy()

fairness_test["loan_status"] = y_test.to_numpy()
fairness_test["predicted_approved"] = pred

fairness_test.to_csv(
    ROOT / "fairness_test_predictions.csv",
    index=False
)

print("Saved fairness_test_predictions.csv")

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision_approved": float(
        precision_score(y_test, pred, zero_division=0)
    ),
    "recall_approved": float(
        recall_score(y_test, pred, zero_division=0)
    ),
    "f1_approved": float(
        f1_score(y_test, pred, zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_test, prob))
}

joblib.dump(best_model, ROOT / "approval_model_tuned.joblib")

results = {
    "best_parameters": search.best_params_,
    "best_cross_validation_f1": float(search.best_score_),
    "test_metrics": metrics,
    "confusion_matrix": confusion_matrix(
        y_test, pred, labels=[0, 1]
    ).tolist()
}

(ROOT / "tuning_results.json").write_text(
    json.dumps(results, indent=2),
    encoding="utf-8"
)

print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validation F1:")
print(search.best_score_)

print("\nFinal held-out test metrics:")
print(metrics)

print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, pred, labels=[0, 1]))

print("\nSaved approval_model_tuned.joblib")
print("Saved tuning_results.json")
