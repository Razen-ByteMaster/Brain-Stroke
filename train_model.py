import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from scipy.stats import randint, uniform

# imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42

# Load dataset
df = pd.read_csv(r"C:\VScode-Project\GRAD PROJECT\New\Brain_Stroke\brain_stroke.csv")

# Basic clean (optional) — ensure there are no stray index columns
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Features / target
X = df.drop("stroke", axis=1)
y = df["stroke"]

categorical_features = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",  # keep as in your dataset
    "smoking_status",
]
numerical_features = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]

# Preprocessing pipelines
categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
)

numerical_transformer = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

# Create imbalanced-learn pipeline: preprocessing -> SMOTE -> classifier
pipe = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ]
)

# Stratified split to preserve label distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Hyperparameter distributions for RandomizedSearchCV
param_distributions = {
    "clf__n_estimators": randint(100, 800),
    "clf__max_depth": [None, 5, 10, 20, 30],
    "clf__min_samples_split": randint(2, 11),
    "clf__min_samples_leaf": randint(1, 6),
    "clf__max_features": ["sqrt", "log2", 0.2, 0.5],
    "clf__class_weight": [None, "balanced"],
    "clf__bootstrap": [True, False],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rs = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=40,  # increase to 80-150 for more thorough search
    scoring="roc_auc",
    n_jobs=-1,
    cv=cv,
    random_state=RANDOM_STATE,
    verbose=1,
    refit=True,
)

# Fit search
print("Starting hyperparameter search...")
rs.fit(X_train, y_train)
print("Search done.")
print("Best params:", rs.best_params_)
print("Best CV ROC AUC:", rs.best_score_)

# Evaluate on test set
best_model = rs.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = (
    best_model.predict_proba(X_test)[:, 1]
    if hasattr(best_model, "predict_proba")
    else None
)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
rocauc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

print("\nTest set performance:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
if rocauc is not None:
    print(f"ROC AUC:   {rocauc:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and metadata
onehot_columns = (
    rs.best_estimator_.named_steps["preprocessor"]
    .transformers_[1][1]
    .named_steps["onehotencoder"]
    .get_feature_names_out(categorical_features)
)


model_data = {
    "model": best_model,
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "onehot_columns": onehot_columns,
}

joblib.dump(model_data, "stroke_model_best.pkl")
print("Saved best model to stroke_model_best.pkl")
