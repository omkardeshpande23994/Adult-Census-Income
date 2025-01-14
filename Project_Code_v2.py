# Import necessary libraries
import timeit
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import IterativeImputer
from lightgbm import LGBMClassifier
import shap  # Explainability library
import optuna  # For hyperparameter tuning

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Start timer for execution time
start_time = timeit.default_timer()

# Load dataset
data_path = "data.csv"  # Update this path with the actual dataset path
data = pd.read_csv(data_path)

# Quick exploration of the data
print("Dataset Overview:")
print(data.head())
print("\nDataset Shape:", data.shape)

# Encode target variable
data["income"] = data["income"].map({"<=50K": 0, ">50K": 1})

# Dropping redundant column
data.drop(columns=["education"], inplace=True)

# Handle missing values using Iterative Imputer
data.replace(["?", " ", "NULL"], np.nan, inplace=True)

imputer = IterativeImputer(max_iter=10, random_state=0)
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Separate features and target variable
X = data_imputed.drop(columns=["income"])
y = data_imputed["income"].astype(int)

# Feature encoding and scaling with a ColumnTransformer
categorical_features = X.select_dtypes(include=["object"]).columns
numerical_features = X.select_dtypes(include=["number"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Principal Component Analysis (PCA) for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% variance
X_train_pca = pca.fit_transform(X_train_resampled)
X_test_pca = pca.transform(X_test)

# Define models for evaluation
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(),
    "LightGBM": LGBMClassifier(random_state=42),
}

# Evaluate models using cross-validation
print("\nModel Evaluation:")
for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"{name}: Mean Accuracy = {cv_scores.mean():.4f}")

# Hyperparameter optimization using Optuna
def objective(trial):
    # Example hyperparameter tuning for Random Forest
    n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 30)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy")
    return cv_scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Train optimized Random Forest model
best_params = study.best_params
optimized_model = RandomForestClassifier(**best_params, random_state=42)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", optimized_model)])
pipeline.fit(X_train, y_train)

# Evaluate optimized model
y_pred = pipeline.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]))

# Feature importance using SHAP
explainer = shap.TreeExplainer(optimized_model)
shap_values = explainer.shap_values(X_test_pca)
shap.summary_plot(shap_values, X_test_pca)

# End timer
end_time = timeit.default_timer()
print("\nTotal Execution Time (minutes):", (end_time - start_time) / 60)
