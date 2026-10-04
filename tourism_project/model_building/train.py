import os
import pandas as pd
import joblib
import mlflow
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# MLflow
# ---------------------------------------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tourism-package-prediction")

# ---------------------------------------------------------
# Load train/test workflow artifacts
# ---------------------------------------------------------
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# ---------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------
numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Handle class imbalance.
negative = int((ytrain == 0).sum())
positive = int((ytrain == 1).sum())
scale_pos_weight = negative / positive if positive else 1.0

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
)

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model,
)

# ---------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

with mlflow.start_run(run_name="best-model-search") as parent_run:

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
    )

    grid_search.fit(Xtrain, ytrain)

    # Log every tuned combination for experimentation tracking.
    results = grid_search.cv_results_

    for i, params in enumerate(results["params"]):
        with mlflow.start_run(
            nested=True,
            run_name=f"trial_{i + 1}",
        ):
            mlflow.log_params(params)
            mlflow.log_metric(
                "mean_cv_recall",
                float(results["mean_test_score"][i]),
            )
            mlflow.log_metric(
                "std_cv_recall",
                float(results["std_test_score"][i]),
            )

    best_model = grid_search.best_estimator_

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric(
        "best_cv_recall",
        float(grid_search.best_score_),
    )
    mlflow.log_metric(
        "scale_pos_weight",
        float(scale_pos_weight),
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------
    threshold = 0.45

    train_probability = best_model.predict_proba(Xtrain)[:, 1]
    test_probability = best_model.predict_proba(Xtest)[:, 1]

    train_prediction = (
        train_probability >= threshold
    ).astype(int)

    test_prediction = (
        test_probability >= threshold
    ).astype(int)

    train_report = classification_report(
        ytrain,
        train_prediction,
        output_dict=True,
        zero_division=0,
    )

    test_report = classification_report(
        ytest,
        test_prediction,
        output_dict=True,
        zero_division=0,
    )

    mlflow.log_metrics({
        "train_accuracy": float(train_report["accuracy"]),
        "train_precision": float(train_report["1"]["precision"]),
        "train_recall": float(train_report["1"]["recall"]),
        "train_f1": float(train_report["1"]["f1-score"]),
        "test_accuracy": float(test_report["accuracy"]),
        "test_precision": float(test_report["1"]["precision"]),
        "test_recall": float(test_report["1"]["recall"]),
        "test_f1": float(test_report["1"]["f1-score"]),
    })

    print("Best parameters:")
    print(grid_search.best_params_)
    print("\nTest classification report:")
    print(
        classification_report(
            ytest,
            test_prediction,
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # Save model for deployment
    # -----------------------------------------------------
    model_path = "tourism_project/deployment/model.joblib"

    os.makedirs(
        "tourism_project/deployment",
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        model_path,
    )

    mlflow.log_artifact(
        model_path,
        artifact_path="model",
    )

    print(f"Model saved to: {model_path}")
