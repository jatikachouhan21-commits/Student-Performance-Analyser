"""
ml_predictor.py
Machine Learning module for academic outcome forecasting and early intervention.
Provides explainable classification (Pass/Fail) and regression (Final Grade G3 prediction)
using Scikit-learn with clear separation between empirical facts and predictions.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Core feature columns used for early-semester prediction
PREDICTOR_FEATURES = [
    "G1", "G2", "studytime", "failures", "absences",
    "Medu", "Fedu", "traveltime", "famrel", "freetime", "goout", "health"
]

FEATURE_LABELS = {
    "G1": "Term 1 Score (G1)",
    "G2": "Term 2 Score (G2)",
    "studytime": "Weekly Study Time",
    "failures": "Past Class Failures",
    "absences": "Recorded Absences",
    "Medu": "Mother's Education",
    "Fedu": "Father's Education",
    "traveltime": "Commute Time",
    "famrel": "Family Relationship",
    "freetime": "Free Time",
    "goout": "Social Time with Friends",
    "health": "Health Status"
}


def train_performance_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains a Random Forest Classifier (Pass/Fail) and Random Forest Regressor (G3 Score).
    
    Returns:
        dict: Trained models, test evaluation metrics, and feature importances.
    """
    if len(df) < 30:
        return {"trained": False, "reason": "Insufficient cohort size for model training (requires >= 30 records)."}

    # Validate feature availability
    available_features = [f for f in PREDICTOR_FEATURES if f in df.columns]
    if len(available_features) < 4 or "G3" not in df.columns:
        return {"trained": False, "reason": "Mandatory predictive features missing."}

    X = df[available_features].copy()
    y_class = (df["G3"] >= 10).astype(int)  # 1 = Pass, 0 = Fail
    y_reg = df["G3"].astype(float)

    # 80/20 train/test split with fixed seed for reproducibility.
    # Split X, y_class, and y_reg synchronously to guarantee matching indices.
    (
        X_train,
        X_test,
        y_class_train,
        y_class_test,
        y_reg_train,
        y_reg_test,
    ) = train_test_split(
        X, y_class, y_reg, test_size=0.20, random_state=42, stratify=y_class
    )

    # 1. Classification Model (Pass/Fail)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_class_train)
    y_class_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_class_test, y_class_pred))
    f1 = float(f1_score(y_class_test, y_class_pred, average="weighted"))

    # 2. Regression Model (Score Forecast)
    reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    reg.fit(X_train, y_reg_train)
    y_reg_pred = reg.predict(X_test)
    r2 = float(r2_score(y_reg_test, y_reg_pred))
    rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)))

    # 3. Extract Feature Importance
    importances = clf.feature_importances_
    feat_df = pd.DataFrame({
        "feature": available_features,
        "feature_name": [FEATURE_LABELS.get(f, f) for f in available_features],
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    # Add predicted values to the entire dataset for interactive inspection
    df_with_preds = df.copy()
    all_preds_class = clf.predict(X)
    all_preds_proba = clf.predict_proba(X)[:, 1]
    all_preds_score = reg.predict(X)

    df_with_preds["predicted_status"] = np.where(all_preds_class == 1, "Predicted Pass", "Predicted Fail")
    df_with_preds["predicted_pass_prob"] = (all_preds_proba * 100.0).round(1)
    df_with_preds["predicted_G3"] = all_preds_score.round(1)

    return {
        "trained": True,
        "classifier": clf,
        "regressor": reg,
        "features": available_features,
        "metrics": {
            "accuracy": round(acc * 100.0, 1),
            "f1_score": round(f1, 2),
            "r2_score": round(r2, 2),
            "rmse": round(rmse, 2),
            "test_size": len(X_test)
        },
        "feature_importances": feat_df,
        "predictions_df": df_with_preds
    }


def predict_single_student(
    model_bundle: Dict[str, Any],
    student_record: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Produces an explainable prediction for an individual student record.
    Clearly tags the outcome as a forecasted ML estimate.
    """
    if not model_bundle.get("trained", False):
        return {"error": "Model bundle is not trained."}

    clf = model_bundle["classifier"]
    reg = model_bundle["regressor"]
    features = model_bundle["features"]

    # Extract feature values with safe defaults
    feature_vector = []
    for f in features:
        val = student_record.get(f, 0)
        feature_vector.append(float(val))

    X_in = pd.DataFrame([feature_vector], columns=features)
    prob_pass = float(clf.predict_proba(X_in)[0, 1]) * 100.0
    pred_pass = bool(clf.predict(X_in)[0] == 1)
    pred_score = float(reg.predict(X_in)[0])

    return {
        "predicted_pass": pred_pass,
        "predicted_status": "Predicted Pass" if pred_pass else "Predicted At-Risk / Fail",
        "pass_probability": round(prob_pass, 1),
        "predicted_G3_score": round(pred_score, 1),
        "confidence": "High" if abs(prob_pass - 50) > 30 else ("Moderate" if abs(prob_pass - 50) > 15 else "Low / Borderline"),
        "key_drivers": [
            f"Term 2 score of {student_record.get('G2', 0)}/20",
            f"Term 1 score of {student_record.get('G1', 0)}/20",
            f"Absence count: {student_record.get('absences', 0)}",
            f"Prior course failures: {student_record.get('failures', 0)}"
        ]
    }
