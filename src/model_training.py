import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


@st.cache_resource
def train_and_evaluate_model(df):
    """
    Trains the LightGBM model and returns the trained pipeline and evaluation metrics.
    It also saves the trained model pipeline to a file named 'delivery_model.joblib'.

    Args:
        df (pandas.DataFrame): The prepared dataframe for training.

    Returns:
        tuple: A tuple containing:
            - model_pipeline (sklearn.pipeline.Pipeline): The trained model.
            - evaluation_results (dict): A dictionary with performance metrics.
    """
    features = [
        "seller_state",
        "customer_state",
        "product_category_name_english",
        "price",
        "freight_value",
        "product_weight_g",
    ]
    target = "is_late"

    X = df[features]
    y = df[target]

    categorical_features = X.select_dtypes(include=["object", "category"]).columns
    numerical_features = X.select_dtypes(
        include=["int64", "float64", "float32", "int32"]
    ).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", lgb.LGBMClassifier(objective="binary", random_state=42)),
        ]
    )

    model_pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(
        y_test, y_pred, target_names=["On-Time (0)", "Late (1)"], output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    evaluation_results = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "report": report,
        "cm": cm,
        "roc_curve": (fpr, tpr),
    }

    # Save the trained model for the prediction page
    joblib.dump(model_pipeline, "delivery_model.joblib")

    return model_pipeline, evaluation_results
