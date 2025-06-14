import numpy as np
import pandas as pd
import pickle
import logging
import json
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/rajeshai2000/mlops-mini-project.mlflow")
dagshub.init(repo_owner='rajeshai2000', repo_name='mlops-mini-project', mlflow=True)
# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)

# ---------------- Load Model ----------------
def load_model(model_path: str):
    try:
        model = pickle.load(open(model_path, 'rb'))
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise

# ---------------- Load Test Data ----------------
def load_test_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logging.info(f"Test data loaded from {path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load test data from {path}: {e}")
        raise

# ---------------- Evaluate Model ----------------
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }

        logging.info(f"Evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        raise

# ---------------- Save Metrics ----------------
def save_metrics(metrics: Dict[str, float], file_path: str) -> None:
    try:
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save metrics: {e}")
        raise

# ---------------- Main Pipeline ----------------
def main():

    mlflow.set_experiment("dvc-pipeline")
    mlflow.start_run()
    try:
        model = load_model('./models/model.pkl')
        test_df = load_test_data('./data/processed/test_bow.csv')

        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, 'reports/metrics.json')

        # log metrics to mlflow

        for metric_name,metric_value in metrics.items():
            mlflow.log_metric(metric_name,metric_value)

        if hasattr(model,'get_params'):
            params = model.get_params()
            for param_name,param_value in params.items():
                mlflow.log_param(param_name,param_value)

        mlflow.sklearn.log_model(model, "model")

        logging.info("Model evaluation completed successfully.")
    except Exception as e:
        logging.exception("Model evaluation pipeline failed.")

# ---------------- Entry Point ----------------
if __name__ == '__main__':
    main()