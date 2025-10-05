"""
Downloads model from Databricks MLflow Registry
Run during Docker build to embed model in container
"""

import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_name, model_stage="Staging", output_dir="./models"):
    """
    Download model from Databricks MLflow
    
    Args:
        model_name: Name in MLflow Registry (e.g., 'churn_prediction_model')
        model_stage: Stage to download from (Staging, Production, None)
        output_dir: Where to save model
    """
    try:
        # Get credentials from environment
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")
        
        if not databricks_host or not databricks_token:
            raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(f"databricks://{databricks_host}")
        os.environ['DATABRICKS_TOKEN'] = databricks_token
        
        logger.info(f"Connecting to Databricks: {databricks_host}")
        logger.info(f"Model: {model_name}, Stage: {model_stage}")
        
        # Get model version
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[model_stage])
        
        if not versions:
            raise ValueError(f"No model found in stage '{model_stage}'")
        
        model_version = versions[0]
        logger.info(f"Found model version: {model_version.version}")
        
        # Download model
        model_uri = f"models:/{model_name}/{model_stage}"
        logger.info(f"Downloading from: {model_uri}")
        
        model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=output_dir
        )
        
        logger.info(f"✅ Model downloaded to: {model_path}")
        
        # Save metadata
        import json
        metadata = {
            "model_name": model_name,
            "version": model_version.version,
            "stage": model_version.current_stage,
            "run_id": model_version.run_id
        }
        
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Metadata saved")
        return model_path
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        raise

if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME", "churn_prediction_model")
    model_stage = os.getenv("MODEL_STAGE", "Staging")
    output_dir = os.getenv("OUTPUT_DIR", "./models")
    
    logger.info("="*60)
    logger.info("MLflow Model Downloader")
    logger.info("="*60)
    
    download_model(model_name, model_stage, output_dir)
    
    logger.info("="*60)
    logger.info("✅ SUCCESS")
    logger.info("="*60)