"""
Downloads model from Databricks MLflow Registry
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
    """
    try:
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")
        
        if not databricks_host or not databricks_token:
            raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set")
        
        # Clean host - remove protocol if present
        databricks_host = databricks_host.replace("https://", "").replace("http://", "")
        
        # Set tracking URI
        tracking_uri = f"https://{databricks_host}"
        
        # CRITICAL: Set token in environment for MLflow to use
        os.environ['DATABRICKS_HOST'] = databricks_host
        os.environ['DATABRICKS_TOKEN'] = databricks_token
        
        # Also set as MLflow environment variable
        os.environ['MLFLOW_TRACKING_TOKEN'] = databricks_token
        
        mlflow.set_tracking_uri(tracking_uri)
        
        logger.info(f"Connecting to: {tracking_uri}")
        logger.info(f"Model: {model_name}, Stage: {model_stage}")
        
        # Create client
        client = MlflowClient(tracking_uri=tracking_uri)
        
        # Get model version
        versions = client.get_latest_versions(model_name, stages=[model_stage])
        
        if not versions:
            raise ValueError(f"No model found in stage '{model_stage}'")
        
        model_version = versions[0]
        logger.info(f"Found model version: {model_version.version}")
        
        # Download model
        model_uri = f"models:/{model_name}/{model_stage}"
        logger.info(f"Downloading from: {model_uri}")
        
        os.makedirs(output_dir, exist_ok=True)
        
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
        
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Metadata saved")
        return model_path
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
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