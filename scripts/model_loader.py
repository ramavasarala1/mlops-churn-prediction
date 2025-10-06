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

def download_model(model_name, model_version=None, model_stage=None, output_dir="./models"):
    """
    Download model from Databricks MLflow
    
    Args:
        model_name: Name of the registered model
        model_version: Specific version number (takes precedence over stage)
        model_stage: Stage name (e.g., 'Staging', 'Production')
        output_dir: Directory to save the model
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
        logger.info(f"Model: {model_name}")
        
        # Create client
        client = MlflowClient(tracking_uri=tracking_uri)
        
        # Determine model URI based on version or stage
        if model_version:
            # Download by specific version number
            logger.info(f"Downloading version: {model_version}")
            model_uri = f"models:/{model_name}/{model_version}"
            
            # Get version details
            model_version_details = client.get_model_version(model_name, model_version)
            current_stage = model_version_details.current_stage
            run_id = model_version_details.run_id
            
        elif model_stage:
            # Download by stage
            logger.info(f"Downloading from stage: {model_stage}")
            
            # Get model version from stage
            versions = client.get_latest_versions(model_name, stages=[model_stage])
            
            if not versions:
                raise ValueError(f"No model found in stage '{model_stage}'")
            
            model_version_details = versions[0]
            model_version = model_version_details.version
            current_stage = model_version_details.current_stage
            run_id = model_version_details.run_id
            
            model_uri = f"models:/{model_name}/{model_stage}"
            
        else:
            raise ValueError("Either model_version or model_stage must be specified")
        
        logger.info(f"Model version: {model_version}")
        logger.info(f"Current stage: {current_stage}")
        logger.info(f"Downloading from: {model_uri}")
        
        # Download model
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
            "version": str(model_version),
            "stage": current_stage,
            "run_id": run_id
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
    model_version = os.getenv("MODEL_VERSION")  # Optional: specific version
    model_stage = os.getenv("MODEL_STAGE", "Staging")  # Default: Staging
    output_dir = os.getenv("OUTPUT_DIR", "./models")
    
    logger.info("="*60)
    logger.info("MLflow Model Downloader")
    logger.info("="*60)
    
    download_model(model_name, model_version=model_version, model_stage=model_stage, output_dir=output_dir)
    
    logger.info("="*60)
    logger.info("✅ SUCCESS")
    logger.info("="*60)