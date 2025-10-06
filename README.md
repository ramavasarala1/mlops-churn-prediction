# Enterprise MLOps Pipeline on Azure

A production-grade MLOps implementation showcasing modern machine learning operations practices with automated model training, deployment, and monitoring on Azure infrastructure.

## Overview

This project demonstrates a complete end-to-end MLOps pipeline for a customer churn prediction model, featuring automated CI/CD, infrastructure as code, container orchestration, and comprehensive monitoring.

## Key Features

- **Infrastructure as Code**: Terraform-managed Azure resources (AKS, ACR, Databricks)
- **Automated ML Pipeline**: Databricks-based model training with MLflow tracking
- **Container Orchestration**: Kubernetes deployment on Azure Kubernetes Service (AKS)
- **CI/CD Automation**: GitHub Actions for continuous integration and deployment
- **Model Registry**: MLflow model versioning with stage transitions (Staging → Production)
- **Monitoring & Observability**: Prometheus metrics and Grafana dashboards
- **API Deployment**: RESTful model serving with FastAPI/Uvicorn

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GitHub Actions                          │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │   CI Build   │→ │ CD Staging   │→ │ CD Production      │   │
│  │  (on PR)     │  │ (on merge)   │  │ (manual approval)  │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Azure Container Registry                     │
│         Docker Images: churn-model:{commit-sha}                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Azure Kubernetes Service (AKS)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐     │
│  │ Dev Namespace│  │Staging (auto)│  │Production (gate) │     │
│  └──────────────┘  └──────────────┘  └──────────────────┘     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Databricks Workspace                         │
│  ┌────────────────────┐         ┌──────────────────────┐       │
│  │  Model Training    │ ──────→ │   MLflow Registry    │       │
│  │  (Notebooks/Jobs)  │         │  (Version Control)   │       │
│  └────────────────────┘         └──────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Cloud Infrastructure
- **Azure Kubernetes Service (AKS)**: Container orchestration
- **Azure Container Registry (ACR)**: Private Docker registry
- **Azure Databricks**: Distributed ML training platform
- **Terraform**: Infrastructure provisioning

### ML & Data
- **MLflow**: Experiment tracking and model registry
- **Python 3.10**: Core ML development
- **Scikit-learn / XGBoost**: ML frameworks
- **Pandas / NumPy**: Data processing

### DevOps & Deployment
- **GitHub Actions**: CI/CD pipelines
- **Docker**: Containerization
- **Kubernetes**: Orchestration with namespaces (dev/staging/production)
- **FastAPI / Uvicorn**: Model serving API

### Monitoring
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Kubernetes Probes**: Health checks (liveness/readiness)

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── deploy_model_api.yaml      # CI/CD pipeline
├── infrastructure/
│   └── terraform/                     # IaC for Azure resources
├── src/
│   ├── model-api/
│   │   ├── app.py                     # FastAPI model serving
│   │   ├── Dockerfile                 # Container definition
│   │   └── requirements.txt           # Python dependencies
│   └── training/
│       └── notebooks/                 # Databricks training notebooks
├── scripts/
│   └── model_loader.py                # MLflow model downloader
├── kubernetes/
│   └── manifests/
│       └── deployment.yaml            # K8s deployment manifests
└── README.md
```

## CI/CD Pipeline

### Continuous Integration (PR)
Triggered on pull requests to `main`:
1. Downloads latest model from MLflow
2. Builds Docker image tagged as `pr-{number}`
3. Pushes to ACR for validation
4. **Does not deploy** - validation only

### Continuous Deployment (Staging)
Triggered on merge to `main`:
1. Gets latest model version from Databricks
2. **Transitions model to "Staging"** in MLflow Registry
3. Builds official Docker image tagged with Git commit SHA
4. Deploys to AKS staging namespace
5. Waits for rollout confirmation

### Production Deployment (Manual Gate)
Requires manual approval:
1. **Waits for human approval** via GitHub Environment protection
2. **Transitions model to "Production"** in MLflow Registry
3. Deploys same Docker image (commit SHA) to production namespace
4. Waits for rollout confirmation
5. Creates Git tag: `prod-{commit-sha}`

## Getting Started

### Prerequisites

- Azure subscription with appropriate permissions
- GitHub repository with Actions enabled
- Databricks workspace with MLflow configured
- Terraform installed locally (for infrastructure setup)
- kubectl configured for AKS access

### Required GitHub Secrets

Configure these in your repository settings:

```bash
AZURE_CREDENTIALS          # Azure service principal credentials (JSON)
ACR_NAME                   # Azure Container Registry name
RESOURCE_GROUP             # Azure resource group name
AKS_CLUSTER                # AKS cluster name
DATABRICKS_HOST            # Databricks workspace URL (without https://)
DATABRICKS_TOKEN           # Databricks personal access token
```

### Environment Setup

1. **Deploy Infrastructure**:
   ```bash
   cd infrastructure/terraform
   terraform init
   terraform plan
   terraform apply
   ```

2. **Configure GitHub Environments**:
   - Create `staging` environment (optional protection rules)
   - Create `production` environment with required reviewers

3. **Train Initial Model**:
   - Run training notebook in Databricks
   - Register model in MLflow Registry

4. **Deploy via CI/CD**:
   ```bash
   # Create feature branch
   git checkout -b feature/my-changes
   
   # Make changes and commit
   git add .
   git commit -m "Your changes"
   git push origin feature/my-changes
   
   # Create PR → CI validates
   # Merge PR → CD deploys to staging
   # Approve → Deploys to production
   ```

## Model Serving API

### Endpoints

- `GET /health` - Health check (liveness probe)
- `GET /ready` - Readiness check
- `POST /predict` - Model inference
- `GET /metrics` - Prometheus metrics

### Example Request

```bash
curl -X POST http://<service-url>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "tenure": 12,
      "monthly_charges": 50.5,
      "total_charges": 606.0
    }
  }'
```

## Monitoring

### Kubernetes Health Checks

- **Liveness Probe**: `/health` endpoint (detects crashed containers)
- **Readiness Probe**: `/ready` endpoint (controls traffic routing)

### Metrics

Prometheus scrapes metrics from `/metrics` endpoint:
- Request latency
- Prediction counts
- Error rates
- Model version info

### Dashboards

Grafana dashboards visualize:
- API performance metrics
- Resource utilization (CPU/Memory)
- Model prediction distribution
- Deployment history

## Best Practices Implemented

✅ **Immutable Deployments**: Git SHA-based image tags  
✅ **Progressive Delivery**: Dev → Staging → Production  
✅ **Approval Gates**: Manual production deployment review  
✅ **Health Checks**: Kubernetes liveness/readiness probes  
✅ **Model Versioning**: MLflow stage transitions with audit trail  
✅ **Secrets Management**: GitHub Secrets for credentials  
✅ **Resource Limits**: CPU/Memory constraints in K8s  
✅ **Non-root Containers**: Security-hardened Docker images  
✅ **Rollback Capability**: K8s revision history maintained  

## Troubleshooting

### CI Build Fails
- Check model exists in Databricks MLflow Registry
- Verify `DATABRICKS_HOST` and `DATABRICKS_TOKEN` secrets
- Review GitHub Actions logs for specific errors

### Deployment Fails
- Verify AKS credentials: `az aks get-credentials`
- Check namespace exists: `kubectl get namespaces`
- Review pod logs: `kubectl logs -n <namespace> <pod-name>`

### Model Download Issues
- Ensure MLflow tracking URI is correct
- Verify Databricks token has registry permissions
- Check model version exists in specified stage

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built following Azure Well-Architected Framework principles
- MLOps practices inspired by Google's MLOps Maturity Model
- Kubernetes deployment patterns from CNCF best practices

---

**Status**: 🚀 Production Ready

For questions or issues, please open a GitHub issue.