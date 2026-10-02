# End-to-End MLOps Pipeline for Image Classification on AWS

A production-style MLOps system that fine-tunes a Vision Transformer (ViT) to classify recycling materials into 11 categories, deployed on AWS SageMaker with a fully automated retraining loop driven by real production traffic.

## What it does

- **Trains** a pretrained ViT (`google/vit-base-patch16-224-in21k`) on a custom recycling dataset using HuggingFace Transformers and SageMaker Training Jobs
- **Serves** predictions via a real-time SageMaker endpoint behind a public REST API (API Gateway + Lambda)
- **Monitors** every inference: low-confidence predictions are saved to S3 and automatically labeled by Claude (Amazon Bedrock, Converse API)
- **Retrains** automatically: once enough labeled examples accumulate, a SageMaker Pipeline is triggered with the new data
- **Gates deployment**: a candidate model only replaces the current one if it beats it on accuracy — zero-downtime blue/green swap

## Architecture

```
User → API Gateway → Lambda (api_handler)
                         │
                         ├─ SageMaker Endpoint (ViT inference)
                         │
                         └─ low confidence? → S3 (pending/)
                                                   │
                                              Bedrock / Claude
                                              (auto-labeling)
                                                   │
                                             SageMaker Pipeline
                                          ┌────────────────────┐
                                          │  TrainingStep      │
                                          │  ProcessingStep    │
                                          │  ConditionStep     │
                                          │  LambdaStep        │
                                          └────────────────────┘
```

## Tech Stack

| Area | Technologies |
|---|---|
| ML/DL | PyTorch, HuggingFace `transformers`, HuggingFace `datasets` |
| Model | Vision Transformer (ViT) — fine-tuned |
| AWS ML | SageMaker Training Jobs, Real-time Endpoints, Pipelines, Processing Jobs |
| Serverless | AWS Lambda, Amazon API Gateway |
| LLM-in-the-loop | Amazon Bedrock (Claude) — automatic labeling of low-confidence samples |
| Experiment Tracking | MLflow (SageMaker managed tracking server) |
| Storage | Amazon S3 |
| Monitoring | Amazon CloudWatch (custom EMF metrics) |

## Project Structure

```
scripts/
├── train.py                  # SageMaker training entry point (ViT fine-tuning)
├── inference.py              # SageMaker endpoint serving hooks
├── pipeline_definition.py    # SageMaker Pipeline DAG definition
├── compare_model_metrics.py  # Evaluation gate (runs as Processing Job)
├── lambda_deploy_fuction.py  # Deploy Lambda (blue/green endpoint swap)
├── lambda_api_handler.py     # API Lambda (inference + feedback loop)
└── requirements.txt

notebooks/
├── SM_Notebook.ipynb         # Bootstrap: EDA, initial training, pipeline registration
└── Deployment.ipynb          # Wiring: Lambda env vars, IAM policies, end-to-end test
```

## Setup

1. Create a SageMaker domain and execution role with the necessary permissions
2. Upload the dataset to S3
3. Deploy the two Lambda functions (`lambda_api_handler`, `lambda_deploy_fuction`)
4. Set up an API Gateway pointing to `lambda_api_handler`
5. Create an MLflow tracking server in SageMaker and set `MLFLOW_TRACKING_URI` in `pipeline_definition.py`
6. Run `SM_Notebook.ipynb` to register the pipeline, then `Deployment.ipynb` to wire everything together
