#!/usr/bin/env bash
# infrastructure/retrain.sh — Trigger monthly model retraining.
#
# Supports two modes:
#   --local        Run training locally (requires torch, parquet cache in data/cache/)
#   (default)      Submit a SageMaker Training Job
#
# SageMaker mode requires:
#   - AWS CLI configured with SageMaker permissions
#   - S3 bucket with parquet cache at s3://${S3_BUCKET}/predictor/price_cache/
#   - A SageMaker execution role
#   - The training Docker image pushed to ECR (same image as Lambda, or separate training image)
#
# Local mode is useful for development or when SageMaker is not configured.
# Training on a g4dn.xlarge (~$0.50/hr) typically completes in <30 minutes.
#
# Usage:
#   ./infrastructure/retrain.sh --local
#   ./infrastructure/retrain.sh                           # SageMaker (default)
#   ./infrastructure/retrain.sh --sagemaker-role ARN      # specify SageMaker role
#
# Environment variables:
#   AWS_REGION           — e.g. us-east-1
#   AWS_ACCOUNT_ID       — 12-digit AWS account ID
#   S3_BUCKET            — S3 bucket name (default: alpha-engine-research)
#   SAGEMAKER_ROLE_ARN   — IAM role for SageMaker (required unless --local)

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
LOCAL_MODE=false
S3_BUCKET="${S3_BUCKET:-alpha-engine-research}"
SAGEMAKER_ROLE_ARN="${SAGEMAKER_ROLE_ARN:-}"
ECR_REPO="alpha-engine-predictor"
JOB_NAME="alpha-engine-predictor-retrain-$(date +%Y%m%d-%H%M%S)"

# Parse flags
for arg in "$@"; do
  case "$arg" in
    --local) LOCAL_MODE=true ;;
    --sagemaker-role)
      shift
      SAGEMAKER_ROLE_ARN="$1"
      ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# Move to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "Working directory: $REPO_ROOT"
echo "Retraining job: $JOB_NAME"
echo ""

# ── Local training mode ───────────────────────────────────────────────────────
if [ "$LOCAL_MODE" = true ]; then
  echo "==> LOCAL MODE: Running training locally..."
  echo "    Data dir   : data/cache/"
  echo "    Output dir : checkpoints/"
  echo ""

  # Check parquet cache exists
  if [ ! -d "data/cache" ] || [ -z "$(ls data/cache/*.parquet 2>/dev/null)" ]; then
    echo "ERROR: No parquet files found in data/cache/"
    echo "       Run: python data/bootstrap_fetcher.py --output-dir data/cache"
    exit 1
  fi

  N_FILES=$(ls data/cache/*.parquet 2>/dev/null | wc -l | tr -d ' ')
  echo "  Found $N_FILES parquet files in data/cache/"
  echo ""

  echo "==> Starting training..."
  python train.py \
    --data-dir data/cache \
    --device cpu \
    --output checkpoints/

  echo ""
  echo "==> Training complete. Checkpoint saved to checkpoints/best.pt"
  echo ""
  echo "Next steps:"
  echo "  1. Review eval_report.json in checkpoints/"
  echo "  2. If metrics pass gates, upload to S3:"
  echo "     aws s3 cp checkpoints/best.pt s3://${S3_BUCKET}/predictor/weights/latest.pt"
  echo "     aws s3 cp checkpoints/best.pt s3://${S3_BUCKET}/predictor/weights/$(date +%Y%m%d).pt"
  echo "  3. Deploy updated Lambda: ./infrastructure/deploy.sh"
  exit 0
fi

# ── SageMaker Training Job ────────────────────────────────────────────────────
echo "==> SAGEMAKER MODE: Submitting training job..."

: "${AWS_REGION:?ERROR: AWS_REGION env var is required}"
: "${AWS_ACCOUNT_ID:?ERROR: AWS_ACCOUNT_ID env var is required}"

if [ -z "$SAGEMAKER_ROLE_ARN" ]; then
  echo "ERROR: SAGEMAKER_ROLE_ARN is required for SageMaker mode."
  echo "       Set: export SAGEMAKER_ROLE_ARN=arn:aws:iam::${AWS_ACCOUNT_ID}:role/SageMakerExecutionRole"
  echo "       Or run in local mode: ./infrastructure/retrain.sh --local"
  exit 1
fi

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
TRAINING_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:latest"
S3_OUTPUT="s3://${S3_BUCKET}/predictor/sagemaker-output/${JOB_NAME}/"
S3_DATA_URI="s3://${S3_BUCKET}/predictor/price_cache/"

echo "  Job name      : $JOB_NAME"
echo "  Training image: $TRAINING_IMAGE"
echo "  Input data    : $S3_DATA_URI"
echo "  Output        : $S3_OUTPUT"
echo "  Role          : $SAGEMAKER_ROLE_ARN"
echo ""

# Build the SageMaker training job JSON
JOB_CONFIG=$(cat <<EOF
{
  "TrainingJobName": "${JOB_NAME}",
  "AlgorithmSpecification": {
    "TrainingImage": "${TRAINING_IMAGE}",
    "TrainingInputMode": "File",
    "ContainerEntrypoint": ["python", "train.py"],
    "ContainerArguments": ["--data-dir", "/opt/ml/input/data/training", "--output", "/opt/ml/model"]
  },
  "RoleArn": "${SAGEMAKER_ROLE_ARN}",
  "InputDataConfig": [
    {
      "ChannelName": "training",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "${S3_DATA_URI}",
          "S3DataDistributionType": "FullyReplicated"
        }
      }
    }
  ],
  "OutputDataConfig": {
    "S3OutputPath": "${S3_OUTPUT}"
  },
  "ResourceConfig": {
    "InstanceType": "ml.g4dn.xlarge",
    "InstanceCount": 1,
    "VolumeSizeInGB": 30
  },
  "StoppingCondition": {
    "MaxRuntimeInSeconds": 3600
  },
  "Environment": {
    "S3_BUCKET": "${S3_BUCKET}"
  }
}
EOF
)

echo "==> Submitting SageMaker training job..."
aws sagemaker create-training-job \
  --cli-input-json "$JOB_CONFIG" \
  --region "${AWS_REGION}" \
  --output json \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('  TrainingJobArn:', d.get('TrainingJobArn','?'))"

echo ""
echo "==> Job submitted: $JOB_NAME"
echo ""
echo "Monitor progress:"
echo "  aws sagemaker describe-training-job --training-job-name $JOB_NAME --region ${AWS_REGION} \\"
echo "    --query 'TrainingJobStatus'"
echo ""
echo "View CloudWatch logs:"
echo "  aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix $JOB_NAME --follow"
echo ""
echo "After job completes, copy weights to S3:"
echo "  aws s3 cp ${S3_OUTPUT}output/model.tar.gz /tmp/ && tar -xzf /tmp/model.tar.gz -C /tmp/"
echo "  aws s3 cp /tmp/best.pt s3://${S3_BUCKET}/predictor/weights/latest.pt"
echo "  aws s3 cp /tmp/best.pt s3://${S3_BUCKET}/predictor/weights/$(date +%Y%m%d).pt"
