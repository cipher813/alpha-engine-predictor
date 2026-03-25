#!/usr/bin/env bash
# infrastructure/deploy.sh — Build and deploy the predictor Lambda container image.
#
# Uses the container image pattern because PyTorch (~350MB CPU-only) exceeds
# the Lambda zip layer limit. The image is pushed to ECR and the Lambda
# function code is updated in-place.
#
# Prerequisites:
#   - Docker installed and running
#   - AWS CLI configured (or IAM role on EC2/CodeBuild)
#   - ECR repo 'alpha-engine-predictor' exists in your account
#   - Lambda function 'alpha-engine-predictor-inference' already created
#
# Usage:
#   ./infrastructure/deploy.sh                # full deploy
#   ./infrastructure/deploy.sh --dry-run      # build image only, skip AWS push
#
# Environment variables (auto-detected if not set):
#   AWS_ACCOUNT_ID   — 12-digit AWS account ID (auto-detected via aws sts)
#   AWS_REGION       — defaults to us-east-1

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
ECR_REPO="alpha-engine-predictor"
LAMBDA_FUNCTION="alpha-engine-predictor-inference"
IMAGE_TAG="latest"
DRY_RUN=false

# Parse flags
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Resolve AWS identity ─────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
if [ -z "${AWS_ACCOUNT_ID:-}" ] && [ "$DRY_RUN" = false ]; then
  AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION" 2>/dev/null) \
    || { echo "ERROR: Could not auto-detect AWS_ACCOUNT_ID. Set it manually or configure AWS CLI."; exit 1; }
  echo "Auto-detected AWS_ACCOUNT_ID: $AWS_ACCOUNT_ID"
fi

# Move to repo root (script may be called from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "Working directory: $REPO_ROOT"

# ── Step 1: Build Docker image ────────────────────────────────────────────────
echo ""
echo "==> Building Docker image..."
docker build \
  --platform linux/amd64 \
  --provenance=false \
  --tag "${ECR_REPO}:${IMAGE_TAG}" \
  --file Dockerfile \
  .

echo "  Image built: ${ECR_REPO}:${IMAGE_TAG}"

if [ "$DRY_RUN" = true ]; then
  echo ""
  echo "==> DRY RUN: Skipping ECR push and Lambda update."
  echo "    Image built successfully as ${ECR_REPO}:${IMAGE_TAG}"
  exit 0
fi

# ── Step 2: Authenticate to ECR ───────────────────────────────────────────────
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

echo ""
echo "==> Authenticating to ECR (${ECR_REGISTRY})..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# ── Step 3: Tag and push image ────────────────────────────────────────────────
echo ""
echo "==> Tagging image: ${ECR_IMAGE}"
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_IMAGE}"

echo "==> Pushing to ECR (this may take a few minutes for first push)..."
docker push "${ECR_IMAGE}"
echo "  Pushed: ${ECR_IMAGE}"

# ── Step 4: Update Lambda function code ──────────────────────────────────────
echo ""
echo "==> Updating Lambda function: ${LAMBDA_FUNCTION}"
aws lambda update-function-code \
  --function-name "${LAMBDA_FUNCTION}" \
  --image-uri "${ECR_IMAGE}" \
  --region "${AWS_REGION}" \
  --output json \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('  FunctionArn:', d.get('FunctionArn','?')); print('  LastModified:', d.get('LastModified','?'))"

# ── Step 5: Wait for update to complete ──────────────────────────────────────
echo ""
echo "==> Waiting for Lambda update to complete..."
aws lambda wait function-updated \
  --function-name "${LAMBDA_FUNCTION}" \
  --region "${AWS_REGION}"

# ── Step 6: Publish version and update 'live' alias ──────────────────────────
echo ""
echo "==> Publishing Lambda version..."
VERSION=$(aws lambda publish-version \
  --function-name "${LAMBDA_FUNCTION}" \
  --query "Version" --output text \
  --region "${AWS_REGION}")
echo "  Published version: ${VERSION}"

echo "==> Updating 'live' alias → version ${VERSION}"
aws lambda update-alias \
  --function-name "${LAMBDA_FUNCTION}" \
  --name live \
  --function-version "${VERSION}" \
  --region "${AWS_REGION}" 2>/dev/null || \
aws lambda create-alias \
  --function-name "${LAMBDA_FUNCTION}" \
  --name live \
  --function-version "${VERSION}" \
  --region "${AWS_REGION}"

echo ""
echo "==> Deploy complete!"
echo "    Function : ${LAMBDA_FUNCTION}"
echo "    Version  : ${VERSION}"
echo "    Alias    : live → ${VERSION}"
echo "    Image    : ${ECR_IMAGE}"
echo ""
echo "To test:  aws lambda invoke --function-name ${LAMBDA_FUNCTION}:live --payload '{\"dry_run\": true}' --cli-binary-format raw-in-base64-out /tmp/response.json --region ${AWS_REGION} && cat /tmp/response.json"
echo "Rollback: bash infrastructure/rollback.sh"
