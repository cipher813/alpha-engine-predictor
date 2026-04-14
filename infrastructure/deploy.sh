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

# ── Stage proprietary config from alpha-engine-config ────────────────────────
# predictor.yaml is gitignored in this repo and lives in the private
# cipher813/alpha-engine-config repo alongside the configs for the other
# modules. Without staging it into the build context, the Dockerfile's
# `COPY config/ config/` only captures predictor.sample.yaml and the Lambda
# fails at import with FileNotFoundError — the silent 2026-04-13 regression.
# Falling back to the sample is explicitly rejected: shipping placeholder
# hyperparameters to production is worse than refusing to deploy.
#
# Local dev workflow is preserved: if config/predictor.yaml already exists
# (the dev has it in place in their laptop checkout), we use it as-is.
CONFIG_REPO_DIR="${CONFIG_REPO_DIR:-$(dirname "$REPO_ROOT")/alpha-engine-config}"
CONFIG_STAGED_FROM_REPO=0

if [ -f "config/predictor.yaml" ]; then
  echo "Using existing config/predictor.yaml (local dev workflow)"
else
  src="$CONFIG_REPO_DIR/predictor/predictor.yaml"
  if [ -f "$src" ]; then
    echo "Staging config/predictor.yaml from $src"
    cp "$src" config/predictor.yaml
    CONFIG_STAGED_FROM_REPO=1
  else
    echo "ERROR: config/predictor.yaml not found — tried:"
    echo "  config/predictor.yaml (local dev)"
    echo "  $src (config repo sibling)"
    echo "Hint: clone cipher813/alpha-engine-config as a sibling directory,"
    echo "      or set CONFIG_REPO_DIR=/path/to/alpha-engine-config"
    exit 1
  fi
fi

# ── Stage alpha-engine-lib into vendor/ ──────────────────────────────────────
# alpha-engine-lib is a private repo that provides BasePreflight +
# setup_logging. The Dockerfile installs it via local pip path because
# git+https inside Docker needs a PAT secret (more ceremony than this
# staging approach). Pattern matches the config staging above.
LIB_REPO_DIR="${LIB_REPO_DIR:-$(dirname "$REPO_ROOT")/alpha-engine-lib}"
LIB_STAGED_FROM_REPO=0

if [ -d "vendor/alpha-engine-lib" ]; then
  echo "Using existing vendor/alpha-engine-lib (local dev workflow)"
else
  if [ -d "$LIB_REPO_DIR" ]; then
    echo "Staging vendor/alpha-engine-lib from $LIB_REPO_DIR"
    mkdir -p vendor
    cp -R "$LIB_REPO_DIR" vendor/alpha-engine-lib
    LIB_STAGED_FROM_REPO=1
  else
    echo "ERROR: alpha-engine-lib not found — tried:"
    echo "  vendor/alpha-engine-lib (local dev)"
    echo "  $LIB_REPO_DIR (sibling checkout)"
    echo "Hint: clone cipher813/alpha-engine-lib as a sibling directory,"
    echo "      or set LIB_REPO_DIR=/path/to/alpha-engine-lib"
    exit 1
  fi
fi

# Cleanup staged artifacts on exit so a failed deploy doesn't leave stray
# files in a dev laptop checkout.
cleanup_staged_artifacts() {
  if [ "$CONFIG_STAGED_FROM_REPO" = "1" ] && [ -f config/predictor.yaml ]; then
    rm -f config/predictor.yaml
  fi
  if [ "$LIB_STAGED_FROM_REPO" = "1" ] && [ -d vendor/alpha-engine-lib ]; then
    rm -rf vendor/alpha-engine-lib
    # Remove vendor/ if it's now empty (we created it).
    rmdir vendor 2>/dev/null || true
  fi
}
trap cleanup_staged_artifacts EXIT

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

# ── Step 4b: Sync env vars from master .env ─────────────────────────────────
# Master .env lives in alpha-engine-data; fall back to local .env
LAMBDA_ENV_FILE="$(dirname "$REPO_ROOT")/alpha-engine-data/.env"
if [ ! -f "$LAMBDA_ENV_FILE" ]; then
  LAMBDA_ENV_FILE="$REPO_ROOT/.env"
fi
if [ -f "$LAMBDA_ENV_FILE" ]; then
  LAMBDA_ENV_JSON=$(python3 -c "
import json
env = {}
with open('$LAMBDA_ENV_FILE') as f:
    for line in f:
        line = line.strip()
        if line == '# LAMBDA_SKIP':
            break
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('\"', \"'\"):
            val = val[1:-1]
        if key and val:
            env[key] = val
if env:
    print(json.dumps({'Variables': env}))
else:
    print('')
")
  if [ -n "$LAMBDA_ENV_JSON" ]; then
    echo ""
    echo "==> Syncing env vars from $LAMBDA_ENV_FILE"
    echo "  Keys: $(echo "$LAMBDA_ENV_JSON" | python3 -c "import sys,json; print(', '.join(json.load(sys.stdin).get('Variables',{}).keys()))")"
    aws lambda wait function-updated --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}" 2>/dev/null || sleep 5
    aws lambda update-function-configuration \
      --function-name "${LAMBDA_FUNCTION}" \
      --environment "$LAMBDA_ENV_JSON" \
      --region "${AWS_REGION}" > /dev/null
  fi
else
  echo "  WARNING: No .env file found — Lambda env vars not updated"
fi

# ── Step 5: Wait for update to complete ──────────────────────────────────────
echo ""
echo "==> Waiting for Lambda update to complete..."
aws lambda wait function-updated \
  --function-name "${LAMBDA_FUNCTION}" \
  --region "${AWS_REGION}"

# ── Step 6: Publish version ──────────────────────────────────────────────────
echo ""
echo "==> Publishing Lambda version..."
VERSION=$(aws lambda publish-version \
  --function-name "${LAMBDA_FUNCTION}" \
  --query "Version" --output text \
  --region "${AWS_REGION}")
echo "  Published version: ${VERSION}"

# ── Step 7: Canary against the new version (NOT live) ────────────────────────
# Invoke the version directly so a broken image cannot reach the live alias.
# If the canary fails, live keeps pointing at the prior good version.
echo ""
echo "==> Running canary invocation against :${VERSION} (dry_run=true)..."
CANARY_OUT=$(mktemp)
aws lambda invoke \
  --function-name "${LAMBDA_FUNCTION}:${VERSION}" \
  --payload '{"dry_run": true}' \
  --cli-binary-format raw-in-base64-out \
  --cli-read-timeout 300 \
  --region "${AWS_REGION}" \
  "$CANARY_OUT" > /dev/null

CANARY_STATUS=$(python3 -c "import json; d=json.load(open('$CANARY_OUT')); print(d.get('statusCode', 0))" 2>/dev/null || echo "0")
CANARY_ERR=$(python3 -c "import json; d=json.load(open('$CANARY_OUT')); print(d.get('FunctionError',''))" 2>/dev/null || echo "")
rm -f "$CANARY_OUT"

if [ "$CANARY_STATUS" != "200" ] || [ -n "$CANARY_ERR" ]; then
  echo ""
  echo "ERROR: Canary failed (statusCode=$CANARY_STATUS, FunctionError=$CANARY_ERR) — refusing to promote :${VERSION} to live."
  echo "       Live alias is unchanged. Investigate logs for function:${LAMBDA_FUNCTION} version ${VERSION}."
  exit 1
fi
echo "  Canary passed (status=$CANARY_STATUS)"

# ── Step 8: Promote version to 'live' (only after canary passes) ─────────────
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
echo "Rollback: bash infrastructure/rollback.sh"
