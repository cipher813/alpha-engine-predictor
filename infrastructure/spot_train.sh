#!/usr/bin/env bash
# infrastructure/spot_train.sh — Run GBM retraining on a spot EC2 instance.
#
# Launches a c5.xlarge spot instance, syncs code, runs training via the
# same train_handler.main() pipeline that Lambda uses (S3 price cache
# download → refresh → train → promote → slim cache → email).
#
# Usage:
#   ./infrastructure/spot_train.sh                  # smoke test (dry_run), then prompt for full train
#   ./infrastructure/spot_train.sh --full-only       # skip smoke test, run full training directly
#   ./infrastructure/spot_train.sh --smoke-only      # run smoke test only, then terminate
#   ./infrastructure/spot_train.sh --instance-type c5.2xlarge  # override instance type
#
# Prerequisites:
#   - AWS CLI configured (uses alpha-engine-executor-profile for S3/email access)
#   - SSH key at ~/.ssh/alpha-engine-key.pem
#   - Code committed and pushed to origin (the instance clones from GitHub)
#   - .env file with EMAIL_SENDER, EMAIL_RECIPIENTS, GMAIL_APP_PASSWORD
#   - config/predictor.yaml (gitignored — SCP'd to EC2 by this script)
#
# The script will:
#   1. Request a spot instance (c5.xlarge, ~$0.06/hr)
#   2. Wait for SSH to become available
#   3. Clone the repo and install dependencies
#   4. Copy config/predictor.yaml and .env from local machine → EC2
#   5. Run smoke test (dry_run=True) to verify config + code
#   6. Prompt to continue with full training (dry_run=False)
#   7. Terminate the spot instance

set -euo pipefail

# ── Load .env ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_FILE="$REPO_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
  # Export all non-comment, non-empty lines from .env
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "Loaded .env from $ENV_FILE"
else
  echo "WARNING: No .env file found at $ENV_FILE"
  echo "         Email notifications will be skipped."
  echo "         Copy .env.example to .env and fill in values."
  echo ""
fi

# ── Configuration ──────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-alpha-engine-research}"
BRANCH="${BRANCH:-main}"
INSTANCE_TYPE="c5.xlarge"
AMI_ID="ami-0c421724a94bba6d6"  # Amazon Linux 2023 x86_64 (Python 3.12)
KEY_NAME="alpha-engine-key"
KEY_FILE="$HOME/.ssh/alpha-engine-key.pem"
SECURITY_GROUP="sg-03cd3c4bd91e610b0"
SUBNET_ID="subnet-e07166ec"
IAM_PROFILE="alpha-engine-executor-profile"
REPO_URL="git@github.com:cipher813/alpha-engine-predictor.git"

# Parse flags
MODE="both"  # both | full-only | smoke-only
for arg in "$@"; do
  case "$arg" in
    --full-only) MODE="full-only" ;;
    --smoke-only) MODE="smoke-only" ;;
    --instance-type)
      shift
      INSTANCE_TYPE="$1"
      ;;
  esac
done

echo "═══════════════════════════════════════════════════════════════"
echo "  GBM Spot Training — $(date +%Y-%m-%d)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Instance type : $INSTANCE_TYPE"
echo "  AMI           : $AMI_ID"
echo "  Region        : $AWS_REGION"
echo "  Branch        : $BRANCH"
echo "  Mode          : $MODE"
echo "  S3 bucket     : $S3_BUCKET"
echo "  Email sender  : ${EMAIL_SENDER:-<not set>}"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────
if [ ! -f "$KEY_FILE" ]; then
  echo "ERROR: SSH key not found at $KEY_FILE"
  exit 1
fi

if [ ! -f "$REPO_ROOT/config/predictor.yaml" ]; then
  echo "ERROR: config/predictor.yaml not found — copy from predictor.sample.yaml"
  exit 1
fi

# Check for uncommitted changes
cd "$REPO_ROOT"
if ! git diff --quiet HEAD -- config.py config/predictor.sample.yaml training/train_handler.py README.md; then
  echo "WARNING: You have uncommitted changes in key files."
  echo "         The spot instance clones from origin/$BRANCH."
  echo "         Commit and push first, or changes won't be included."
  echo ""
  read -p "Continue anyway? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. Commit and push first:"
    echo "  git add -A && git commit -m 'GBM signal strength + monitoring' && git push"
    exit 1
  fi
fi

# ── Launch spot instance ──────────────────────────────────────────────────────
echo "==> Requesting spot instance ($INSTANCE_TYPE)..."

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SECURITY_GROUP" \
  --subnet-id "$SUBNET_ID" \
  --iam-instance-profile Name="$IAM_PROFILE" \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=alpha-engine-gbm-train-$(date +%Y%m%d)}]" \
  --region "$AWS_REGION" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "  Instance ID: $INSTANCE_ID"

# Cleanup function — always terminate the instance
cleanup() {
  echo ""
  echo "==> Terminating spot instance $INSTANCE_ID..."
  aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" --output text > /dev/null 2>&1 || true
  echo "  Instance terminated."
}
trap cleanup EXIT

# Wait for instance to be running
echo "==> Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region "$AWS_REGION")

if [ "$PUBLIC_IP" = "None" ] || [ -z "$PUBLIC_IP" ]; then
  echo "ERROR: Instance has no public IP. Check subnet/VPC configuration."
  exit 1
fi

echo "  Public IP: $PUBLIC_IP"

# ── Wait for SSH ──────────────────────────────────────────────────────────────
echo "==> Waiting for SSH to become available..."
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -o LogLevel=ERROR"

for i in $(seq 1 30); do
  if ssh $SSH_OPTS -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "echo ok" 2>/dev/null; then
    echo "  SSH ready."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "ERROR: SSH not available after 150s"
    exit 1
  fi
  sleep 5
done

# Helper: run command on EC2
run_remote() {
  ssh $SSH_OPTS -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "$@"
}

# ── Bootstrap environment on EC2 ──────────────────────────────────────────────
echo "==> Bootstrapping EC2 environment..."
run_remote bash -s <<'BOOTSTRAP'
set -euo pipefail

# Amazon Linux 2023: install Python 3.12, git, gcc, and pip
sudo dnf install -y -q python3.12 python3.12-pip python3.12-devel git gcc 2>/dev/null || \
  sudo dnf install -y -q python3 python3-pip python3-devel git gcc

# Determine python binary
if command -v python3.12 &>/dev/null; then
  PYTHON=python3.12
elif command -v python3 &>/dev/null; then
  PYTHON=python3
else
  echo "ERROR: No python3 found"
  exit 1
fi

echo "Using: $($PYTHON --version)"

# Set up SSH for GitHub
mkdir -p ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
BOOTSTRAP

echo "==> Cloning repository (branch: $BRANCH)..."
# Try SSH clone first (via agent forwarding), fall back to HTTPS
ssh -A $SSH_OPTS -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" \
  "git clone --depth 1 --branch $BRANCH $REPO_URL /home/ec2-user/predictor" 2>/dev/null || {
  echo "  SSH clone failed — trying HTTPS..."
  HTTPS_URL="https://github.com/cipher813/alpha-engine-predictor.git"
  run_remote "git clone --depth 1 --branch $BRANCH $HTTPS_URL /home/ec2-user/predictor"
}

echo "==> Installing Python dependencies..."
run_remote bash -s <<'DEPS'
set -euo pipefail
cd /home/ec2-user/predictor

if command -v python3.12 &>/dev/null; then
  PIP="python3.12 -m pip"
else
  PIP="python3 -m pip"
fi

$PIP install --upgrade pip -q
# Filter out private packages (flow-doctor) that aren't on PyPI
grep -v '^flow-doctor' requirements.txt | $PIP install -q -r /dev/stdin
echo "Dependencies installed."
$PIP list --format=columns | grep -iE 'numpy|pandas|lightgbm|scipy|shap|pyyaml' || true
DEPS

# ── Copy local config + .env to EC2 ──────────────────────────────────────────
echo "==> Uploading config/predictor.yaml and .env..."
scp $SSH_OPTS -i "$KEY_FILE" \
  "$REPO_ROOT/config/predictor.yaml" \
  ec2-user@"$PUBLIC_IP":/home/ec2-user/predictor/config/predictor.yaml

if [ -f "$ENV_FILE" ]; then
  scp $SSH_OPTS -i "$KEY_FILE" \
    "$ENV_FILE" \
    ec2-user@"$PUBLIC_IP":/home/ec2-user/predictor/.env
fi

# ── Build env export command for remote shells ────────────────────────────────
# Source .env on the remote side so all training/email code sees the vars
ENV_SOURCE='set -a; [ -f /home/ec2-user/predictor/.env ] && source /home/ec2-user/predictor/.env; set +a; export XDG_CACHE_HOME=/tmp;'

# ── Determine python binary on remote ─────────────────────────────────────────
REMOTE_PYTHON=$(run_remote "command -v python3.12 || command -v python3")

# ── Smoke test ────────────────────────────────────────────────────────────────
if [ "$MODE" != "full-only" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  SMOKE TEST (dry_run=True)"
  echo "═══════════════════════════════════════════════════════════════"
  echo ""

  run_remote bash -s <<SMOKE
set -euo pipefail
cd /home/ec2-user/predictor
${ENV_SOURCE}

$REMOTE_PYTHON -c "
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('S3_BUCKET', os.environ.get('S3_BUCKET', 'alpha-engine-research'))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s')

from training.train_handler import main as train_main
result = train_main(bucket=os.environ.get('S3_BUCKET', 'alpha-engine-research'), dry_run=True)

print()
print('=' * 60)
print('  SMOKE TEST RESULTS')
print('=' * 60)
print(f'  Test IC:        {result.get(\"test_ic\", \"n/a\")}')
print(f'  MSE IC:         {result.get(\"mse_ic\", \"n/a\")}')
print(f'  Rank IC:        {result.get(\"rank_ic\", \"n/a\")}')
print(f'  Ensemble IC:    {result.get(\"ensemble_ic\", \"n/a\")}')
print(f'  IC IR:          {result.get(\"ic_ir\", \"n/a\")}')
print(f'  Passes IC gate: {result.get(\"passes_ic_gate\", \"n/a\")}')
print(f'  Promoted:       {result.get(\"promoted\", \"n/a\")}')
print(f'  Walk-forward:   {\"PASS\" if result.get(\"walk_forward\", {}).get(\"passes_wf\") else \"FAIL/skipped\"}')
print(f'  Elapsed:        {result.get(\"elapsed_s\", \"n/a\")}s')
print(f'  Noise features: {result.get(\"noise_candidates\", [])}')
fics = result.get('feature_ics', {})
if fics:
    sorted_fics = sorted(fics.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f'  Top 5 feature ICs:')
    for name, ic in sorted_fics[:5]:
        print(f'    {name:<22} {ic:+.4f}')
print('=' * 60)
"
SMOKE

  echo ""
  echo "Smoke test complete."

  if [ "$MODE" = "smoke-only" ]; then
    echo "==> Smoke-only mode — skipping full training."
    exit 0
  fi

  echo ""
  read -p "Proceed with full training (writes to S3, sends email)? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. Instance will be terminated."
    exit 0
  fi
fi

# ── Full training ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  FULL TRAINING (dry_run=False)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_remote bash -s <<TRAIN
set -euo pipefail
cd /home/ec2-user/predictor
${ENV_SOURCE}

$REMOTE_PYTHON -c "
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('S3_BUCKET', os.environ.get('S3_BUCKET', 'alpha-engine-research'))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s')

from training.train_handler import main as train_main
result = train_main(bucket=os.environ.get('S3_BUCKET', 'alpha-engine-research'), dry_run=False)

print()
print('=' * 60)
print('  FULL TRAINING RESULTS')
print('=' * 60)
print(f'  Test IC:        {result.get(\"test_ic\", \"n/a\")}')
print(f'  MSE IC:         {result.get(\"mse_ic\", \"n/a\")}')
print(f'  Rank IC:        {result.get(\"rank_ic\", \"n/a\")}')
print(f'  Ensemble IC:    {result.get(\"ensemble_ic\", \"n/a\")}')
print(f'  IC IR:          {result.get(\"ic_ir\", \"n/a\")}')
print(f'  Passes IC gate: {result.get(\"passes_ic_gate\", \"n/a\")}')
print(f'  Promoted:       {result.get(\"promoted\", \"n/a\")}')
print(f'  Promoted mode:  {result.get(\"promoted_mode\", \"n/a\")}')
print(f'  Walk-forward:   {\"PASS\" if result.get(\"walk_forward\", {}).get(\"passes_wf\") else \"FAIL/skipped\"}')
print(f'  Elapsed:        {result.get(\"elapsed_s\", \"n/a\")}s')
print(f'  Slim cache:     {result.get(\"slim_cache_tickers\", \"n/a\")} tickers')
print('=' * 60)
"
TRAIN

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Training complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"
