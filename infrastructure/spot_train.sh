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
#   - config/predictor.yaml on S3 or SCP'd separately (see below)
#
# The script will:
#   1. Request a spot instance (c5.xlarge, ~$0.06/hr)
#   2. Wait for SSH to become available
#   3. Clone the repo and install dependencies
#   4. Copy config/predictor.yaml from local machine → EC2
#   5. Run smoke test (dry_run=True) to verify config + code
#   6. Prompt to continue with full training (dry_run=False)
#   7. Terminate the spot instance
#
# Environment variables:
#   AWS_REGION           — default: us-east-1
#   S3_BUCKET            — default: alpha-engine-research
#   BRANCH               — git branch to checkout (default: main)
#   EMAIL_SENDER         — forwarded to training email
#   EMAIL_RECIPIENTS     — forwarded to training email
#   GMAIL_APP_PASSWORD   — forwarded to training email

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-alpha-engine-research}"
BRANCH="${BRANCH:-main}"
INSTANCE_TYPE="c5.xlarge"
AMI_ID="ami-05024c2628f651b80"  # Amazon Linux 2 x86_64
KEY_NAME="alpha-engine-key"
KEY_FILE="$HOME/.ssh/alpha-engine-key.pem"
SECURITY_GROUP="sg-03cd3c4bd91e610b0"
SUBNET_ID="subnet-e07166ec"
IAM_PROFILE="alpha-engine-executor-profile"
REPO_URL="git@github.com:cipher813/alpha-engine-predictor.git"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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

# Install Python 3.12, git, and build tools
sudo yum install -y python3.12 python3.12-pip git gcc python3.12-devel 2>/dev/null || {
  # Amazon Linux 2 might need amazon-linux-extras
  sudo amazon-linux-extras install python3.8 -y 2>/dev/null || true
  sudo yum install -y python3 python3-pip git gcc python3-devel 2>/dev/null
}

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

# Set up SSH for GitHub (deploy key or HTTPS fallback handled below)
mkdir -p ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
BOOTSTRAP

echo "==> Cloning repository (branch: $BRANCH)..."
# Try SSH clone first, fall back to HTTPS
# Forward SSH agent for GitHub access
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

# Use whichever python is available
if command -v python3.12 &>/dev/null; then
  PYTHON=python3.12
  PIP="python3.12 -m pip"
else
  PYTHON=python3
  PIP="python3 -m pip"
fi

$PIP install --upgrade pip
$PIP install -r requirements.txt
echo "Dependencies installed."
DEPS

# ── Copy local config to EC2 ──────────────────────────────────────────────────
echo "==> Uploading config/predictor.yaml..."
scp $SSH_OPTS -i "$KEY_FILE" \
  "$REPO_ROOT/config/predictor.yaml" \
  ec2-user@"$PUBLIC_IP":/home/ec2-user/predictor/config/predictor.yaml

# ── Forward email env vars ────────────────────────────────────────────────────
# Build env var export string from local environment
ENV_EXPORTS=""
for var in EMAIL_SENDER EMAIL_RECIPIENTS GMAIL_APP_PASSWORD AWS_REGION S3_BUCKET; do
  val="${!var:-}"
  if [ -n "$val" ]; then
    ENV_EXPORTS+="export ${var}='${val}'; "
  fi
done
ENV_EXPORTS+="export S3_BUCKET='${S3_BUCKET}'; "
ENV_EXPORTS+="export XDG_CACHE_HOME=/tmp; "

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
${ENV_EXPORTS}

$REMOTE_PYTHON -c "
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('S3_BUCKET', '${S3_BUCKET}')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s')

from training.train_handler import main as train_main
result = train_main(bucket='${S3_BUCKET}', dry_run=True)

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
# Feature ICs
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
${ENV_EXPORTS}

$REMOTE_PYTHON -c "
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('S3_BUCKET', '${S3_BUCKET}')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s')

from training.train_handler import main as train_main
result = train_main(bucket='${S3_BUCKET}', dry_run=False)

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
