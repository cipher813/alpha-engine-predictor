#!/usr/bin/env bash
# infrastructure/spot_train.sh — Run GBM retraining on a spot EC2 instance.
#
# Launches a c5.large spot instance, syncs code, runs training via the
# same train_handler.main() pipeline that Lambda uses (S3 price cache
# download → refresh → train → promote → slim cache → email).
#
# Communication is via `aws ssm send-command` (IAM-authenticated, CloudTrail-
# audited) — NOT SSH/SCP. Config is staged through S3; secrets are read on
# the spot via alpha_engine_lib.secrets.get_secret() (SSM Parameter Store),
# so there is no `.env` SCP and no `~/.ssh/alpha-engine-key.pem` dependency
# in the workflow. (PR 2 of the spot-train-260512 SSH/SCP→SSM migration;
# canonical plan: alpha-engine-docs/private/spot-train-260512.md.)
#
# Usage:
#   ./infrastructure/spot_train.sh                  # smoke (dry_run) then full
#   ./infrastructure/spot_train.sh --full-only       # full training only (Saturday SF)
#   ./infrastructure/spot_train.sh --smoke-only      # smoke only, then terminate
#   ./infrastructure/spot_train.sh --preflight-only  # boot + import/lib-pin +
#                                                    # ArcticDB connectivity probe,
#                                                    # then exit 0 — NO training,
#                                                    # NO promotion, ZERO S3/config
#                                                    # writes (Friday shell_run dry path)
#   ./infrastructure/spot_train.sh --instance-type c5.2xlarge  # override type
#
# Prerequisites:
#   - AWS CLI configured (alpha-engine-executor-profile — S3 + SSM + email).
#     The instance profile carries AmazonSSMManagedInstanceCore so the spot
#     registers with SSM; this script polls SSM for readiness (no port 22).
#   - Code committed + pushed to origin/$BRANCH (the spot clones HTTPS).
#   - config/predictor.yaml present locally (gitignored — staged to S3).
#
# The script will:
#   1. Request a spot instance (c5.large, ~$0.03/hr)
#   2. Wait for the SSM agent to register (no SSH)
#   3. Stage config/predictor.yaml to S3; spot bootstraps + fetches it
#   4. Run smoke (dry_run=True), then full training (dry_run=False)
#      — OR, under --preflight-only, run the import/lib-pin + read-only
#        ArcticDB connectivity probe and exit 0 (no training, no promotion,
#        no S3/config writes; Friday shell_run dry path)
#   5. Terminate the spot instance + clean the S3 staging prefix
#
# Rollback: `git revert` this commit restores the SSH/SCP script. Port 22
# ingress on the SG is intentionally left in place until the migration's
# PR 3 (SG cleanup), so emergency `ssh`/`aws ssm start-session` remains
# available during the validation window.

set -euo pipefail

# SSM RunCommand executes as root with a minimal env — set HOME/cache dirs
# explicitly wherever the workload runs (done per-step below too).
export HOME="${HOME:-/home/ec2-user}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Configuration ──────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-alpha-engine-research}"
BRANCH="${BRANCH:-main}"
# c5.large = 2 vCPU / 4 GB RAM. Right-sized 2026-04-28: meta-trainer
# steady-state is ~1–1.5 GB. Override via ``--instance-type`` if a future
# feature pushes peak memory past ~3 GB.
INSTANCE_TYPE="c5.large"
AMI_ID="ami-0c421724a94bba6d6"  # Amazon Linux 2023 x86_64 (Python 3.12, SSM agent preinstalled)
# Spot-side watchdog budget: meta-trainer typically completes 40-70 min;
# include pip install + smoke + full run. 90 min with headroom.
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-5400}"
# KEY_NAME is still passed to run-instances so emergency SSH stays possible
# during the validation window (the SG's port 22 ingress is dropped only in
# the migration's PR 3, after this PR validates against a Saturday SF).
KEY_NAME="alpha-engine-key"
SECURITY_GROUP="sg-03cd3c4bd91e610b0"
SUBNET_ID="subnet-e07166ec"
IAM_PROFILE="alpha-engine-executor-profile"
REPO_URL="https://github.com/cipher813/alpha-engine-predictor.git"  # public repo, no auth

# Parse flags
MODE="both"  # both | full-only | smoke-only | preflight-only
while [ $# -gt 0 ]; do
  case "$1" in
    --full-only) MODE="full-only" ;;
    --smoke-only) MODE="smoke-only" ;;
    --preflight-only) MODE="preflight-only" ;;
    --instance-type) shift; INSTANCE_TYPE="$1" ;;
  esac
  shift
done

echo "═══════════════════════════════════════════════════════════════"
echo "  GBM Spot Training — $(date +%Y-%m-%d)  (SSM transport)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Instance type : $INSTANCE_TYPE"
echo "  AMI           : $AMI_ID"
echo "  Region        : $AWS_REGION"
echo "  Branch        : $BRANCH"
echo "  Mode          : $MODE"
echo "  S3 bucket     : $S3_BUCKET"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────
if [ ! -f "$REPO_ROOT/config/predictor.yaml" ]; then
  echo "ERROR: config/predictor.yaml not found — copy from predictor.sample.yaml"
  exit 1
fi

# Uncommitted-changes check — WARN only (non-interactive: this runs under the
# Saturday Step Function with no TTY). The spot clones origin/$BRANCH, so
# uncommitted local changes simply won't be included.
cd "$REPO_ROOT"
if ! git diff --quiet HEAD -- config.py config/predictor.sample.yaml training/train_handler.py model/ data/ README.md 2>/dev/null; then
  echo "WARNING: uncommitted changes in key files — the spot clones origin/$BRANCH,"
  echo "         so those changes will NOT be included. Commit + push first if intended."
  echo ""
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
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=alpha-engine-gbm-train-$(date +%Y%m%d)}]" \
  --region "$AWS_REGION" \
  --query 'Instances[0].InstanceId' \
  --output text)
echo "  Instance ID: $INSTANCE_ID"

RUN_ID="$(date +%Y%m%dT%H%M%SZ)-${INSTANCE_ID}"
S3_STAGING_PREFIX="tmp/spot_train/${RUN_ID}"
S3_STAGING="s3://${S3_BUCKET}/${S3_STAGING_PREFIX}"

# Cleanup — always terminate the instance + remove the S3 staging prefix.
# (S3 lifecycle on tmp/ is the belt-and-suspenders if the trap never fires.)
cleanup() {
  echo ""
  echo "==> Terminating spot instance $INSTANCE_ID..."
  aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" --output text > /dev/null 2>&1 || true
  aws s3 rm "$S3_STAGING" --recursive --quiet 2>/dev/null || true
  echo "  Instance terminated; S3 staging cleaned."
}
trap cleanup EXIT

echo "==> Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

# Stage config/predictor.yaml to S3 (spot fetches via its IAM role).
echo "==> Staging config/predictor.yaml → ${S3_STAGING}/predictor.yaml"
aws s3 cp "$REPO_ROOT/config/predictor.yaml" "${S3_STAGING}/predictor.yaml" --region "$AWS_REGION" --quiet

# ── Wait for the SSM agent to register ────────────────────────────────────────
# Replaces the old SSH-readiness poll. AL2023 ships the SSM agent; with the
# instance profile's AmazonSSMManagedInstanceCore it registers within ~1 min.
echo "==> Waiting for SSM agent to come Online..."
for i in $(seq 1 36); do  # 36 × 5s = 180s budget
  ping=$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --query 'InstanceInformationList[0].PingStatus' \
    --output text --region "$AWS_REGION" 2>/dev/null || true)
  if [ "$ping" = "Online" ]; then
    echo "  SSM agent Online."
    break
  fi
  if [ "$i" -eq 36 ]; then
    echo "ERROR: SSM agent not Online after 180s (instance $INSTANCE_ID)"
    exit 1
  fi
  sleep 5
done

# ── SSM command primitive ─────────────────────────────────────────────────────
# run_ssm "<description>" "<bash script>" [timeout_seconds]
#
# Ships the script body via AWS-RunShellScript (base64-wrapped so embedded
# Python/heredoc quoting is transport-safe), captures full stdout/stderr to
# S3 (past SSM's 24KB response cap), and streams the StandardOutputContent
# delta to the dispatcher while polling. Returns non-zero on any non-Success
# terminal status. jq-free (python3 + `aws --query` only) so it runs on the
# Saturday-SF dispatcher (ae-dashboard) without extra deps.
run_ssm() {
  local description="$1" script="$2" timeout_s="${3:-3600}"
  local b64 pfile cmd_id status out last=0 err

  b64=$(printf '%s' "$script" | base64 | tr -d '\n')
  pfile=$(mktemp)
  REMOTE_CMD="echo $b64 | base64 -d | bash" python3 - "$pfile" <<'PYJSON'
import json, os, sys
json.dump({"commands": [os.environ["REMOTE_CMD"]]}, open(sys.argv[1], "w"))
PYJSON

  cmd_id=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name AWS-RunShellScript \
    --comment "predictor-training: $description" \
    --timeout-seconds "$timeout_s" \
    --parameters "file://$pfile" \
    --output-s3-bucket-name "$S3_BUCKET" \
    --output-s3-key-prefix "${S3_STAGING_PREFIX}/ssm-output" \
    --region "$AWS_REGION" \
    --query 'Command.CommandId' --output text) || { rm -f "$pfile"; return 1; }
  rm -f "$pfile"

  echo "    [ssm $description] command-id=$cmd_id"
  while :; do
    sleep 5
    status=$(aws ssm get-command-invocation --command-id "$cmd_id" \
      --instance-id "$INSTANCE_ID" --region "$AWS_REGION" \
      --query 'Status' --output text 2>/dev/null || echo "Pending")
    out=$(aws ssm get-command-invocation --command-id "$cmd_id" \
      --instance-id "$INSTANCE_ID" --region "$AWS_REGION" \
      --query 'StandardOutputContent' --output text 2>/dev/null || echo "")
    if [ "${#out}" -gt "$last" ]; then
      printf '%s' "${out:$last}"
      last=${#out}
    elif [ "${#out}" -lt "$last" ]; then
      # 24KB cap rotated the buffer — full log is in S3.
      echo "    [ssm $description] (stdout exceeded 24KB cap — full log: ${S3_STAGING}/ssm-output/)"
      last=${#out}
    fi
    case "$status" in
      Success) echo ""; return 0 ;;
      Pending|InProgress|Delayed) : ;;
      *)
        err=$(aws ssm get-command-invocation --command-id "$cmd_id" \
          --instance-id "$INSTANCE_ID" --region "$AWS_REGION" \
          --query 'StandardErrorContent' --output text 2>/dev/null || echo "")
        echo ""
        echo "ERROR: SSM step '$description' terminal status=$status"
        [ -n "$err" ] && { echo "--- stderr (24KB cap; full: ${S3_STAGING}/ssm-output/) ---"; printf '%s\n' "$err"; }
        return 1 ;;
    esac
  done
}

# Each run_ssm step is a fresh SSM shell with a minimal env. The
# .env-deprecation arc deleted the sourced .env, so AWS_REGION/
# AWS_DEFAULT_REGION (which boto3 + training/preflight.py's
# check_env_vars("AWS_REGION") require) are no longer set unless each
# step's export line sets them. Same #247 regression as alpha-engine-data's
# spot scripts; spot_train.sh is a sibling repo the original arc missed.
# System is single-region us-east-1 (matches this file's own
# ${AWS_REGION:-us-east-1} defaults). Origin: 2026-05-16 Saturday SF
# PredictorTraining preflight failure.
# ── Bootstrap (watchdog + deps + clone + staged config) ───────────────────────
echo "==> Bootstrapping spot (watchdog, python, clone, config)..."
run_ssm "bootstrap" "$(cat <<BOOTSTRAP
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1

# Spot-side hard-timeout watchdog. The dispatcher-side 'trap cleanup EXIT'
# only fires if THIS script exits; if the dispatcher is killed/cancelled the
# spot would orphan. systemd-run shuts the box down after MAX_RUNTIME_SECONDS
# regardless of dispatcher state.
systemd-run --on-active=${MAX_RUNTIME_SECONDS} --unit=alpha-engine-watchdog \
  --description='alpha-engine spot hard-timeout' /sbin/shutdown -h now

dnf install -y -q python3.12 python3.12-pip python3.12-devel git gcc 2>/dev/null || \
  dnf install -y -q python3 python3-pip python3-devel git gcc
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
echo "Using: \$(\$PY --version)"

git clone --depth 1 --branch ${BRANCH} ${REPO_URL} /home/ec2-user/predictor
aws s3 cp ${S3_STAGING}/predictor.yaml /home/ec2-user/predictor/config/predictor.yaml --region ${AWS_REGION}
echo "Bootstrap complete: repo cloned, predictor.yaml staged from S3."
BOOTSTRAP
)" 600

# ── Dependencies ──────────────────────────────────────────────────────────────
echo "==> Installing Python dependencies..."
run_ssm "deps" "$(cat <<'DEPS'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PIP="python3.12 -m pip" || PIP="python3 -m pip"
$PIP install --upgrade pip -q
# alpha-engine-lib is public (git+https in requirements.txt, no auth).
# flow-doctor is private + not on PyPI — filtered out (same as legacy).
grep -v '^flow-doctor' requirements.txt | $PIP install -q -r /dev/stdin
echo "Dependencies installed."
$PIP list --format=columns | grep -iE 'numpy|pandas|lightgbm|scikit-learn|scipy|shap|pyyaml|alpha-engine-lib' || true
DEPS
)" 900

# ── Preflight-only (Friday shell_run dry path) ────────────────────────────────
# Boot + lib-pin/import + read-only ArcticDB/universe-freshness probe, then
# exit 0. This runs the SAME bootstrap+deps steps the real Saturday run uses
# (so it catches lib-pin drift, sys.path breakage, image gaps, SSM timeouts,
# stale ArcticDB) but stops HERE — before the smoke step and before the
# full-training step.
#
# Hard invariant under this mode:
#   • run_meta_training() is NEVER invoked → NO model training, NO walk-forward.
#   • The `if not dry_run:` upload/promote block in meta_trainer.py is never
#     reached → NO weights/meta/* write, NO manifest, NO dated archive.
#   • train_handler.main()'s training_summary / triple-barrier-gate / email /
#     health-status writes are never reached (they live after run_meta_training).
#   • The probe imports the training package + runs TrainingPreflight (env +
#     S3-bucket *reachability* check — no object writes) + a read-only
#     ArcticDB `list_symbols()` / latest-index probe. No put_object, no
#     config write, no external API (yfinance/Anthropic) call.
# The `exit 0` is a clean dispatcher exit; `trap cleanup EXIT` still fires
# (terminates the spot, clears the S3 staging prefix — staging cleanup only).
if [ "$MODE" = "preflight-only" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  PREFLIGHT-ONLY (no training, no promotion, no writes)"
  echo "═══════════════════════════════════════════════════════════════"
  run_ssm "preflight-only" "$(cat <<'PREFLIGHT'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
$PY - <<'PYEOF'
import os, sys
sys.path.insert(0, '.')
os.environ.setdefault('S3_BUCKET', os.environ.get('S3_BUCKET', 'alpha-engine-research'))
bucket = os.environ.get('S3_BUCKET', 'alpha-engine-research')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s')
log = logging.getLogger('preflight-only')

# 1. Import the training package (catches sys.path / lib-pin / image gaps).
#    Importing train_handler transitively imports the lib + training stack
#    WITHOUT invoking main(), so no training runs.
log.info('[1/3] Importing training package...')
import alpha_engine_lib  # lib-pin presence (version asserted by requirements.txt pin)
from training import train_handler  # noqa: F401  (import-only; main() NOT called)
from training.preflight import TrainingPreflight
log.info('       OK — alpha_engine_lib + training.train_handler import clean')

# 2. Reuse the EXISTING training preflight (env vars + S3 bucket
#    *reachability*; check_s3_bucket is a read/head, no object write).
log.info('[2/3] Running TrainingPreflight (env + S3 connectivity)...')
TrainingPreflight(bucket=bucket).run()
log.info('       OK — env vars present, S3 bucket reachable')

# 3. Read-only ArcticDB connectivity + universe-freshness probe.
#    list_symbols() + a single read().tail(1) — NO download_from_arctic(),
#    NO parquet writes, NO training array build. Mirrors the connectivity
#    the real run depends on without doing any work.
log.info('[3/3] ArcticDB connectivity + universe-freshness probe...')
from store.arctic_reader import _get_arctic
arctic = _get_arctic(bucket)
universe = arctic.get_library('universe')
symbols = universe.list_symbols()
n = len(symbols)
if n == 0:
    raise RuntimeError(
        'ArcticDB universe library is empty/unreachable — '
        'Saturday DataPhase1 + weekly backfill have not run cleanly.'
    )
probe = sorted(symbols)[0]
df_tail = universe.read(probe).data.tail(1)
latest = df_tail.index.max() if not df_tail.empty else 'n/a'
log.info('       OK — universe has %d symbols; %s latest index=%s', n, probe, latest)

print()
print('=' * 60)
print('  PREFLIGHT-ONLY RESULT: PASS')
print('=' * 60)
print(f'  Imports:        alpha_engine_lib + training stack clean')
print(f'  TrainingPreflight: PASS (env + S3 reachable)')
print(f'  ArcticDB:       {n} universe symbols (probe {probe} latest={latest})')
print(f'  Training:       SKIPPED (no run_meta_training call)')
print(f'  Promotion:      SKIPPED (no weights/meta write)')
print(f'  S3/config writes: NONE')
print('=' * 60)
PYEOF
PREFLIGHT
)" 600
  echo ""
  echo "==> Preflight-only mode — PASS. No training, no promotion, no writes."
  echo "    Exiting 0 BEFORE smoke + full-training steps."
  exit 0
fi

# ── Smoke test (dry_run=True) ─────────────────────────────────────────────────
if [ "$MODE" != "full-only" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  SMOKE TEST (dry_run=True)"
  echo "═══════════════════════════════════════════════════════════════"
  run_ssm "smoke" "$(cat <<'SMOKE'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
$PY - <<'PYEOF'
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
v = result.get('model_version', '')
is_meta = 'meta' in str(v).lower()

if is_meta:
    print(f'  Architecture:   v3.0 Meta-Model')
    print(f'  Meta-Model IC:  {result.get("meta_model_ic", result.get("test_ic", "n/a"))}')
    print(f'  Momentum IC:    {result.get("momentum_test_ic", "n/a")}')
    print(f'  Volatility IC:  {result.get("volatility_test_ic", "n/a")}')
    print(f'  Regime Acc:     {(result.get("regime_accuracy", 0) * 100):.1f}%')
    rc = result.get('research_calibrator_metrics', {})
    if rc:
        print(f'  Research Cal:   {rc.get("n_samples", 0)} samples, overall hit={rc.get("overall_hit_rate", "n/a")}')
        for bucket, info in rc.get('buckets', {}).items():
            if info.get('n', 0) > 0:
                print(f'    Score {bucket}: hit_rate={info["hit_rate"]:.1%} (n={info["n"]})')
    wf = result.get('walk_forward', {})
    print(f'  WF Momentum:    median_IC={wf.get("momentum_median_ic", "n/a")}')
    print(f'  WF Volatility:  median_IC={wf.get("volatility_median_ic", "n/a")}')
    wf_status = 'PASS' if wf.get('passes_wf') else 'FAIL'
    print(f'  WF Status:      {wf_status}')
    coefs = result.get('meta_coefficients', {})
    if coefs:
        print(f'  Meta-model coefficients:')
        for name, val in sorted(coefs.items(), key=lambda x: -abs(x[1])):
            if name != 'intercept' and abs(val) > 0.0001:
                print(f'    {name:<30} {val:+.4f}')
        print(f'    {"intercept":<30} {coefs.get("intercept", 0):+.4f}')
    if wf.get('folds'):
        print(f'  Per-fold ICs (momentum / volatility):')
        for f in wf['folds']:
            print(f'    Fold {f["fold"]:>2}: mom={f["mom_ic"]:+.4f}  vol={f["vol_ic"]:+.4f}  [{f["test_start"]} -> {f["test_end"]}]')
else:
    print(f'  Architecture:   v2.0 Single/Ensemble GBM')
    print(f'  Test IC:        {result.get("test_ic", "n/a")}')
    print(f'  MSE IC:         {result.get("mse_ic", "n/a")}')
    print(f'  Rank IC:        {result.get("rank_ic", "n/a")}')
    print(f'  Ensemble IC:    {result.get("ensemble_ic", "n/a")}')
    if result.get('catboost_enabled'):
        print(f'  CatBoost IC:    {result.get("catboost_ic", "n/a")}')
        print(f'  LGB-Cat Blend:  {result.get("lgb_cat_blend_ic", "n/a")}  weights={result.get("blend_weights", "n/a")}')
    print(f'  IC IR:          {result.get("ic_ir", "n/a")}')
    wf = result.get('walk_forward', {})
    wf_status = 'PASS' if wf.get('passes_wf') else 'FAIL/skipped'
    print(f'  Walk-forward:   {wf_status}  (median_IC={wf.get("median_ic", "n/a")})')
    fics = result.get('feature_ics', {})
    if fics:
        sorted_fics = sorted(fics.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f'  Top 5 feature ICs:')
        for name, ic in sorted_fics[:5]:
            print(f'    {name:<22} {ic:+.4f}')

print(f'  Promoted:       {result.get("promoted", "n/a")}')
print(f'  Elapsed:        {result.get("elapsed_s", "n/a")}s')
noise = result.get('noise_candidates', [])
if noise:
    print(f'  Noise features: {noise}')
print('=' * 60)
PYEOF
SMOKE
)" 1800
  echo "Smoke test complete."
  if [ "$MODE" = "smoke-only" ]; then
    echo "==> Smoke-only mode — skipping full training."
    exit 0
  fi
fi

# ── Full training (dry_run=False) ─────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  FULL TRAINING (dry_run=False)"
echo "═══════════════════════════════════════════════════════════════"
run_ssm "full-training" "$(cat <<'TRAIN'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
$PY - <<'PYEOF'
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
v = result.get('model_version', '')
is_meta = 'meta' in str(v).lower()

if is_meta:
    print(f'  Architecture:   v3.0 Meta-Model')
    print(f'  Meta-Model IC:  {result.get("meta_model_ic", result.get("test_ic", "n/a"))}')
    print(f'  Momentum IC:    {result.get("momentum_test_ic", "n/a")}')
    print(f'  Volatility IC:  {result.get("volatility_test_ic", "n/a")}')
    print(f'  Regime Acc:     {(result.get("regime_accuracy", 0) * 100):.1f}%')
    rc = result.get('research_calibrator_metrics', {})
    if rc:
        print(f'  Research Cal:   {rc.get("n_samples", 0)} samples, overall hit={rc.get("overall_hit_rate", "n/a")}')
    wf = result.get('walk_forward', {})
    print(f'  WF Momentum:    median_IC={wf.get("momentum_median_ic", "n/a")}')
    print(f'  WF Volatility:  median_IC={wf.get("volatility_median_ic", "n/a")}')
    coefs = result.get('meta_coefficients', {})
    if coefs:
        print(f'  Meta-model coefficients:')
        for name, val in sorted(coefs.items(), key=lambda x: -abs(x[1])):
            if name != 'intercept' and abs(val) > 0.0001:
                print(f'    {name:<30} {val:+.4f}')
else:
    print(f'  Architecture:   v2.0 Single/Ensemble GBM')
    print(f'  Test IC:        {result.get("test_ic", "n/a")}')
    print(f'  MSE IC:         {result.get("mse_ic", "n/a")}')
    print(f'  Rank IC:        {result.get("rank_ic", "n/a")}')
    print(f'  Ensemble IC:    {result.get("ensemble_ic", "n/a")}')
    wf = result.get('walk_forward', {})
    wf_status = 'PASS' if wf.get('passes_wf') else 'FAIL/skipped'
    print(f'  Walk-forward:   {wf_status}  (median_IC={wf.get("median_ic", "n/a")})')

print(f'  Promoted:       {result.get("promoted", "n/a")}')
print(f'  Promoted mode:  {result.get("promoted_mode", "n/a")}')
print(f'  Elapsed:        {result.get("elapsed_s", "n/a")}s')
print(f'  Slim cache:     {result.get("slim_cache_tickers", "n/a")} tickers')
print('=' * 60)
PYEOF
TRAIN
)" "${MAX_RUNTIME_SECONDS}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Training complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"

# CloudWatch heartbeat on successful completion (unchanged).
aws cloudwatch put-metric-data \
  --namespace "AlphaEngine" \
  --metric-name "Heartbeat" \
  --dimensions "Process=predictor-training" \
  --value 1 --unit "Count" \
  --region "${AWS_REGION:-us-east-1}" 2>/dev/null \
  && echo "Heartbeat emitted: predictor-training" \
  || echo "WARNING: Failed to emit heartbeat (non-fatal)"
