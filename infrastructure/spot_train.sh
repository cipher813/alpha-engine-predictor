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
#   1. Request a spot instance (r5.large ≈ $0.04/hr spot; ≥8 GiB RAM)
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
# config#1066 — experiment package. config.py searches
# ~/alpha-engine-config/experiments/$ALPHA_ENGINE_EXPERIMENT_ID/predictor/predictor.yaml
# FIRST (the 2026-06-12 experiment-package adoption), then the legacy
# config/predictor.yaml. The spot is a bare predictor clone with NO
# alpha-engine-config tree, so the experiment path is absent and resolution
# silently falls through to config/predictor.yaml — a coincidence that was the
# 6/13 inert-rotation fragility (MODEL_SPECS empty → 0 challengers trained).
# We pin the id here, EXPORT it into every spot heredoc, and stage the yaml to
# BOTH the experiment-package path AND config/predictor.yaml on the spot so
# config.py resolves DETERMINISTICALLY to the staged, MODEL_SPECS-populated yaml
# via the SAME path it uses on the always-on box. Default "reference" matches
# config.py's own _EXPERIMENT_ID default.
ALPHA_ENGINE_EXPERIMENT_ID="${ALPHA_ENGINE_EXPERIMENT_ID:-reference}"
BRANCH="${BRANCH:-main}"
# Defer-email propagation: the Saturday SF exports PREDICTOR_DEFER_TRAINING_EMAIL
# before invoking this script for the --full-only champion retrain, so the base
# retrain defers its per-run training email to the consolidated model-zoo digest
# (the --model-zoo-weekly workload sends the digest). The full-training heredoc
# is a single-quoted heredoc and cannot interpolate a bash var, so we compute an
# export line HERE and prepend it to the heredoc body via string concatenation
# (an interpolating prefix + the quoted body). Empty when the var is unset, so a
# bare full-only run is byte-equivalent to before. Keep this single-line and
# paren/apostrophe-free per the bash 3.2 run_ssm note.
if [ -n "${PREDICTOR_DEFER_TRAINING_EMAIL:-}" ]; then
  DEFER_EMAIL_EXPORT="export PREDICTOR_DEFER_TRAINING_EMAIL=${PREDICTOR_DEFER_TRAINING_EMAIL}"$'\n'
else
  DEFER_EMAIL_EXPORT=""
fi
# Capacity-resilient instance-type fallback set (2026-05-22 incident:
# spot launches in single-AZ subnet-e07166ec/us-east-1f hit
# InsufficientInstanceCapacity). Order = preference; the lib CLI tries
# each in turn until one launches.
#
# 2026-06-06 — memory-optimized (≥8 GiB, all 2 vCPU). The prior set
# (c5.large/c6i.large/c5a.large = 4 GiB) OOM-killed full-training on the
# Saturday SF: the meta-trainer's peak RSS now exceeds 4 GiB (universe +
# history growth plus the observe-only canonical-alpha matrix), so the
# rotation picked c5.large (4 GiB) and the kernel SIGKILL'd the process
# right after regime-data load. This is the SECOND OOM on a 4 GiB box
# (first: 2026-04-28, addressed by the meta_trainer.py streaming refactor;
# data growth since re-crossed 4 GiB). Lead with r5.large (16 GiB) for
# ~4× headroom over the failing footprint; m5.large (8 GiB) is the
# last-resort capacity fallback. The old "steady-state ~1-1.5 GB" note
# was stale — see test_meta_trainer_streaming.py for the peak-RSS context.
INSTANCE_TYPES="${INSTANCE_TYPES:-r5.large,r5a.large,r6i.large,m5.large}"
INSTANCE_TYPE=""  # backward-compat: --instance-type X collapses INSTANCE_TYPES to single value
AMI_ID="ami-0c421724a94bba6d6"  # Amazon Linux 2023 x86_64 (Python 3.12, SSM agent preinstalled)
# Spot-side watchdog budget: meta-trainer typically completes 40-70 min;
# include pip install + smoke + full run. 90 min with headroom.
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-5400}"
# KEY_NAME is still passed to run-instances so emergency SSH stays possible
# during the validation window (the SG's port 22 ingress is dropped only in
# the migration's PR 3, after this PR validates against a Saturday SF).
KEY_NAME="alpha-engine-key"
SECURITY_GROUP="sg-03cd3c4bd91e610b0"
# All 6 default-VPC subnets across us-east-1{a..f}. The lib CLI rotates
# across this list on capacity error. Same VPC + same SG as the data +
# backtester spots; lockstep with their launchers.
SUBNETS="${SUBNETS:-subnet-a61ec0fb,subnet-1e58307a,subnet-789d3857,subnet-c670118d,subnet-7cff7c43,subnet-e07166ec}"
IAM_PROFILE="alpha-engine-executor-profile"
# Lib CLI path: ae-dashboard is the SSM target for the PredictorTraining
# state; the dispatcher's .venv has alpha-engine-lib installed (see
# deploy-on-merge.sh in the dashboard repo).
LIB_PYTHON="${LIB_PYTHON:-/home/ec2-user/alpha-engine-dashboard/.venv/bin/python}"
REPO_URL="https://github.com/nousergon/crucible-predictor.git"  # public repo, no auth

# Parse flags
MODE="both"  # both | full-only | smoke-only | preflight-only | model-zoo-weekly | model-zoo-spec | model-zoo-select
MODEL_ZOO_SPEC_ID=""  # set by --model-zoo-spec <id>
while [ $# -gt 0 ]; do
  case "$1" in
    --full-only) MODE="full-only" ;;
    --smoke-only) MODE="smoke-only" ;;
    --preflight-only) MODE="preflight-only" ;;
    # L4544: train the weekly model-zoo rotation + immediate CPCV selection
    # (challenger-first; champion retrain is the separate --full-only run).
    --model-zoo-weekly) MODE="model-zoo-weekly" ;;
    # config#1083 PARALLEL fan-out: train exactly ONE challenger spec on this
    # spot (the SF ModelZooTrainMap launches one spot per spec id).
    --model-zoo-spec) shift; MODE="model-zoo-spec"; MODEL_ZOO_SPEC_ID="$1" ;;
    # config#1083 PARALLEL fan-out: run ONLY the selection over whatever specs
    # registered for the date (the SF ModelZooSelect joins after the Map).
    --model-zoo-select) MODE="model-zoo-select" ;;
    --instance-type) shift; INSTANCE_TYPE="$1" ;;
  esac
  shift
done

# config#1083 — fail loud if --model-zoo-spec was given without a spec id (the
# Map iteration must carry an explicit spec; a blank id is a wiring bug).
if [ "$MODE" = "model-zoo-spec" ] && [ -z "$MODEL_ZOO_SPEC_ID" ]; then
  echo "ERROR: --model-zoo-spec requires a spec id (got empty)" >&2
  exit 2
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  GBM Spot Training — $(date +%Y-%m-%d)  (SSM transport)"
echo "═══════════════════════════════════════════════════════════════"
if [ -n "$INSTANCE_TYPE" ]; then
  INSTANCE_TYPES="$INSTANCE_TYPE"  # --instance-type X collapses to single value
fi
echo "  Instance types: $INSTANCE_TYPES"
echo "  Subnets       : $SUBNETS"
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
# Capacity-resilient launch via nousergon_lib.ec2_spot (lib v0.26.0+).
# Rotates (instance_type × subnet) on InsufficientInstanceCapacity etc.
# Replaces the broken-by-design hardcoded single-subnet + single-instance-type
# pattern (2026-05-22 incident — Evaluator failed in sibling backtester spot).
echo "==> Requesting spot instance (lib CLI rotation: types=[$INSTANCE_TYPES], subnets=[$SUBNETS])..."
INSTANCE_ID=$("$LIB_PYTHON" -m nousergon_lib.ec2_spot launch \
  --types "$INSTANCE_TYPES" \
  --subnets "$SUBNETS" \
  --image-id "$AMI_ID" \
  --key-name "$KEY_NAME" \
  --security-group "$SECURITY_GROUP" \
  --iam-profile "$IAM_PROFILE" \
  --name "alpha-engine-gbm-train-$(date +%Y%m%d)" \
  --region "$AWS_REGION")
ec2_spot_rc=$?
if [ "$ec2_spot_rc" -ne 0 ] || [ -z "$INSTANCE_ID" ]; then
  if [ "$ec2_spot_rc" -eq 64 ]; then
    echo "ERROR: capacity exhausted across all instance_type × subnet combinations" >&2
  fi
  exit "${ec2_spot_rc:-1}"
fi
echo "  Instance ID: $INSTANCE_ID"

RUN_ID="$(date +%Y%m%dT%H%M%SZ)-${INSTANCE_ID}"
S3_STAGING_PREFIX="tmp/spot_train/${RUN_ID}"
S3_STAGING="s3://${S3_BUCKET}/${S3_STAGING_PREFIX}"

# Cleanup — always terminate the instance + remove the S3 staging prefix.
# (S3 lifecycle on tmp/ is the belt-and-suspenders if the trap never fires.)
cleanup() {
  echo ""
  # Belt-and-suspenders (STEP 3): BEFORE terminating the spot, confirm where
  # each workload's spot-side log landed in S3. The spot SELF-SHIP via
  # nousergon_lib.ssm_log_capture (each workload heredoc) is PRIMARY — this
  # is only a bounded best-effort confirmation + a one-hop pointer in the
  # dispatcher log so an operator triaging a failure (esp. an OOM RC=-1 where
  # SSM get-command-invocation returns empty) can find the full log immediately.
  # Bounded: a single short `aws s3 ls` per slug, all failures swallowed, never
  # blocks teardown. Key shape: _ssm_logs/{slug}/{YYYY-MM-DD}/{host}-{HHMMSSZ}.log
  # (nousergon_lib.ssm_log_capture._exit_key). The exit-time UTC date is the
  # key component; on a run straddling UTC midnight the log lands under the exit
  # date, so probe today's date.
  local _logdate_now _hit
  _logdate_now="$(date -u +%Y-%m-%d)"
  echo "==> Confirming spot-side workload logs in s3://${S3_BUCKET}/_ssm_logs/ ..."
  for _slug in spot-smoke spot-model-zoo-weekly spot-model-zoo-spec spot-model-zoo-select spot-full-training; do
    _hit="$(aws s3 ls "s3://${S3_BUCKET}/_ssm_logs/${_slug}/${_logdate_now}/" --region "$AWS_REGION" 2>/dev/null | tail -1 || true)"
    if [ -n "$_hit" ]; then
      echo "    ${_slug}: s3://${S3_BUCKET}/_ssm_logs/${_slug}/${_logdate_now}/$(echo "$_hit" | awk '{print $NF}')"
    fi
  done
  echo "    (spot logs above are the FULL workload stdout/stderr — primary diagnostic on RC=-1/OOM)"
  echo "    Failure diagnostics record (if any): s3://${S3_BUCKET}/_spot_diagnostics/ae-predictor/${_logdate_now}.json"
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
# **2026-05-27 — Lib chokepoint lift (ROADMAP L342 PR 4).** This helper
# was the 54-line inline ``aws ssm send-command`` + poll + stream + S3
# capture bash function that L342 was explicitly chartered to retire.
# The lib equivalent ships in ``nousergon_lib.ssm_dispatcher`` (lib
# v0.35.0+, [#73](https://github.com/nousergon/nousergon-lib/pull/73))
# with identical contract: base64-wrap → SendCommand → poll → stream
# StandardOutputContent delta → fetch StandardErrorContent on terminal
# non-Success → propagate exit. Adds InvocationDoesNotExist
# registration-grace handling (2026-05-23 SF event-16 substrate weakness)
# that the pre-lift inline form lacked.
#
# The calling convention is unchanged so the existing 5 call sites
# (bootstrap / deps / preflight-only / smoke / full-training) need no
# rewrite. The body is piped to the lib CLI's ``--script-stdin``, which
# reads it verbatim (no command-substitution scanning) — matching the
# pattern alpha-engine-data PR 2 (#330) and alpha-engine-backtester
# PR 3 (#251) adopted for their migrations.
# L394 cascade: --diagnostics-bucket + --diagnostics-prefix activate the
# lib v0.39.0 chokepoint that writes a JSON failure record (status +
# command_id + 4KB stdout/stderr tails + instance_id) to
# s3://${S3_BUCKET}/_spot_diagnostics/ae-predictor/{YYYY-MM-DD}.json on
# terminal non-Success. Best-effort write inside the lib — S3 failure
# swallowed; inner SSM exit always preserved. Substrate is failure-only
# (no-op on Success). Per-repo subprefix discriminates cascade A
# (ae-data) + cascade B (ae-backtester) sibling writes — lib's
# {date}.json key shape would otherwise clobber within a shared prefix.
run_ssm() {
  local description="$1" script="$2" timeout_s="${3:-3600}"
  printf '%s' "$script" | "$LIB_PYTHON" -m nousergon_lib.ssm_dispatcher run \
    --instance-id "$INSTANCE_ID" \
    --description "predictor-training: $description" \
    --timeout "$timeout_s" \
    --output-bucket "$S3_BUCKET" \
    --output-key-prefix "${S3_STAGING_PREFIX}/ssm-output" \
    --region "$AWS_REGION" \
    --diagnostics-bucket "$S3_BUCKET" \
    --diagnostics-prefix "_spot_diagnostics/ae-predictor" \
    --script-stdin
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
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=${ALPHA_ENGINE_EXPERIMENT_ID}

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
# config#1066 — stage the yaml to BOTH paths config.py searches: the
# experiment-package path it tries FIRST and the legacy config/predictor.yaml
# fallback. Both copies are byte-identical from the same staged source, so
# MODEL_SPECS populates deterministically regardless of which path wins.
mkdir -p /home/ec2-user/alpha-engine-config/experiments/${ALPHA_ENGINE_EXPERIMENT_ID}/predictor
aws s3 cp ${S3_STAGING}/predictor.yaml /home/ec2-user/alpha-engine-config/experiments/${ALPHA_ENGINE_EXPERIMENT_ID}/predictor/predictor.yaml --region ${AWS_REGION}
aws s3 cp ${S3_STAGING}/predictor.yaml /home/ec2-user/predictor/config/predictor.yaml --region ${AWS_REGION}
echo "Bootstrap complete: repo cloned, predictor.yaml staged to experiment package ${ALPHA_ENGINE_EXPERIMENT_ID} plus config fallback."
BOOTSTRAP
)" 600

# ── Dependencies ──────────────────────────────────────────────────────────────
echo "==> Installing Python dependencies..."
run_ssm "deps" "$(cat <<'DEPS'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference
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
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference
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
from training import model_zoo  # noqa: F401  (L4544 rotation path; import-only)
from training.preflight import TrainingPreflight
log.info('       OK — alpha_engine_lib + training.train_handler + model_zoo import clean')

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
# model-zoo modes skip the champion smoke (they train/select challenger variants).
if [ "$MODE" != "full-only" ] && [ "$MODE" != "model-zoo-weekly" ] && [ "$MODE" != "model-zoo-spec" ] && [ "$MODE" != "model-zoo-select" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  SMOKE TEST (dry_run=True)"
  echo "═══════════════════════════════════════════════════════════════"
  run_ssm "smoke" "$(cat <<'SMOKE'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
# Spot-side log durability — the python workload below ran inline via $PY - so
# its stdout/stderr lived ONLY in SSM get-command-invocation, which returns
# EMPTY when the spot dies mid-run e.g. OOM RC=-1 and is destroyed when the
# dispatcher cleanup EXIT trap terminates the box. Route the workload through
# the lib chokepoint nousergon_lib.ssm_log_capture: it tees combined
# stdout+stderr to a spot-local logfile AND ships that logfile to S3 on EXIT
# including SIGKILL of the workload BEFORE the dispatcher tears the box down,
# then propagates the workload exit code verbatim so set -eo pipefail and the
# SF still see the real failure. The wrapper is a lightweight separate process
# so the kernel OOM-killer reaps the heavy workload subprocess, not the shipper.
# NOTE keep this region free of apostrophes and parens: bash 3.2 scans even a
# quoted heredoc body for the closing paren of the enclosing run_ssm command
# substitution.
cat > /tmp/spot-smoke.py <<'PYEOF'
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
$PY -m nousergon_lib.ssm_log_capture run --slug spot-smoke --log /var/log/spot-smoke.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-smoke.py
SMOKE
)" 1800
  echo "Smoke test complete."
  if [ "$MODE" = "smoke-only" ]; then
    echo "==> Smoke-only mode — skipping full training."
    exit 0
  fi
fi

# ── Model-zoo weekly rotation + immediate CPCV selection (L4544) ──────────────
# Trains the N stalest challenger specs, ranks them by leak-free CPCV, writes a
# leaderboard, and (only if MODEL_ZOO_AUTO_PROMOTE_WINNER) promotes the winner.
# Challenger-first + live-contract-restore are enforced inside model_zoo, so this
# never disturbs the live champion. Runs INSTEAD OF the champion retrain (that's
# the separate --full-only state); exits 0 when done.
if [ "$MODE" = "model-zoo-weekly" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  MODEL-ZOO WEEKLY ROTATION + SELECT (observe-first by default)"
  echo "═══════════════════════════════════════════════════════════════"
  run_ssm "model-zoo-weekly" "$(cat <<'ZOO'
set -eo pipefail
# config#1066 — pin ALPHA_ENGINE_EXPERIMENT_ID so config.py loads the staged
# experiment-package yaml, MODEL_SPECS populates, and the rotation trains
# challengers. The probe below logs the resolved path + count for diagnosis.
# NOTE keep this heredoc free of apostrophes and parens: bash 3.2 scans even a
# quoted heredoc body for the closing paren of the enclosing run_ssm command
# substitution.
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
# Spot-side log durability — see the smoke step comment. Route the workload
# through nousergon_lib.ssm_log_capture so the model-zoo log reaches S3 on
# EXIT including OOM-kill before the dispatcher terminates the box. Paren-free
# and apostrophe-free per the bash 3.2 note above.
cat > /tmp/spot-model-zoo-weekly.py <<'PYEOF'
import os, sys
sys.path.insert(0, '.')
os.environ.setdefault('S3_BUCKET', os.environ.get('S3_BUCKET', 'alpha-engine-research'))
bucket = os.environ.get('S3_BUCKET', 'alpha-engine-research')

import logging
# Flow-doctor wiring: importing training.model_zoo runs its module-top
# setup_logging (predictor-model-zoo + flow-doctor-model-zoo.yaml: email +
# S3 sink), which clears+reinstalls the root handler. We ALSO call it
# explicitly here BEFORE the import so this entrypoint stays wired even if
# the import order is later changed, and so a config-import crash inside the
# import is captured. Idempotent: setup_logging clears existing handlers.
# NOTE keep this heredoc free of apostrophes per the bash 3.2 note above.
import os.path as _osp
_FD_YAML = _osp.join(_osp.abspath("."), "flow-doctor-model-zoo.yaml")
from alpha_engine_lib.logging import setup_logging
setup_logging("predictor-model-zoo", flow_doctor_yaml=_FD_YAML, exclude_patterns=[])

import config as cfg
from training.model_zoo import run_rotation_and_select

# config#1051 logging probe: pin WHAT the child spot loaded so an empty
# MODEL_SPECS (the 6/13 inert-rotation root cause) is diagnosable from the log.
log = logging.getLogger('model_zoo.spot')
log.info('model_zoo spot probe: MODEL_SPECS=%d  config=%s  ALPHA_ENGINE_EXPERIMENT_ID=%s',
         len(getattr(cfg, 'MODEL_SPECS', [])),
         getattr(cfg, '_CONFIG_PATH', '?'),
         getattr(cfg, '_EXPERIMENT_ID', os.environ.get('ALPHA_ENGINE_EXPERIMENT_ID', 'reference')))

# config#1051: pass a real trading_day so leaderboard / trial_log key on a date,
# not null (the 6/13 leaderboard had date=null). now_dual is backward-looking.
try:
    from alpha_engine_lib.dates import now_dual
    _td = now_dual().trading_day
    date_str = _td.isoformat() if hasattr(_td, 'isoformat') else str(_td)
except Exception:
    log.warning('model_zoo spot: now_dual unavailable — run_rotation_and_select will self-default', exc_info=True)
    date_str = None

budget = int(os.environ.get('MODEL_ZOO_WEEKLY_BUDGET', getattr(cfg, 'MODEL_ZOO_WEEKLY_BUDGET', 3)))
board = run_rotation_and_select(bucket, budget=budget, date_str=date_str)

print()
print('=' * 60)
print('  MODEL-ZOO ROTATION + SELECT')
print('=' * 60)
print(f'  Mode:           {board.get("mode")}')
champ = board.get('champion', {})
print(f'  Champion CPCV:  {champ.get("cpcv_mean_ic")} (fwd={champ.get("forward_days")})')
for c in board.get('candidates', []):
    print(f'    {c.get("spec_id"):<18} cpcv={c.get("cpcv_mean_ic")} fwd={c.get("forward_days")} '
          f'gate={c.get("passes_gate")} eligible={c.get("eligible")} ({c.get("reason")})')
print(f'  Winner:         {board.get("winner_version_id")}')
print(f'  Promoted:       {board.get("promoted")}')
print('=' * 60)
PYEOF
$PY -m nousergon_lib.ssm_log_capture run --slug spot-model-zoo-weekly --log /var/log/spot-model-zoo-weekly.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-model-zoo-weekly.py
ZOO
)" "${MAX_RUNTIME_SECONDS}"

  aws cloudwatch put-metric-data \
    --namespace "AlphaEngine" \
    --metric-name "Heartbeat" \
    --dimensions "Process=predictor-model-zoo" \
    --value 1 --unit "Count" \
    --region "${AWS_REGION:-us-east-1}" 2>/dev/null \
    && echo "Heartbeat emitted: predictor-model-zoo" \
    || echo "WARNING: Failed to emit heartbeat (non-fatal)"

  echo ""
  echo "==> Model-zoo rotation complete. Instance will be terminated."
  exit 0
fi

# ── Model-zoo PARALLEL: train ONE challenger spec (config#1083) ────────────────
# The SF ModelZooTrainMap launches one spot per spec id; this trains exactly that
# spec as a challenger (challenger-first + live-contract restore are enforced
# inside model_zoo.train_one_spec) and exits NON-ZERO only on a real training
# failure — so the Map iteration records THIS spec's failure without aborting
# siblings (the per-spec isolation property). Mirrors the model-zoo-weekly
# workload wrapping: ssm_log_capture ship-on-exit, flow-doctor setup_logging,
# experiment-package staging. No selection / promotion happens here — that's the
# separate ModelZooSelect state (--model-zoo-select).
if [ "$MODE" = "model-zoo-spec" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  MODEL-ZOO TRAIN ONE SPEC: ${MODEL_ZOO_SPEC_ID}"
  echo "═══════════════════════════════════════════════════════════════"
  # Interpolating export prefix so the quoted heredoc body (which must stay
  # paren/apostrophe-free per the bash 3.2 note) reads the spec id from the env.
  MZ_SPEC_EXPORT="export MODEL_ZOO_SPEC_ID=${MODEL_ZOO_SPEC_ID}"$'\n'
  run_ssm "model-zoo-spec" "${MZ_SPEC_EXPORT}$(cat <<'ZOOSPEC'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
# Spot-side log durability + flow-doctor wiring — see the model-zoo-weekly step.
# Paren-free and apostrophe-free per the bash 3.2 note above.
cat > /tmp/spot-model-zoo-spec.py <<'PYEOF'
import os, sys
sys.path.insert(0, '.')
os.environ.setdefault('S3_BUCKET', os.environ.get('S3_BUCKET', 'alpha-engine-research'))
bucket = os.environ.get('S3_BUCKET', 'alpha-engine-research')

import logging
import os.path as _osp
_FD_YAML = _osp.join(_osp.abspath("."), "flow-doctor-model-zoo.yaml")
from alpha_engine_lib.logging import setup_logging
setup_logging("predictor-model-zoo", flow_doctor_yaml=_FD_YAML, exclude_patterns=[])

import config as cfg
from training.model_zoo import train_one_spec

log = logging.getLogger('model_zoo.spot')
spec_id = os.environ.get('MODEL_ZOO_SPEC_ID', '')
log.info('model_zoo train-spec probe: spec=%s  MODEL_SPECS=%d  config=%s  ALPHA_ENGINE_EXPERIMENT_ID=%s',
         spec_id, len(getattr(cfg, 'MODEL_SPECS', [])),
         getattr(cfg, '_CONFIG_PATH', '?'),
         getattr(cfg, '_EXPERIMENT_ID', os.environ.get('ALPHA_ENGINE_EXPERIMENT_ID', 'reference')))
if not spec_id:
    raise SystemExit('MODEL_ZOO_SPEC_ID not set on the spot')

try:
    from alpha_engine_lib.dates import now_dual
    _td = now_dual().trading_day
    date_str = _td.isoformat() if hasattr(_td, 'isoformat') else str(_td)
except Exception:
    log.warning('model_zoo spot: now_dual unavailable', exc_info=True)
    date_str = None

# Raises on a real training failure → non-zero exit → the Map iteration records
# THIS spec failed without aborting siblings.
train_one_spec(spec_id, bucket, date_str=date_str)
print()
print('=' * 60)
print('  MODEL-ZOO TRAIN-SPEC ' + spec_id + ' COMPLETE')
print('=' * 60)
PYEOF
$PY -m nousergon_lib.ssm_log_capture run --slug spot-model-zoo-spec --log /var/log/spot-model-zoo-spec.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-model-zoo-spec.py
ZOOSPEC
)" "${MAX_RUNTIME_SECONDS}"

  aws cloudwatch put-metric-data \
    --namespace "AlphaEngine" \
    --metric-name "Heartbeat" \
    --dimensions "Process=predictor-model-zoo-spec" \
    --value 1 --unit "Count" \
    --region "${AWS_REGION:-us-east-1}" 2>/dev/null \
    && echo "Heartbeat emitted: predictor-model-zoo-spec" \
    || echo "WARNING: Failed to emit heartbeat (non-fatal)"

  echo ""
  echo "==> Model-zoo train-spec ${MODEL_ZOO_SPEC_ID} complete. Instance will be terminated."
  exit 0
fi

# ── Model-zoo PARALLEL: SELECT over the registered specs (config#1083) ─────────
# Runs AFTER the ModelZooTrainMap joins (one spot). Selects over whatever spec-*
# challengers registered for the date (failed Map iterations are simply absent —
# tolerated), writes the leaderboard to BOTH the dated key AND latest.json,
# promotes the winner if MODEL_ZOO_AUTO_PROMOTE_WINNER, and sends the one
# consolidated digest. No training happens here.
if [ "$MODE" = "model-zoo-select" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  MODEL-ZOO SELECT (observe-first by default)"
  echo "═══════════════════════════════════════════════════════════════"
  run_ssm "model-zoo-select" "$(cat <<'ZOOSEL'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
# Spot-side log durability + flow-doctor wiring — see the model-zoo-weekly step.
# Paren-free and apostrophe-free per the bash 3.2 note above.
cat > /tmp/spot-model-zoo-select.py <<'PYEOF'
import os, sys
sys.path.insert(0, '.')
os.environ.setdefault('S3_BUCKET', os.environ.get('S3_BUCKET', 'alpha-engine-research'))
bucket = os.environ.get('S3_BUCKET', 'alpha-engine-research')

import logging
import os.path as _osp
_FD_YAML = _osp.join(_osp.abspath("."), "flow-doctor-model-zoo.yaml")
from alpha_engine_lib.logging import setup_logging
setup_logging("predictor-model-zoo", flow_doctor_yaml=_FD_YAML, exclude_patterns=[])

import config as cfg
from training.model_zoo import run_select_only

log = logging.getLogger('model_zoo.spot')
log.info('model_zoo select probe: MODEL_SPECS=%d  config=%s  ALPHA_ENGINE_EXPERIMENT_ID=%s',
         len(getattr(cfg, 'MODEL_SPECS', [])),
         getattr(cfg, '_CONFIG_PATH', '?'),
         getattr(cfg, '_EXPERIMENT_ID', os.environ.get('ALPHA_ENGINE_EXPERIMENT_ID', 'reference')))

try:
    from alpha_engine_lib.dates import now_dual
    _td = now_dual().trading_day
    date_str = _td.isoformat() if hasattr(_td, 'isoformat') else str(_td)
except Exception:
    log.warning('model_zoo spot: now_dual unavailable - run_select_only will self-default', exc_info=True)
    date_str = None

board = run_select_only(bucket, date_str=date_str)
print()
print('=' * 60)
print('  MODEL-ZOO SELECT')
print('=' * 60)
champ = board.get('champion', {})
print('  Mode:           ' + str(board.get('mode')))
print('  Champion CPCV:  ' + str(champ.get('cpcv_mean_ic')) + ' fwd=' + str(champ.get('forward_days')))
for c in board.get('candidates', []):
    print('    ' + str(c.get('spec_id')) + ' cpcv=' + str(c.get('cpcv_mean_ic')) + ' fwd=' + str(c.get('forward_days')) + ' eligible=' + str(c.get('eligible')) + ' (' + str(c.get('reason')) + ')')
print('  Winner:         ' + str(board.get('winner_version_id')))
print('  Promoted:       ' + str(board.get('promoted')))
print('=' * 60)
PYEOF
$PY -m nousergon_lib.ssm_log_capture run --slug spot-model-zoo-select --log /var/log/spot-model-zoo-select.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-model-zoo-select.py
ZOOSEL
)" "${MAX_RUNTIME_SECONDS}"

  aws cloudwatch put-metric-data \
    --namespace "AlphaEngine" \
    --metric-name "Heartbeat" \
    --dimensions "Process=predictor-model-zoo-select" \
    --value 1 --unit "Count" \
    --region "${AWS_REGION:-us-east-1}" 2>/dev/null \
    && echo "Heartbeat emitted: predictor-model-zoo-select" \
    || echo "WARNING: Failed to emit heartbeat (non-fatal)"

  echo ""
  echo "==> Model-zoo select complete. Instance will be terminated."
  exit 0
fi

# ── Full training (dry_run=False) ─────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  FULL TRAINING (dry_run=False)"
echo "═══════════════════════════════════════════════════════════════"
run_ssm "full-training" "${DEFER_EMAIL_EXPORT}$(cat <<'TRAIN'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1 ALPHA_ENGINE_EXPERIMENT_ID=reference S3_BUCKET=alpha-engine-research
cd /home/ec2-user/predictor
command -v python3.12 >/dev/null && PY=python3.12 || PY=python3
# Spot-side log durability — this is THE workload whose log was lost on the
# off-cycle full-only OOM RC=-1 incident the python ran inline via $PY - so its
# full training log lived only in SSM get-command-invocation which returns empty
# on instance death. Route it through nousergon_lib.ssm_log_capture: tee
# combined stdout+stderr to /var/log/spot-full-training.log AND ship to S3 on
# EXIT including SIGKILL BEFORE the dispatcher cleanup EXIT trap terminates the
# box, propagating the workload exit code so set -eo pipefail and the SF still
# see the real failure. Paren-free and apostrophe-free per the bash 3.2 note.
cat > /tmp/spot-full-training.py <<'PYEOF'
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
$PY -m nousergon_lib.ssm_log_capture run --slug spot-full-training --log /var/log/spot-full-training.log --bucket "$S3_BUCKET" -- $PY /tmp/spot-full-training.py
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
