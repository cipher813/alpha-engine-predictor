#!/usr/bin/env bash
# infrastructure/setup-cloudwatch-alarms.sh — Create CloudWatch alarms for ALL scheduled processes.
#
# Covers every automated process in the Alpha Engine system:
#
#   LAMBDA FUNCTIONS (error alarms + dead-man switches):
#     - alpha-engine-predictor-inference  (daily weekdays 6:10 AM PT)
#     - alpha-engine-research-runner      (weekly Saturday 06:00 UTC)
#     - alpha-engine-research-alerts      (every 30 min during market hours Mon-Fri)
#
#   EC2 PROCESSES (custom heartbeat metrics):
#     - executor-morning     (trading EC2, on boot ~6:20 AM PT weekdays)
#     - executor-daemon-start (trading EC2, on boot)
#     - executor-eod         (trading EC2, 1:20 PM PT weekdays)
#
#   SPOT INSTANCES (custom heartbeat metrics):
#     - backtester           (spot instance, weekly Saturday 08:00 UTC)
#     - predictor-training   (spot instance, weekly Saturday 07:00 UTC)
#
#   CRON JOBS (custom heartbeat metrics):
#     - rag-ingestion        (micro EC2, weekly Saturday 05:00 UTC)
#
# Heartbeat metrics are emitted by emit-heartbeat.sh calls in systemd services,
# spot scripts, and cron jobs. Dead-man switch alarms fire when heartbeats stop.
#
# Idempotent — safe to re-run. Existing resources are reused.
#
# Usage:
#   ./infrastructure/setup-cloudwatch-alarms.sh <email>
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
SNS_TOPIC_NAME="alpha-engine-alerts"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <alert-email-address>"
  exit 1
fi
ALERT_EMAIL="$1"

echo "==> Setting up CloudWatch alarms for ALL Alpha Engine processes"
echo "    Region: $AWS_REGION"
echo ""

# ── SNS Topic ───────────────────────────────────────────────────────────────
echo "==> Creating SNS topic: $SNS_TOPIC_NAME"
TOPIC_ARN=$(aws sns create-topic \
  --name "$SNS_TOPIC_NAME" \
  --query "TopicArn" --output text \
  --region "$AWS_REGION")
echo "  Topic ARN: $TOPIC_ARN"

echo "==> Subscribing $ALERT_EMAIL to topic"
aws sns subscribe \
  --topic-arn "$TOPIC_ARN" \
  --protocol email \
  --notification-endpoint "$ALERT_EMAIL" \
  --region "$AWS_REGION" > /dev/null
echo "  Subscription created (check inbox to confirm if new)"

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: LAMBDA ERROR ALARMS
# ════════════════════════════════════════════════════════════════════════════

LAMBDA_FUNCTIONS=(
  "alpha-engine-predictor-inference"
  "alpha-engine-research-runner"
  "alpha-engine-research-alerts"
)

for FUNC in "${LAMBDA_FUNCTIONS[@]}"; do
  ALARM_NAME="${FUNC}-errors"
  echo ""
  echo "==> Lambda error alarm: $ALARM_NAME"
  aws cloudwatch put-metric-alarm \
    --alarm-name "$ALARM_NAME" \
    --alarm-description "Lambda $FUNC returned 1+ errors in 5 min" \
    --namespace "AWS/Lambda" \
    --metric-name "Errors" \
    --dimensions "Name=FunctionName,Value=$FUNC" \
    --statistic "Sum" \
    --period 300 \
    --evaluation-periods 1 \
    --threshold 1 \
    --comparison-operator "GreaterThanOrEqualToThreshold" \
    --alarm-actions "$TOPIC_ARN" \
    --treat-missing-data "notBreaching" \
    --region "$AWS_REGION"
  echo "  Created: $ALARM_NAME"
done

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: LAMBDA DEAD-MAN SWITCHES (tuned per schedule frequency)
# ════════════════════════════════════════════════════════════════════════════

# Predictor inference: daily weekdays — 24h window
echo ""
echo "==> Lambda dead-man: predictor-inference (24h)"
aws cloudwatch put-metric-alarm \
  --alarm-name "alpha-engine-predictor-inference-no-invocations" \
  --alarm-description "Predictor inference not invoked in 24h — EventBridge may be broken" \
  --namespace "AWS/Lambda" \
  --metric-name "Invocations" \
  --dimensions "Name=FunctionName,Value=alpha-engine-predictor-inference" \
  --statistic "Sum" \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 1 \
  --comparison-operator "LessThanThreshold" \
  --alarm-actions "$TOPIC_ARN" \
  --treat-missing-data "breaching" \
  --region "$AWS_REGION"

# Research runner: weekly Saturday — 8 day window
echo "==> Lambda dead-man: research-runner (8 days)"
aws cloudwatch put-metric-alarm \
  --alarm-name "alpha-engine-research-runner-no-invocations" \
  --alarm-description "Research runner not invoked in 8 days — weekly schedule may be broken" \
  --namespace "AWS/Lambda" \
  --metric-name "Invocations" \
  --dimensions "Name=FunctionName,Value=alpha-engine-research-runner" \
  --statistic "Sum" \
  --period 86400 \
  --evaluation-periods 7 \
  --threshold 1 \
  --comparison-operator "LessThanThreshold" \
  --alarm-actions "$TOPIC_ARN" \
  --treat-missing-data "breaching" \
  --region "$AWS_REGION"

# Research alerts: intraday every 30 min — 2h window (market hours only, notBreaching outside)
echo "==> Lambda dead-man: research-alerts (2h)"
aws cloudwatch put-metric-alarm \
  --alarm-name "alpha-engine-research-alerts-no-invocations" \
  --alarm-description "Price alerts not invoked in 2h — schedule may be broken" \
  --namespace "AWS/Lambda" \
  --metric-name "Invocations" \
  --dimensions "Name=FunctionName,Value=alpha-engine-research-alerts" \
  --statistic "Sum" \
  --period 7200 \
  --evaluation-periods 1 \
  --threshold 1 \
  --comparison-operator "LessThanThreshold" \
  --alarm-actions "$TOPIC_ARN" \
  --treat-missing-data "notBreaching" \
  --region "$AWS_REGION"

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: EC2 PROCESS HEARTBEAT ALARMS (custom metrics)
# ════════════════════════════════════════════════════════════════════════════
# These fire when emit-heartbeat.sh stops being called by systemd services.
# All use namespace "AlphaEngine", MetricName "Heartbeat", dimension "Process".

# Executor morning batch: daily weekdays ~6:20 AM PT — 24h window
echo ""
echo "==> EC2 heartbeat alarm: executor-morning (24h)"
aws cloudwatch put-metric-alarm \
  --alarm-name "alpha-engine-executor-morning-no-heartbeat" \
  --alarm-description "Executor morning batch did not complete in 24h" \
  --namespace "AlphaEngine" \
  --metric-name "Heartbeat" \
  --dimensions "Name=Process,Value=executor-morning" \
  --statistic "Sum" \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 1 \
  --comparison-operator "LessThanThreshold" \
  --alarm-actions "$TOPIC_ARN" \
  --treat-missing-data "breaching" \
  --region "$AWS_REGION"

# Executor daemon start: daily weekdays — 24h window
echo "==> EC2 heartbeat alarm: executor-daemon-start (24h)"
aws cloudwatch put-metric-alarm \
  --alarm-name "alpha-engine-executor-daemon-no-heartbeat" \
  --alarm-description "Executor daemon did not start in 24h" \
  --namespace "AlphaEngine" \
  --metric-name "Heartbeat" \
  --dimensions "Name=Process,Value=executor-daemon-start" \
  --statistic "Sum" \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 1 \
  --comparison-operator "LessThanThreshold" \
  --alarm-actions "$TOPIC_ARN" \
  --treat-missing-data "breaching" \
  --region "$AWS_REGION"

# Executor EOD: daily weekdays 1:20 PM PT — 24h window
echo "==> EC2 heartbeat alarm: executor-eod (24h)"
aws cloudwatch put-metric-alarm \
  --alarm-name "alpha-engine-executor-eod-no-heartbeat" \
  --alarm-description "EOD reconciliation did not complete in 24h" \
  --namespace "AlphaEngine" \
  --metric-name "Heartbeat" \
  --dimensions "Name=Process,Value=executor-eod" \
  --statistic "Sum" \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 1 \
  --comparison-operator "LessThanThreshold" \
  --alarm-actions "$TOPIC_ARN" \
  --treat-missing-data "breaching" \
  --region "$AWS_REGION"

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: SPOT INSTANCE + CRON HEARTBEAT ALARMS (weekly processes)
# ════════════════════════════════════════════════════════════════════════════

# Backtester: weekly Saturday — 8 day window
echo ""
echo "==> Spot heartbeat alarm: backtester (8 days)"
aws cloudwatch put-metric-alarm \
  --alarm-name "alpha-engine-backtester-no-heartbeat" \
  --alarm-description "Backtester spot instance did not complete in 8 days" \
  --namespace "AlphaEngine" \
  --metric-name "Heartbeat" \
  --dimensions "Name=Process,Value=backtester" \
  --statistic "Sum" \
  --period 86400 \
  --evaluation-periods 7 \
  --threshold 1 \
  --comparison-operator "LessThanThreshold" \
  --alarm-actions "$TOPIC_ARN" \
  --treat-missing-data "breaching" \
  --region "$AWS_REGION"

# Predictor training: weekly Saturday — 8 day window
echo "==> Spot heartbeat alarm: predictor-training (8 days)"
aws cloudwatch put-metric-alarm \
  --alarm-name "alpha-engine-predictor-training-no-heartbeat" \
  --alarm-description "Predictor training spot instance did not complete in 8 days" \
  --namespace "AlphaEngine" \
  --metric-name "Heartbeat" \
  --dimensions "Name=Process,Value=predictor-training" \
  --statistic "Sum" \
  --period 86400 \
  --evaluation-periods 7 \
  --threshold 1 \
  --comparison-operator "LessThanThreshold" \
  --alarm-actions "$TOPIC_ARN" \
  --treat-missing-data "breaching" \
  --region "$AWS_REGION"

# RAG ingestion: weekly Saturday 05:00 UTC — 8 day window
echo "==> Cron heartbeat alarm: rag-ingestion (8 days)"
aws cloudwatch put-metric-alarm \
  --alarm-name "alpha-engine-rag-ingestion-no-heartbeat" \
  --alarm-description "RAG ingestion did not complete in 8 days" \
  --namespace "AlphaEngine" \
  --metric-name "Heartbeat" \
  --dimensions "Name=Process,Value=rag-ingestion" \
  --statistic "Sum" \
  --period 86400 \
  --evaluation-periods 7 \
  --threshold 1 \
  --comparison-operator "LessThanThreshold" \
  --alarm-actions "$TOPIC_ARN" \
  --treat-missing-data "breaching" \
  --region "$AWS_REGION"

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════

echo ""
echo "==> All alarms created successfully!"
echo ""
echo "Lambda error alarms (3):"
echo "  - alpha-engine-predictor-inference-errors"
echo "  - alpha-engine-research-runner-errors"
echo "  - alpha-engine-research-alerts-errors"
echo ""
echo "Lambda dead-man switches (3):"
echo "  - alpha-engine-predictor-inference-no-invocations  (24h)"
echo "  - alpha-engine-research-runner-no-invocations      (8 days)"
echo "  - alpha-engine-research-alerts-no-invocations      (2h)"
echo ""
echo "EC2 heartbeat alarms (3):"
echo "  - alpha-engine-executor-morning-no-heartbeat       (24h)"
echo "  - alpha-engine-executor-daemon-no-heartbeat        (24h)"
echo "  - alpha-engine-executor-eod-no-heartbeat           (24h)"
echo ""
echo "Spot/cron heartbeat alarms (3):"
echo "  - alpha-engine-backtester-no-heartbeat             (8 days)"
echo "  - alpha-engine-predictor-training-no-heartbeat     (8 days)"
echo "  - alpha-engine-rag-ingestion-no-heartbeat          (8 days)"
echo ""
echo "Total: 12 alarms across all processes"
echo ""
echo "IMPORTANT: Confirm the SNS email subscription in your inbox."
echo ""
echo "To verify:"
echo "  aws cloudwatch describe-alarms --alarm-name-prefix alpha-engine --query 'MetricAlarms[].AlarmName' --output table --region $AWS_REGION"
