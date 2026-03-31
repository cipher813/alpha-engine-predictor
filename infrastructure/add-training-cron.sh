#!/bin/bash
# Register the weekly predictor training cron job on the always-on EC2 instance.
# Safe to run multiple times — replaces existing entry.
#
# Schedule: Saturdays at 07:00 UTC (Saturday ~12am PT)
# Runs after Research Lambda (06:00 UTC), before Backtester (08:00 UTC).
# Independent of Research — staggered to avoid yfinance rate limits.
#
# Launches a spot instance via spot_train.sh for cost efficiency (~$0.06/hr).
#
# Secrets sourced from ~/.alpha-engine.env (shared with executor/backtester).
#
# Usage:
#   bash infrastructure/add-training-cron.sh

set -euo pipefail

ENV_FILE="/home/ec2-user/.alpha-engine.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: ${ENV_FILE} not found."
    echo "Create it with EMAIL_SENDER, EMAIL_RECIPIENTS, GMAIL_APP_PASSWORD, then chmod 600."
    exit 1
fi

SOURCE_ENV=". ${ENV_FILE} &&"

CRON_LINE="0 7 * * 6  cd /home/ec2-user/alpha-engine-predictor && git pull --ff-only >> /var/log/predictor-training.log 2>&1 && ${SOURCE_ENV} bash infrastructure/spot_train.sh --full-only >> /var/log/predictor-training.log 2>&1"

# Remove existing predictor training entry, then add new one
EXISTING=$(crontab -l 2>/dev/null || true)
FILTERED=$(echo "$EXISTING" | grep -v "predictor.*train\|spot_train" || true)

{
    echo "$FILTERED"
    echo "$CRON_LINE"
} | crontab -

echo "Predictor training cron job registered: Saturdays 07:00 UTC"
echo "  Mode: spot instance (launched from always-on EC2)"
echo "  Secrets: sourced from ${ENV_FILE}"
echo "  Log: /var/log/predictor-training.log"
echo ""
echo "Current crontab:"
crontab -l
