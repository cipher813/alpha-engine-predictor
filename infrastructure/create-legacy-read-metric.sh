#!/usr/bin/env bash
# infrastructure/create-legacy-read-metric.sh — CloudWatch Logs metric filter for
# [LEGACY_PRICE_READ] markers emitted by inference.stages.load_prices.run().
#
# Purpose: ROADMAP Phase 7d migration gate. The filter counts log lines tagged
# [LEGACY_PRICE_READ] consumer=predictor_inference emitted from the inference
# Lambda. Phase 7a (inference → ArcticDB migration) removes the code paths that
# emit these markers; post-migration the metric should drop to zero.
#
# Emits to custom metric namespace AlphaEngine/Migration, metric name
# legacy_price_read_count, dimension Consumer=predictor_inference. Each matching
# log line increments the metric by 1.
#
# Idempotent — put-metric-filter creates or updates in place.
#
# Usage: ./infrastructure/create-legacy-read-metric.sh

set -euo pipefail

LOG_GROUP="/aws/lambda/alpha-engine-predictor-inference"
FILTER_NAME="legacy-price-read-marker"
NAMESPACE="AlphaEngine/Migration"
METRIC_NAME="legacy_price_read_count"
REGION="${AWS_REGION:-us-east-1}"

echo "Creating metric filter '${FILTER_NAME}' on ${LOG_GROUP} ..."

aws logs put-metric-filter \
  --region "${REGION}" \
  --log-group-name "${LOG_GROUP}" \
  --filter-name "${FILTER_NAME}" \
  --filter-pattern '"[LEGACY_PRICE_READ]"' \
  --metric-transformations \
      "metricName=${METRIC_NAME},metricNamespace=${NAMESPACE},metricValue=1,defaultValue=0,dimensions={Consumer=predictor_inference}"

echo "OK metric filter created/updated."
echo
echo "To inspect after next inference run:"
echo "  aws cloudwatch get-metric-statistics --region ${REGION} \\"
echo "    --namespace ${NAMESPACE} --metric-name ${METRIC_NAME} \\"
echo "    --dimensions Name=Consumer,Value=predictor_inference \\"
echo "    --start-time \$(date -u -v-7d +%Y-%m-%dT%H:%M:%S) \\"
echo "    --end-time \$(date -u +%Y-%m-%dT%H:%M:%S) \\"
echo "    --period 86400 --statistics Sum"
