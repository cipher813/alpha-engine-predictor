#!/bin/bash
# rollback.sh — Shift the 'live' alias back to the previous Lambda version.
#
# Usage: bash infrastructure/rollback.sh
set -euo pipefail

LAMBDA_FUNCTION="alpha-engine-predictor-inference"
AWS_REGION="${AWS_REGION:-us-east-1}"

CURRENT=$(aws lambda get-alias \
    --function-name "$LAMBDA_FUNCTION" \
    --name live \
    --query "FunctionVersion" --output text \
    --region "$AWS_REGION")

if [ "$CURRENT" -le 1 ]; then
    echo "Cannot rollback: current version is $CURRENT (no prior version)"
    exit 1
fi

PREV=$((CURRENT - 1))

aws lambda update-alias \
    --function-name "$LAMBDA_FUNCTION" \
    --name live \
    --function-version "$PREV" \
    --region "$AWS_REGION" > /dev/null

echo "Rolled back: live → version $PREV (was $CURRENT)"
echo "To verify: aws lambda get-alias --function-name $LAMBDA_FUNCTION --name live --query FunctionVersion --output text --region $AWS_REGION"
