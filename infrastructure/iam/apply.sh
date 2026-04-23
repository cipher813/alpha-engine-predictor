#!/usr/bin/env bash
#
# apply.sh — Apply IAM inline policies in this directory to their target roles.
#
# Each JSON file is a role policy document. Filename (minus .json) is the
# IAM role name. The policy name is derived by stripping a trailing "-role"
# suffix (if present) and appending "-policy", so the convention matches the
# historical inline-policy names already attached to alpha-engine roles:
#
#   alpha-engine-predictor-role.json
#     role name:   alpha-engine-predictor-role
#     policy name: alpha-engine-predictor-policy
#
# Idempotent — safe to re-run. put-role-policy replaces the full policy body.
#
# Usage:
#   ./infrastructure/iam/apply.sh                          # apply every policy
#   ./infrastructure/iam/apply.sh alpha-engine-predictor-role   # one role
#   ./infrastructure/iam/apply.sh --dry-run                # print planned cmds
#
# Prerequisites:
#   - AWS CLI with iam:PutRolePolicy on the target roles
#   - Target IAM roles already exist (this script only writes inline policies)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"

DRY_RUN=0
TARGET_ROLE=""

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) TARGET_ROLE="$arg" ;;
  esac
done

derive_policy_name() {
  local role="$1"
  echo "${role%-role}-policy"
}

apply_one() {
  local file="$1"
  local role
  role="$(basename "$file" .json)"
  local policy_name
  policy_name="$(derive_policy_name "$role")"

  if ! python3 -c "import json; json.load(open('$file'))" 2>/dev/null; then
    echo "ERROR: $file is not valid JSON — skipping" >&2
    return 1
  fi

  echo "Applying $file -> role=$role policy=$policy_name"
  if [ "$DRY_RUN" = 1 ]; then
    echo "  [dry-run] aws iam put-role-policy --role-name $role --policy-name $policy_name --policy-document file://$file --region $REGION"
    return 0
  fi

  aws iam put-role-policy --role-name "$role" --policy-name "$policy_name" --policy-document "file://$file" --region "$REGION"
  echo "  OK"
}

cd "$SCRIPT_DIR"

if [ -n "$TARGET_ROLE" ]; then
  file="${TARGET_ROLE}.json"
  if [ ! -f "$file" ]; then
    echo "ERROR: $file not found in $SCRIPT_DIR" >&2
    exit 1
  fi
  apply_one "$file"
else
  shopt -s nullglob
  files=( *.json )
  if [ ${#files[@]} -eq 0 ]; then
    echo "No .json policy files found in $SCRIPT_DIR"
    exit 0
  fi
  for file in "${files[@]}"; do
    apply_one "$file"
  done
fi
