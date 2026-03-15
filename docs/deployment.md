# Deployment — alpha-engine-predictor

## Current deployment state (as of 2026-03-12)

| Resource | Name / Value |
|---|---|
| **AWS Account** | `711398986525` |
| **Region** | `us-east-1` |
| **Lambda function** | `alpha-engine-predictor-inference` |
| **Lambda runtime** | Python 3.12 (zip deployment) |
| **Lambda memory** | 1024 MB |
| **Lambda timeout** | 300 s |
| **Lambda handler** | `inference.handler.handler` |
| **Lambda env vars** | `S3_BUCKET=alpha-engine-research`, `LD_LIBRARY_PATH=/var/task/lib` |
| **IAM role** | `alpha-engine-predictor-role` |
| **S3 bucket** | `alpha-engine-research` |
| **GBM weights** | `s3://alpha-engine-research/predictor/weights/gbm_latest.txt` |
| **Predictions** | `s3://alpha-engine-research/predictor/predictions/` |
| **Lambda zip** | `s3://alpha-engine-research/lambda/alpha-engine-predictor.zip` |
| **ECR repo** | `711398986525.dkr.ecr.us-east-1.amazonaws.com/alpha-engine-predictor` (reserved for future container builds) |
| **EventBridge rule** | `ae-predictor-run` |
| **Schedule** | `cron(15 13 ? * MON-FRI *)` → 6:15am PDT / 5:15am PST |

---

## Scheduling

```
5:45am PT  alpha-engine-research Lambda completes → writes signals.json to S3
6:15am PT  ae-predictor-run fires → alpha-engine-predictor-inference Lambda
6:30am PT  ae-executor-start fires → EC2 executor starts (reads predictions from S3)
```

The predictor sits in a 30-minute window between the research pipeline and the
executor, giving the GBM time to score all research tickers before trading begins.

**DST behaviour**: The rule fires at 13:15 UTC regardless of US clock changes.
- PDT (Mar–Nov): 13:15 UTC = 6:15am PT ✓ (nominal target)
- PST (Nov–Mar): 13:15 UTC = 5:15am PT (1 hour early — still before 6:30am executor)

No manual cron updates are needed when clocks change; the 1-hour shift in winter
is harmless because the research pipeline also runs earlier (same UTC anchor).

---

## Lambda zip deployment

The Lambda uses **zip deployment**, not a container image. The original design
called for a container image because PyTorch (~350 MB) exceeds Lambda's 250 MB
unzipped limit. Since we switched to GBM (`model_type=gbm`) for inference, PyTorch
is no longer imported at load time and the full dependency set fits in a zip.

### Build the deployment zip

```bash
# Install Lambda deps (linux/amd64 wheels, no scipy, no torch, no pyarrow)
pip install \
  --platform manylinux_2_28_x86_64 \
  --target /tmp/lambda-pkg \
  --implementation cp \
  --python-version 3.12 \
  --only-binary=:all: \
  -r requirements-lambda.txt

# Remove scipy and boto3/botocore (scipy pulled in transitively but not needed;
# boto3 is provided by Lambda runtime)
rm -rf /tmp/lambda-pkg/scipy* /tmp/lambda-pkg/boto3 \
       /tmp/lambda-pkg/botocore /tmp/lambda-pkg/s3transfer

# Add libgomp.so.1 (see note below)
mkdir -p /tmp/lambda-pkg/lib
cp lib/libgomp.so.1 /tmp/lambda-pkg/lib/

# Add scipy stub (see note below)
cp -r lib/scipy_stub/scipy /tmp/lambda-pkg/scipy

# Copy application source
cp config.py /tmp/lambda-pkg/
mkdir -p /tmp/lambda-pkg/config
cp config/predictor.yaml /tmp/lambda-pkg/config/  # gitignored — must exist locally
cp -r model /tmp/lambda-pkg/
cp -r inference /tmp/lambda-pkg/
cp -r data /tmp/lambda-pkg/
rm -rf /tmp/lambda-pkg/data/cache  # exclude local dev caches

# Zip and upload
cd /tmp/lambda-pkg
zip -r /tmp/alpha-engine-predictor.zip . --exclude "*.pyc" --exclude "__pycache__/*" -q
aws s3 cp /tmp/alpha-engine-predictor.zip \
  s3://alpha-engine-research/lambda/alpha-engine-predictor.zip

# Update Lambda
aws lambda update-function-code \
  --function-name alpha-engine-predictor-inference \
  --s3-bucket alpha-engine-research \
  --s3-key lambda/alpha-engine-predictor.zip \
  --region us-east-1
aws lambda wait function-updated --function-name alpha-engine-predictor-inference
```

**Zip size**: ~46 MB compressed / ~153 MB unzipped (well within the 250 MB limit).

### libgomp.so.1 — why it's bundled

LightGBM's native library (`lib_lightgbm.so`) is dynamically linked against
`libgomp.so.1` (GCC's OpenMP runtime). The Lambda AL2023 runtime does **not**
include this library. The fix is to bundle `libgomp.so.1` in `lib/` and set
`LD_LIBRARY_PATH=/var/task/lib` in the Lambda environment.

The bundled copy was extracted from the CentOS Stream 9 `libgomp-11.5.0` RPM
(compatible with AL2023 / glibc 2.34+, x86_64). To refresh it if needed:

```python
# Extract libgomp.so.1 from CentOS 9 RPM (requires: pip install zstandard)
import urllib.request, zstandard, struct, io

rpm_url = ("https://mirror.stream.centos.org/9-stream/BaseOS/x86_64/os"
           "/Packages/libgomp-11.5.0-14.el9.x86_64.rpm")

with urllib.request.urlopen(rpm_url) as r:
    rpm_data = r.read()

# zstd payload starts at offset 15421 in this RPM
payload = rpm_data[15421:]
cpio_data = zstandard.ZstdDecompressor().decompress(payload, max_output_size=50<<20)

pos = 0
while pos < len(cpio_data) - 110:
    if cpio_data[pos:pos+6] not in (b'070701', b'070702'):
        pos += 1; continue
    namesize = int(cpio_data[pos+94:pos+102], 16)
    filesize  = int(cpio_data[pos+54:pos+62], 16)
    name_start = pos + 110
    name = cpio_data[name_start:name_start+namesize].rstrip(b'\x00').decode()
    data_start = (name_start + namesize + 3) & ~3
    if 'libgomp.so.1.0.0' in name and filesize > 1000:
        with open('lib/libgomp.so.1', 'wb') as f:
            f.write(cpio_data[data_start:data_start+filesize])
        print(f"Extracted {filesize:,} bytes")
    data_end = (data_start + filesize + 3) & ~3
    if name == 'TRAILER!!!': break
    pos = data_end
```

### scipy stub — why it exists

LightGBM's `basic.py` imports `scipy.sparse` at **module level** (not
conditionally). The real scipy package is ~138 MB, which would push the zip
over the 250 MB limit. Since we never pass sparse matrices (all inference
inputs are dense numpy arrays), a minimal stub satisfies the import:

```
scipy/
  __init__.py          # empty stub
  sparse/
    __init__.py        # spmatrix, csr_matrix, csc_matrix, issparse() stubs
```

The stubs raise `ImportError` if sparse construction is actually called at
runtime, so any accidental use fails loudly rather than silently.

---

## First-time setup

```bash
# 1. Create ECR repo (exists; skip if already created)
aws ecr create-repository --repository-name alpha-engine-predictor --region us-east-1

# 2. Create IAM role
aws iam create-role --role-name alpha-engine-predictor-role \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

# Attach inline policy (S3 + CloudWatch Logs)
aws iam put-role-policy --role-name alpha-engine-predictor-role \
  --policy-name alpha-engine-predictor-policy \
  --policy-document file://infrastructure/lambda-iam-policy.json

# 3. Create Lambda function (first time only)
aws lambda create-function \
  --function-name alpha-engine-predictor-inference \
  --runtime python3.12 \
  --role arn:aws:iam::711398986525:role/alpha-engine-predictor-role \
  --handler inference.handler.handler \
  --code S3Bucket=alpha-engine-research,S3Key=lambda/alpha-engine-predictor.zip \
  --timeout 300 \
  --memory-size 1024 \
  --environment 'Variables={S3_BUCKET=alpha-engine-research,LD_LIBRARY_PATH=/var/task/lib}' \
  --region us-east-1

# 4. Create EventBridge schedule
aws events put-rule \
  --name ae-predictor-run \
  --schedule-expression "cron(15 13 ? * MON-FRI *)" \
  --state ENABLED \
  --region us-east-1

aws lambda add-permission \
  --function-name alpha-engine-predictor-inference \
  --statement-id ae-predictor-eventbridge \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:us-east-1:711398986525:rule/ae-predictor-run \
  --region us-east-1

aws events put-targets \
  --rule ae-predictor-run \
  --targets '[{"Id":"predictor-lambda","Arn":"arn:aws:lambda:us-east-1:711398986525:function:alpha-engine-predictor-inference"}]' \
  --region us-east-1
```

---

## Updating the Lambda (redeployment)

After code changes:

```bash
# Rebuild zip (see "Build the deployment zip" above)
# Upload and update:
aws s3 cp /tmp/alpha-engine-predictor.zip \
  s3://alpha-engine-research/lambda/alpha-engine-predictor.zip
aws lambda update-function-code \
  --function-name alpha-engine-predictor-inference \
  --s3-bucket alpha-engine-research \
  --s3-key lambda/alpha-engine-predictor.zip \
  --region us-east-1
aws lambda wait function-updated --function-name alpha-engine-predictor-inference
echo "Deploy complete"
```

After GBM retraining (weights-only update — no code change needed):

```bash
aws s3 cp checkpoints/gbm_best.txt \
  s3://alpha-engine-research/predictor/weights/gbm_latest.txt
# Lambda picks up new weights on next invocation (no redeploy needed)
```

---

## Testing the deployed Lambda

```bash
# Dry run (no S3 writes)
aws lambda invoke \
  --function-name alpha-engine-predictor-inference \
  --payload '{"dry_run": true}' \
  --cli-binary-format raw-in-base64-out \
  --log-type Tail \
  --region us-east-1 \
  /tmp/response.json && cat /tmp/response.json

# Live run for a specific date
aws lambda invoke \
  --function-name alpha-engine-predictor-inference \
  --payload '{"date": "2026-03-12"}' \
  --cli-binary-format raw-in-base64-out \
  --region us-east-1 \
  /tmp/response.json && cat /tmp/response.json
```

Expected successful response:
```json
{"statusCode": 200, "body": "Predictions written for 2026-03-12"}
```

---

## CloudWatch logs

```bash
# Tail the last Lambda invocation log
aws logs tail /aws/lambda/alpha-engine-predictor-inference \
  --follow --format short --region us-east-1
```

---

## GBM retraining

The GBM is retrained on the EC2 backtester instance (`alpha-engine-backtester`)
via the Sunday backtester cron. To retrain manually:

```bash
# On the backtester EC2 or locally
python train_gbm.py --data-dir data/cache

# Review output
cat checkpoints/gbm_eval_report.json

# Promote to production if IC > 0.05
aws s3 cp checkpoints/gbm_best.txt \
  s3://alpha-engine-research/predictor/weights/gbm_latest.txt
```

**Retraining trigger conditions:**
- Sunday weekly backtester run (automatic)
- Rolling 30-day IC drops below 0.02 for 5+ consecutive days
- New features added to `config.FEATURES` or sector-neutral label logic changes
- After significant market regime change (discretionary)

**Retraining IC gate**: New model must achieve test IC ≥ 0.04 before upload to S3.
Current production baseline: IC ≈ 0.046 (sector-neutral labels, 21 features, 5-day horizon).

---

## IAM policy reference

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::alpha-engine-research",
        "arn:aws:s3:::alpha-engine-research/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:711398986525:log-group:/aws/lambda/alpha-engine-predictor-inference:*"
    }
  ]
}
```

---

## Container image path (future / MLP reactivation)

The ECR repo (`alpha-engine-predictor`) is provisioned and ready. If the MLP
model is reactivated (requiring PyTorch), switch back to a container image:

```bash
# Requires Docker Desktop installed
export AWS_ACCOUNT_ID=711398986525
export AWS_REGION=us-east-1
./infrastructure/deploy.sh
```

The Dockerfile installs PyTorch CPU-only (~350 MB) and is otherwise identical to
the zip-based deployment. The Lambda function would need to be updated to
`--package-type Image` pointing to the ECR image URI.
