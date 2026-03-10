# Deployment — alpha-engine-predictor

## AWS prerequisites

Before deploying, ensure the following AWS resources exist:

### IAM role (Lambda execution role)

The Lambda function needs an IAM role with these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:iam::ACCOUNT_ID:role/alpha-engine-predictor-lambda",
        "arn:aws:s3:::alpha-engine-research/predictor/*",
        "arn:aws:s3:::alpha-engine-research/signals/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

### S3 bucket

Uses the same bucket as alpha-engine-research: `alpha-engine-research`.

Predictor writes to the `predictor/` prefix:
- `predictor/weights/` — model checkpoints
- `predictor/predictions/` — daily prediction JSONs
- `predictor/metrics/` — model health summaries
- `predictor/price_cache/` — bootstrapped OHLCV parquets (optional)

### ECR repository

Create the ECR repo before first deploy:

```bash
aws ecr create-repository \
  --repository-name alpha-engine-predictor \
  --region us-east-1
```

### Lambda function

Create the Lambda function pointing to the ECR image:

```bash
# First: build and push image (deploy.sh handles this)
./infrastructure/deploy.sh

# Create function (first time only)
aws lambda create-function \
  --function-name alpha-engine-predictor-inference \
  --package-type Image \
  --code ImageUri=ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/alpha-engine-predictor:latest \
  --role arn:aws:iam::ACCOUNT_ID:role/alpha-engine-predictor-lambda \
  --memory-size 1024 \
  --timeout 300 \
  --environment Variables={S3_BUCKET=alpha-engine-research} \
  --region us-east-1
```

---

## First-time setup walkthrough

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/alpha-engine-predictor.git
cd alpha-engine-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Bootstrap training data (one-time, ~30 min for 900 tickers)
python data/bootstrap_fetcher.py --output-dir data/cache

# 4. Train the model
python train.py --data-dir data/cache --device cpu --output checkpoints/

# 5. Review evaluation metrics in checkpoints/eval_report.json
cat checkpoints/eval_report.json

# 6. If metrics pass gates, upload weights to S3
aws s3 cp checkpoints/best.pt s3://alpha-engine-research/predictor/weights/latest.pt
aws s3 cp checkpoints/best.pt s3://alpha-engine-research/predictor/weights/$(date +%Y%m%d).pt

# 7. Set environment variables for deploy
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=us-east-1

# 8. Build and deploy Lambda
./infrastructure/deploy.sh

# 9. Create EventBridge rule (see below)
```

---

## Container image build and deploy

The predictor uses a container image Lambda because PyTorch CPU-only (~350MB)
exceeds the Lambda zip layer limit (250MB unzipped).

```bash
# Build only (no AWS push)
./infrastructure/deploy.sh --dry-run

# Full build + push + Lambda update
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=us-east-1
./infrastructure/deploy.sh
```

The script:
1. Builds a linux/amd64 Docker image from the Dockerfile
2. Authenticates to ECR using `aws ecr get-login-password`
3. Tags and pushes the image to ECR
4. Updates the Lambda function code to point to the new image digest
5. Waits for the update to complete

### Image size optimization

PyTorch CPU-only is installed from `https://download.pytorch.org/whl/cpu`,
which excludes CUDA libraries. This reduces the image from ~4GB (CUDA) to
~1.2GB (CPU-only), well within Lambda's 10GB container limit.

If image size is a concern, consider:
- `torch+cpu` slim wheels (no CUDA)
- Multi-stage builds to exclude build tools
- Lambda SnapStart for faster cold starts (Java only as of 2026)

---

## EventBridge scheduling

Create a rule to trigger the predictor 30 minutes after the research pipeline:

```bash
# Research runs at 5:45am PT → predictor runs at 6:15am PT
# Pacific Time = UTC-8 (PST) or UTC-7 (PDT)
# PST: 6:15am PT = 14:15 UTC
# PDT: 6:15am PT = 13:15 UTC

# Create rule (use 14:15 UTC for PST, swap to 13:15 UTC for PDT in summer)
aws events put-rule \
  --name alpha-engine-predictor-daily \
  --schedule-expression "cron(15 14 ? * MON-FRI *)" \
  --state ENABLED \
  --region us-east-1

# Add Lambda as target
aws events put-targets \
  --rule alpha-engine-predictor-daily \
  --targets "Id=predictor-lambda,Arn=arn:aws:lambda:us-east-1:ACCOUNT_ID:function:alpha-engine-predictor-inference" \
  --region us-east-1

# Grant EventBridge permission to invoke Lambda
aws lambda add-permission \
  --function-name alpha-engine-predictor-inference \
  --statement-id EventBridgeInvoke \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:us-east-1:ACCOUNT_ID:rule/alpha-engine-predictor-daily \
  --region us-east-1
```

**DST handling**: Update the cron expression manually when clocks change
(second Sunday of March → switch to 13:15 UTC; first Sunday of November →
switch back to 14:15 UTC). Alternatively, set the schedule to both times with
a day-of-week condition, but this is operationally complex for minimal benefit.

---

## Retraining on SageMaker vs. EC2 spot

### SageMaker Training Jobs (recommended)

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=123456789012
export SAGEMAKER_ROLE_ARN=arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole
./infrastructure/retrain.sh
```

**Pros**: Managed, auditable, auto-scales, cost-tracked per job.
**Cons**: Higher per-hour cost than EC2 spot, requires SageMaker IAM setup.
**Cost**: `ml.g4dn.xlarge` ~$0.70/hr; training completes in <30 min → ~$0.35/run.

### EC2 spot (cheaper, manual)

```bash
# Run locally or on any machine with data/cache populated
./infrastructure/retrain.sh --local
```

**Pros**: Cheaper (~$0.15/hr for g4dn.xlarge spot), full control.
**Cons**: Spot interruption risk, manual setup, no automatic output management.
**Recommendation**: Use spot for development; SageMaker for production retraining
that feeds model weights into the live system.

### Retraining trigger

Retrain when any of:
- Monthly schedule fires (first Sunday of month)
- Rolling 30-day hit rate drops below 0.50 for 5+ consecutive days
- IC drops below 0.02 for two consecutive weeks
- New features added to `config.FEATURES`

After retraining, validate the new checkpoint against the old one before
promoting to `weights/latest.pt`. A new model must not regress on the test
set relative to the current production model.
