# Dockerfile — Lambda container image for alpha-engine-predictor.
#
# LightGBM + CatBoost inference with Platt calibration.
# Training runs on EC2 spot (spot_train.sh); Lambda is inference-only.
# Container image is ~500MB (LightGBM + CatBoost + scikit-learn).
#
# Build:
#   docker build --platform linux/amd64 -t alpha-engine-predictor .
#
# Run locally (simulates Lambda):
#   docker run -p 9000:8080 alpha-engine-predictor
#   curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
#        -d '{"dry_run": true}'
#
# The CMD points to inference.handler.handler, matching the Lambda handler
# configuration in infrastructure/deploy.sh.

FROM public.ecr.aws/lambda/python:3.12

# Install libgomp (OpenMP runtime required by LightGBM).
RUN dnf install -y libgomp && dnf clean all

# Copy and install Python requirements first for better layer caching.
COPY requirements-lambda.txt .

RUN pip install --no-cache-dir -r requirements-lambda.txt

# Copy application code
COPY retry.py .
COPY health_status.py .
COPY ssm_secrets.py .
COPY config.py .
COPY config/ config/
COPY data/ data/
COPY model/ model/
COPY inference/ inference/
COPY training/ training/
COPY store/ store/

# Lambda handler entry point
CMD ["inference.handler.handler"]
