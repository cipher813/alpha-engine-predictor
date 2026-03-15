# Dockerfile — Lambda container image for alpha-engine-predictor.
#
# GBM-only inference. PyTorch removed since switching to LightGBM.
# Container image is ~300MB (vs ~1.5GB with PyTorch).
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

# Copy and install Python requirements first for better layer caching.
COPY requirements-lambda.txt .

RUN pip install --no-cache-dir -r requirements-lambda.txt

# Copy application code
COPY config.py .
COPY data/ data/
COPY model/ model/
COPY inference/ inference/

# Lambda handler entry point
CMD ["inference.handler.handler"]
