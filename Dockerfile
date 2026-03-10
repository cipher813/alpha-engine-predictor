# Dockerfile — Lambda container image for alpha-engine-predictor.
#
# PyTorch CPU-only wheel is ~350MB packaged, which exceeds the Lambda zip
# layer limit. Container images support up to 10GB, giving ample headroom.
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

# Install system dependencies (required by some numpy/pandas builds)
RUN dnf install -y gcc g++ && dnf clean all

# Copy and install Python requirements first for better layer caching.
# torch is installed separately with the CPU-only index URL to avoid
# pulling in CUDA libraries (~3x larger than CPU-only build).
COPY requirements.txt .

RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    yfinance>=0.2.30 \
    boto3>=1.26.0 \
    scipy>=1.10.0 \
    tqdm>=4.65.0 \
    "pyarrow>=12.0.0" \
    requests>=2.28.0

# Install PyTorch CPU-only (keeps image size manageable)
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY config.py .
COPY data/ data/
COPY model/ model/
COPY inference/ inference/

# Lambda handler entry point
CMD ["inference.handler.handler"]
