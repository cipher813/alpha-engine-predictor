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

# Install libgomp (OpenMP runtime required by LightGBM) and git (required by
# the alpha-engine-lib install below, which uses pip's git+https:// scheme).
RUN dnf install -y libgomp git && dnf clean all

# Bake the source commit SHA into the image so PredictorPreflight can detect
# deploy drift (deployed SHA vs origin/main HEAD). Passed by deploy.sh via
# `--build-arg GIT_SHA=<sha>` (CI uses $GITHUB_SHA; local dev defaults to
# `git rev-parse HEAD`). A file is chosen over an env var so the stamp
# travels with the image artifact itself — you can't have a "deployed image"
# with a different stamp than what was baked.
ARG GIT_SHA=unknown
RUN echo "${GIT_SHA}" > /var/task/GIT_SHA.txt

# Copy and install Python requirements first for better layer caching.
# alpha-engine-lib is installed from public git+https (lib was flipped
# public 2026-05-03; previous versions vendored a local copy).
COPY requirements-lambda.txt .

RUN pip install --no-cache-dir "alpha-engine-lib[arcticdb,flow_doctor] @ git+https://github.com/cipher813/alpha-engine-lib@v0.2.4" && \
    pip install --no-cache-dir -r requirements-lambda.txt && \
    rm -rf /root/.cache/pip

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

# flow-doctor.yaml at LAMBDA_TASK_ROOT is loaded by setup_logging() at
# module-top of inference/handler.py. The path resolves via:
#   os.environ.get("LAMBDA_TASK_ROOT", os.path.dirname(os.path.dirname(...)))
# Mirrors alpha-engine-research / alpha-engine-data Dockerfiles.
# (flow-doctor-training.yaml is NOT shipped here — training runs on EC2
# spot, not Lambda; that yaml is read from the repo root via the local
# checkout that spot_train.sh sets up.)
COPY flow-doctor.yaml ./

# Lambda handler entry point
CMD ["inference.handler.handler"]
