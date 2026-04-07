# Multi-stage Lambda container image based on python:3.11-slim.
# Smaller than the AWS-provided lambda/python image (~450MB savings).
# Uses the Lambda Runtime Interface Client (awslambdaric) so the slim
# image can speak the Lambda Runtime API.

# ---------------------------------------------------------------------------
# Stage 1: builder — install deps + bake the cross-encoder model
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Build tools needed for cffi/cryptography wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /var/task

# Install Python deps + Lambda Runtime Interface Client into /var/task
COPY requirements.txt .
RUN pip install --no-cache-dir --target /var/task \
        -r requirements.txt \
        awslambdaric

# Bake the cross-encoder model into the image so cold starts skip
# the download step. cache_folder must be a path that survives the
# stage copy below.
RUN python -c "import sys; sys.path.insert(0, '/var/task'); \
    from sentence_transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', \
                 cache_folder='/var/task/models')"

# ---------------------------------------------------------------------------
# Stage 2: final — drop build tools, keep only what runtime needs
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /var/task

# Copy installed deps + baked model from builder stage
COPY --from=builder /var/task /var/task

# Copy application code
COPY query_rag.py server.py lambda_handler.py ./
COPY static/ ./static/

# Tell sentence-transformers where to find the baked model
ENV SENTENCE_TRANSFORMERS_HOME=/var/task/models
# /var/task is read-only at runtime; HuggingFace needs a writable cache dir
ENV HF_HOME=/tmp/huggingface
# Make installed packages importable
ENV PYTHONPATH=/var/task

# Lambda Runtime Interface Client is the entrypoint; CMD is the handler
ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]
CMD ["lambda_handler.handler"]