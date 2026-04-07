"""
lambda_handler.py
-----------------
AWS Lambda entry point. Wraps the FastAPI app with Mangum so API Gateway /
Function URL events are translated into ASGI calls.

The cross-encoder is warmed at module import time (not per-invocation) so the
cost is paid once on cold start, not on every request.
"""

from mangum import Mangum

from server import app
from query_rag import get_cross_encoder

# Warm the reranker once at cold start
get_cross_encoder()

# `lifespan="off"` because Mangum doesn't fully run FastAPI's lifespan events
# in Lambda; we warm the model manually above instead.
handler = Mangum(app, lifespan="off")