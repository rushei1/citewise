# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
# CPU-only torch keeps the image ~2 GB smaller than the default CUDA wheels.
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/app/.cache/huggingface \
    DB_PATH=/app/data/reviews.db
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app app/ app/
USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
