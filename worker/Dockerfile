# Build context: repository root (needs ../shared)
FROM python:3.11-slim
WORKDIR /app
COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# PyTorch only for dqn/ppo:  docker compose build --build-arg WITH_TORCH=1 worker
ARG WITH_TORCH=0
RUN if [ "$WITH_TORCH" = "1" ]; then pip install --no-cache-dir torch; fi
COPY shared/gol_shared gol_shared
COPY worker/gol_worker gol_worker
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
CMD ["python", "-m", "gol_worker.run"]
