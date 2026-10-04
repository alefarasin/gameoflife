# Build context: repository root (needs ../shared)
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/gol_shared gol_shared
COPY backend/gol_backend gol_backend
ENV PYTHONPATH=/app GOL_DB=/data/gol.db
VOLUME /data
EXPOSE 8000
CMD ["python", "-m", "gol_backend.main"]
