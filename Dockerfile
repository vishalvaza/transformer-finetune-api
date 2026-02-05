FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml /app/
RUN pip install -U pip && pip install ".[dev]" --no-cache-dir

COPY src /app/src
COPY artifacts /app/artifacts

ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
