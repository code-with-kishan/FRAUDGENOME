FROM python:3.11-slim

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN chown -R appuser:appuser /app
USER appuser

ENV MULEGUARD_MODEL_DIR=/app/models

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
