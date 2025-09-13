FROM python:3.11-slim

# System deps (optional but useful for certs/openssl time zones etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Env for uvicorn
ENV PORT=8080

# Expose port (useful locally; App Runner discovers automatically)
EXPOSE 8080

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
