# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app code
COPY app.py .

# copy model artifacts
COPY artifacts/ artifacts/

EXPOSE 5000
CMD ["python", "app.py"]
