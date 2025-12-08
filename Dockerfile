FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Copy model artifacts (best_model.pkl + metrics.json)
COPY artifacts/ artifacts/

EXPOSE 5000

CMD ["python", "app.py"]
