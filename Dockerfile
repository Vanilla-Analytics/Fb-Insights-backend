# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

ENV PORT=8000
EXPOSE 8000

# Launch application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
