FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY word_bocce_mvp_fastapi.py .
COPY index.html .
COPY presentation.html .
COPY setup_embeddings.py .

# Download embeddings at build time (smaller model for Docker image)
RUN python setup_embeddings.py --model glove-100 --output ./embeddings

ENV MODEL_PATH=./embeddings/glove-100.bin
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "word_bocce_mvp_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
