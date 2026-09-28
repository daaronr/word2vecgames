FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The browser game (self-contained: vectors are in web/data/)
COPY web/ ./web/
# Server + legacy multiplayer UI at /classic
COPY word_bocce_mvp_fastapi.py setup_embeddings.py ./
COPY archive/legacy-ui/index.html ./archive/legacy-ui/index.html

# Full embeddings are only needed by the legacy API (/classic, /match, /puzzle/*/solve).
# Build with --build-arg WITH_EMBEDDINGS=0 for a small image that serves just the game.
ARG WITH_EMBEDDINGS=1
RUN if [ "$WITH_EMBEDDINGS" = "1" ]; then python setup_embeddings.py --model glove-100 --output ./embeddings; fi
ENV MODEL_PATH=./embeddings/glove-100.bin

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
EXPOSE ${PORT}
CMD uvicorn word_bocce_mvp_fastapi:app --host 0.0.0.0 --port ${PORT}
