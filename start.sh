#!/bin/bash

# Word Bocce - Quick Start Script
# This script helps you start both the backend and frontend servers

set -e

echo "🎯 Word Bocce - Starting servers..."
echo ""

# Check if embeddings exist
if [ -z "$MODEL_PATH" ]; then
    if [ -f "./embeddings/glove-100.bin" ]; then
        export MODEL_PATH="./embeddings/glove-100.bin"
        echo "✓ Found embeddings at: $MODEL_PATH"
    elif [ -f "./embeddings/glove-300.bin" ]; then
        export MODEL_PATH="./embeddings/glove-300.bin"
        echo "✓ Found embeddings at: $MODEL_PATH"
    elif [ -f "./embeddings/google-news.bin" ]; then
        export MODEL_PATH="./embeddings/google-news.bin"
        echo "✓ Found embeddings at: $MODEL_PATH"
    else
        echo "❌ No embeddings found!"
        echo ""
        echo "Please download embeddings first:"
        echo "  python setup_embeddings.py --model glove-100"
        echo ""
        exit 1
    fi
else
    echo "✓ Using MODEL_PATH: $MODEL_PATH"
fi

echo ""
echo "Starting servers..."
echo "  - Backend: http://localhost:8000"
echo "  - Frontend: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start backend in background
uvicorn word_bocce_mvp_fastapi:app --reload --port 8000 &
BACKEND_PID=$!

# Give backend time to start
sleep 2

# Start frontend in background
python run_frontend.py --port 8080 &
FRONTEND_PID=$!

# Wait a bit then open browser
sleep 1
echo ""
echo "✓ Servers running!"
echo ""
echo "Open http://localhost:8080 in your browser to play!"
echo ""

# Trap Ctrl+C to kill both processes
trap "echo ''; echo 'Shutting down servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT

# Wait for processes
wait
