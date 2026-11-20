FROM python:3.11-slim

WORKDIR /app

# Install minimal runtime dependencies (curl for health check; avoid bulky build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Health check validates both Streamlit web server responsiveness and recommender model artifact readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health && \
        python -c "from recommender import MovieRecommender; r = MovieRecommender(); assert r.load_artifacts() and r.tfidf_matrix is not None" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
