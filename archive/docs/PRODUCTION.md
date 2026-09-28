# 🔒 Production Hardening Guide

Transform Word Bocce from MVP to production-ready application.

**Current Status**: MVP - Works well for demos and small groups
**Target**: Production - Secure, scalable, and reliable

---

## Security Hardening

### 1. Rate Limiting

**Problem**: API can be spammed, causing server overload.

**Solution**: Add rate limiting with slowapi:

```bash
pip install slowapi
```

```python
# In word_bocce_mvp_fastapi.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add to endpoints
@app.post("/match/create")
@limiter.limit("5/minute")  # Max 5 matches per minute per IP
async def create_match(request: Request, req: CreateReq):
    # ... existing code

@app.post("/match/{match_id}/round/{round_id}/submit")
@limiter.limit("30/minute")  # Max 30 submissions per minute
async def submit_move(request: Request, match_id: str, round_id: str, req: SubmitReq):
    # ... existing code
```

**Recommended Limits**:
- Match creation: 5/minute per IP
- Move submission: 30/minute per IP
- Puzzle solving: 20/minute per IP
- Visualization: 10/minute per IP

### 2. Input Validation

**Problem**: No validation on user input (player names, match IDs, card tokens).

**Solution**: Add Pydantic validators:

```python
from pydantic import BaseModel, Field, validator
import re

class CreateReq(BaseModel):
    player_name: str = Field(..., min_length=1, max_length=50)

    @validator('player_name')
    def validate_name(cls, v):
        # Remove excessive whitespace
        v = ' '.join(v.split())
        # Block profanity (basic example)
        if any(word in v.lower() for word in ['badword1', 'badword2']):
            raise ValueError('Inappropriate player name')
        # Only allow alphanumeric and basic punctuation
        if not re.match(r'^[a-zA-Z0-9 \-_]+$', v):
            raise ValueError('Player name contains invalid characters')
        return v

class JokerCard(BaseModel):
    token: str = Field(..., min_length=1, max_length=30)

    @validator('token')
    def validate_token(cls, v):
        v = v.lower().strip()
        if not re.match(r'^[a-z]+$', v):
            raise ValueError('Joker token must be alphabetic')
        if len(v) < 2:
            raise ValueError('Joker token too short')
        return v
```

**Add profanity filter**:

```bash
pip install better-profanity
```

```python
from better_profanity import profanity

profanity.load_censor_words()

def is_clean(text: str) -> bool:
    return not profanity.contains_profanity(text)
```

### 3. CORS Configuration

**Current**: Allows all origins (`allow_origins=["*"]`)

**Production**: Restrict to your domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://wordbocce.com",
        "https://www.wordbocce.com",
        "https://wordbocce.up.railway.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 4. Environment Variables

**Current**: Some defaults hardcoded

**Production**: Move all config to environment:

```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    model_path: str
    deck_size: int = 10000
    wildcard_ratio: float = 0.2
    round_timeout_secs: int = 120

    # Production settings
    allowed_origins: list = ["*"]
    rate_limit_enabled: bool = True
    log_level: str = "INFO"
    sentry_dsn: Optional[str] = None

    class Config:
        env_file = ".env"

settings = Settings()
```

### 5. HTTPS Enforcement

**Add to nginx config** (if self-hosting):

```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name wordbocce.com www.wordbocce.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name wordbocce.com www.wordbocce.com;

    ssl_certificate /etc/letsencrypt/live/wordbocce.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/wordbocce.com/privkey.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    location / {
        proxy_pass http://localhost:8000;
        # ... other proxy settings
    }
}
```

**Cloud platforms**: HTTPS is automatic (Railway, Render, etc.)

---

## Persistence & Scalability

### 6. Add Redis for Match Storage

**Problem**: Matches stored in memory, lost on restart.

**Solution**: Use Redis for persistence:

```bash
pip install redis
```

```python
import redis
import json

redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)

# Store match
def save_match(match_id: str, match_data: dict):
    redis_client.setex(
        f"match:{match_id}",
        86400,  # Expire after 24 hours
        json.dumps(match_data, default=str)
    )

# Retrieve match
def get_match(match_id: str) -> dict:
    data = redis_client.get(f"match:{match_id}")
    return json.loads(data) if data else None
```

**Free Redis hosting**:
- Upstash: https://upstash.com (10k commands/day free)
- Redis Cloud: https://redis.com (30MB free)
- Railway Redis plugin (free tier)

### 7. Database for Puzzle Progress

**Add PostgreSQL** for user puzzle completion tracking:

```bash
pip install sqlalchemy psycopg2-binary
```

```python
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class PuzzleSolution(Base):
    __tablename__ = 'puzzle_solutions'

    id = Column(Integer, primary_key=True)
    player_name = Column(String(50), index=True)
    puzzle_id = Column(Integer, index=True)
    similarity = Column(Float)
    stars = Column(Integer)
    submitted_at = Column(DateTime, default=datetime.datetime.utcnow)

engine = create_engine(os.getenv('DATABASE_URL'))
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
```

**Free PostgreSQL hosting**:
- Supabase: https://supabase.com (500MB free)
- Railway Postgres plugin (free tier)
- Render Postgres (90 days free)

### 8. Horizontal Scaling

**Problem**: Single server can't handle many concurrent players.

**Solution 1: Sticky Sessions** (simpler):

```python
# Use consistent hashing for match assignment
import hashlib

def get_server_for_match(match_id: str, num_servers: int) -> int:
    hash_val = int(hashlib.md5(match_id.encode()).hexdigest(), 16)
    return hash_val % num_servers
```

**Solution 2: Centralized State** (better):
- Move all match state to Redis
- Any server can handle any request
- Load balancer distributes traffic

**Load Balancer Example** (nginx):

```nginx
upstream wordbocce_backend {
    least_conn;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://wordbocce_backend;
    }
}
```

---

## Monitoring & Observability

### 9. Structured Logging

**Current**: Minimal print statements

**Production**: Structured JSON logs:

```bash
pip install python-json-logger
```

```python
import logging
from pythonjsonlogger import jsonlogger

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Usage
logger.info("Match created", extra={
    "match_id": match_id,
    "player_count": 1,
    "event": "match_create"
})
```

### 10. Error Tracking with Sentry

**Add Sentry** for error monitoring:

```bash
pip install sentry-sdk
```

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1,  # 10% of requests
)
```

**Free tier**: 5k errors/month at https://sentry.io

### 11. Health Checks

**Add health endpoint**:

```python
@app.get("/health")
def health_check():
    checks = {
        "embeddings": store is not None,
        "puzzles": len(puzzles_data) > 0
    }

    # Optional: Check Redis/DB connectivity
    try:
        redis_client.ping()
        checks["redis"] = True
    except:
        checks["redis"] = False

    all_healthy = all(checks.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": time.time()
    }
```

### 12. Metrics with Prometheus

**Add metrics endpoint**:

```bash
pip install prometheus-fastapi-instrumentator
```

```python
from prometheus_fastapi_instrumentator import Instrumentator

@app.on_event("startup")
async def startup():
    Instrumentator().instrument(app).expose(app)
```

**Metrics exposed at** `/metrics`:
- Request count
- Response time
- Error rate
- Active matches
- Puzzle completion rate

---

## Performance Optimization

### 13. Caching

**Cache nearest neighbor lookups**:

```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_nearest_cached(result_vector_tuple, k=1):
    # Convert tuple back to array
    result_vec = np.array(result_vector_tuple)
    # ... existing nearest neighbor logic
```

**Cache puzzle responses**:

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

@app.on_event("startup")
async def startup():
    FastAPICache.init(RedisBackend(redis_client), prefix="fastapi-cache")

@app.get("/puzzles")
@cache(expire=3600)  # Cache for 1 hour
async def list_puzzles():
    # ... existing code
```

### 14. Async Operations

**Make I/O operations async**:

```python
from fastapi import BackgroundTasks

@app.post("/puzzle/{puzzle_id}/solve")
async def solve_puzzle(
    puzzle_id: int,
    solution: PuzzleSolution,
    background_tasks: BackgroundTasks
):
    # Immediate validation
    result = validate_solution_sync(puzzle_id, solution)

    # Log async in background
    background_tasks.add_task(log_puzzle_solution, puzzle_id, result)

    return result
```

### 15. Vector Optimization

**Use Annoy index** for faster nearest neighbor:

```python
from annoy import AnnoyIndex

# Build index once at startup
def build_annoy_index():
    dim = store.kv.vector_size
    index = AnnoyIndex(dim, 'angular')

    for i, token in enumerate(deck_tokens):
        vec = store._norms[token]
        index.add_item(i, vec)

    index.build(10)  # 10 trees
    return index

annoy_index = build_annoy_index()

# Fast nearest neighbor
def find_nearest_fast(vec, k=1):
    indices = annoy_index.get_nns_by_vector(vec, k)
    return [deck_tokens[i] for i in indices]
```

---

## Testing

### 16. Unit Tests

```bash
pip install pytest pytest-asyncio httpx
```

```python
# test_api.py
import pytest
from fastapi.testclient import TestClient
from word_bocce_mvp_fastapi import app

client = TestClient(app)

def test_create_match():
    response = client.post("/match/create", json={
        "player_name": "TestPlayer"
    })
    assert response.status_code == 200
    data = response.json()
    assert "match_id" in data
    assert "player_id" in data

def test_invalid_player_name():
    response = client.post("/match/create", json={
        "player_name": ""  # Empty name
    })
    assert response.status_code == 422

def test_puzzle_list():
    response = client.get("/puzzles")
    assert response.status_code == 200
    puzzles = response.json()
    assert len(puzzles) > 0
```

Run tests:
```bash
pytest test_api.py -v
```

### 17. Load Testing

```bash
pip install locust
```

```python
# locustfile.py
from locust import HttpUser, task, between

class WordBocceUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def create_match(self):
        self.client.post("/match/create", json={
            "player_name": f"Player{self.user_id}"
        })

    @task(1)
    def list_puzzles(self):
        self.client.get("/puzzles")
```

Run:
```bash
locust -f locustfile.py --host=http://localhost:8000
```

---

## Deployment Best Practices

### 18. CI/CD Pipeline

**GitHub Actions** (.github/workflows/deploy.yml):

```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Railway
        run: |
          # Railway auto-deploys on push
          echo "Deployment triggered"
```

### 19. Backup Strategy

**Automated backups** for Redis/Postgres:

```bash
# Daily Redis backup
0 2 * * * redis-cli --rdb /backups/redis-$(date +\%Y\%m\%d).rdb

# Daily Postgres backup
0 2 * * * pg_dump $DATABASE_URL > /backups/postgres-$(date +\%Y\%m\%d).sql
```

**Retention**: Keep 7 daily, 4 weekly, 12 monthly backups.

### 20. Disaster Recovery

**Recovery plan**:

1. **Service Down**:
   - Check health endpoint
   - Review logs in Sentry
   - Restart service
   - If persistent, rollback to previous deploy

2. **Data Loss**:
   - Restore from most recent backup
   - Notify affected users
   - Review backup process

3. **Security Breach**:
   - Rotate all secrets
   - Review access logs
   - Patch vulnerability
   - Notify users if needed

---

## Checklist

Before going to production:

### Security
- [ ] Rate limiting enabled
- [ ] Input validation on all endpoints
- [ ] CORS restricted to production domain
- [ ] HTTPS enforced
- [ ] Security headers configured
- [ ] Profanity filter active
- [ ] Environment variables configured

### Persistence
- [ ] Redis for match storage
- [ ] Database for puzzle progress (optional)
- [ ] Backup strategy in place

### Monitoring
- [ ] Structured logging
- [ ] Error tracking (Sentry)
- [ ] Health check endpoint
- [ ] Metrics exposed
- [ ] Alerting configured

### Performance
- [ ] Caching enabled
- [ ] Async operations
- [ ] Vector index optimized
- [ ] Load tested

### Testing
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Load tests completed
- [ ] Security scan done

### Deployment
- [ ] CI/CD pipeline configured
- [ ] Automated backups
- [ ] Disaster recovery plan
- [ ] Rollback procedure tested

---

## Cost Estimates (Production)

### Monthly Costs

**Minimal Production** ($20-30/month):
- Railway Hobby: $5 (app hosting)
- Upstash Redis: Free tier
- Sentry: Free tier
- Cloudflare: Free (CDN/DDoS)

**Standard Production** ($50-80/month):
- Railway Pro: $20 (app hosting)
- Upstash Redis: $10 (pay as you go)
- Supabase: $25 (Postgres)
- Sentry: Free tier
- Cloudflare: Free

**Enterprise** ($200+/month):
- Multiple Railway instances: $60
- Redis Cloud: $40
- RDS Postgres: $50
- Sentry Pro: $26
- Cloudflare Pro: $20
- Datadog: $15

---

## Next Steps

1. **Phase 1**: Security (rate limiting, validation, CORS)
2. **Phase 2**: Monitoring (logging, Sentry, health checks)
3. **Phase 3**: Persistence (Redis, backups)
4. **Phase 4**: Performance (caching, async, optimization)
5. **Phase 5**: Testing (unit, integration, load)
6. **Phase 6**: Production deployment

**Recommended**: Start with Phase 1 and 2 immediately. Add Phase 3 as traffic grows.

---

## Resources

- **FastAPI Security**: https://fastapi.tiangolo.com/tutorial/security/
- **OWASP Top 10**: https://owasp.org/www-project-top-ten/
- **12 Factor App**: https://12factor.net/
- **Production Best Practices**: https://fastapi.tiangolo.com/deployment/
