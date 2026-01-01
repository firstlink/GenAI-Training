# Session 8: Production Deployment

**Duration**: 90 minutes  
**Difficulty**: Advanced  
**Colab Notebook**: [08_Production.ipynb](../notebooks/08_Production.ipynb)

## Learning Objectives
- 🎯 Build production-ready FastAPI application
- 🎯 Implement caching with Redis
- 🎯 Add monitoring and logging
- 🎯 Set up security and guardrails
- 🎯 Deploy to cloud (Railway/Render)
- 🎯 Optimize performance and costs

## Capstone: SupportGenie v1.0 - Production Ready!

**Final version** with:
- FastAPI REST API
- Redis caching
- PostgreSQL database
- Monitoring dashboard
- Rate limiting
- Security features
- Cloud deployment
- **PRODUCTION READY!** 🚀

## Part 1: Production Architecture

```
┌─────────────┐
│   Gradio    │  (Frontend UI)
│     UI      │
└──────┬──────┘
       │
┌──────▼──────┐
│   FastAPI   │  (Backend API)
│   Server    │
└──────┬──────┘
       │
    ┌──┴───┬────────┬─────────┬──────────┐
    ↓      ↓        ↓         ↓          ↓
┌───────┐ ┌────┐ ┌─────┐ ┌────────┐ ┌────────┐
│ LLM   │ │RAG │ │Redis│ │Postgres│ │Monitor │
│ API   │ │ DB │ │Cache│ │   DB   │ │  Log   │
└───────┘ └────┘ └─────┘ └────────┘ └────────┘
```

## Part 2: FastAPI Application

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI(title="SupportGenie API", version="1.0")

class QueryRequest(BaseModel):
    message: str
    customer_id: str = None
    
class QueryResponse(BaseModel):
    response: str
    sources: list = []
    confidence: float = 1.0

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """Main chat endpoint"""
    try:
        # Initialize agent
        agent = SupportGenieV6()
        
        # Process query
        response = agent.handle_query(
            query=request.message,
            customer_id=request.customer_id
        )
        
        return QueryResponse(
            response=response['answer'],
            sources=response.get('sources', []),
            confidence=response.get('confidence', 1.0)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Part 3: Caching with Redis

```python
import redis
import json
import hashlib

class ResponseCache:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hour
    
    def get_cache_key(self, query, customer_id=None):
        """Generate cache key"""
        data = f"{query}:{customer_id}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get(self, query, customer_id=None):
        """Get cached response"""
        key = self.get_cache_key(query, customer_id)
        cached = self.redis.get(key)
        
        if cached:
            return json.loads(cached)
        return None
    
    def set(self, query, response, customer_id=None):
        """Cache response"""
        key = self.get_cache_key(query, customer_id)
        self.redis.setex(
            key,
            self.ttl,
            json.dumps(response)
        )

# Usage in API
cache = ResponseCache()

@app.post("/chat")
async def chat(request: QueryRequest):
    # Check cache first
    cached = cache.get(request.message, request.customer_id)
    if cached:
        return QueryResponse(**cached)
    
    # Process query
    response = agent.handle_query(request.message, request.customer_id)
    
    # Cache result
    cache.set(request.message, response, request.customer_id)
    
    return QueryResponse(**response)
```

## Part 4: Logging and Monitoring

```python
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class MetricsLogger:
    def __init__(self):
        self.metrics = []
    
    def log_request(self, query, response, latency, cost, customer_id=None):
        """Log API request metrics"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "customer_id": customer_id,
            "query_length": len(query),
            "response_length": len(response),
            "latency_ms": latency,
            "cost_usd": cost,
            "success": True
        }
        
        self.metrics.append(metric)
        logger.info(f"Request processed: {json.dumps(metric)}")

metrics = MetricsLogger()

@app.post("/chat")
async def chat(request: QueryRequest):
    import time
    start_time = time.time()
    
    try:
        response = agent.handle_query(request.message)
        latency = (time.time() - start_time) * 1000
        
        metrics.log_request(
            query=request.message,
            response=response['answer'],
            latency=latency,
            cost=response.get('cost', 0),
            customer_id=request.customer_id
        )
        
        return QueryResponse(**response)
    
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

## Part 5: Rate Limiting

```python
from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/chat")
@limiter.limit("10/minute")  # 10 requests per minute
async def chat(request: Request, query: QueryRequest):
    # Process normally
    pass
```

## Part 6: Security

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    token = credentials.credentials
    
    # Verify against database or environment
    if token != os.getenv("API_SECRET_KEY"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return token

@app.post("/chat", dependencies=[Depends(verify_token)])
async def chat(request: QueryRequest):
    # Only accessible with valid token
    pass

# Input validation
def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input"""
    text = text.strip()
    
    if len(text) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"Input too long (max {max_length} characters)"
        )
    
    # Check for malicious patterns
    dangerous_patterns = ['<script>', 'javascript:', '<?php']
    for pattern in dangerous_patterns:
        if pattern in text.lower():
            raise HTTPException(
                status_code=400,
                detail="Input contains prohibited content"
            )
    
    return text
```

## Part 7: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=supportgenie
      - POSTGRES_PASSWORD=secret
    ports:
      - "5432:5432"
```

## Part 8: Cloud Deployment

### Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up

# Set environment variables
railway variables set OPENAI_API_KEY=sk-...
```

### Deploy to Render

```yaml
# render.yaml
services:
  - type: web
    name: supportgenie
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: OPENAI_API_KEY
        sync: false
```

## Part 9: Monitoring Dashboard

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Simple monitoring dashboard"""
    
    # Get metrics
    total_requests = len(metrics.metrics)
    avg_latency = sum(m['latency_ms'] for m in metrics.metrics) / total_requests
    total_cost = sum(m['cost_usd'] for m in metrics.metrics)
    
    html = f"""
    <html>
        <head><title>SupportGenie Dashboard</title></head>
        <body>
            <h1>SupportGenie Metrics</h1>
            <div>
                <p>Total Requests: {total_requests}</p>
                <p>Avg Latency: {avg_latency:.2f}ms</p>
                <p>Total Cost: ${total_cost:.4f}</p>
            </div>
        </body>
    </html>
    """
    return html
```

## Part 10: Cost Optimization

```python
class CostOptimizer:
    def __init__(self):
        self.cache = ResponseCache()
        self.model_selector = ModelSelector()
    
    def process_query(self, query, complexity="auto"):
        """Select optimal model based on complexity"""
        
        # Check cache first (free!)
        cached = self.cache.get(query)
        if cached:
            return cached
        
        # Select model based on complexity
        if complexity == "auto":
            complexity = self.assess_complexity(query)
        
        if complexity == "simple":
            model = "gpt-3.5-turbo"  # $0.50/$1.50 per 1M tokens
        elif complexity == "moderate":
            model = "gpt-4-turbo"     # $10/$30 per 1M tokens
        else:
            model = "gpt-4"           # $30/$60 per 1M tokens
        
        # Process with selected model
        response = self.agent.process(query, model=model)
        
        # Cache result
        self.cache.set(query, response)
        
        return response
```

## Exercises
1. Deploy FastAPI app locally
2. Add Redis caching
3. Implement rate limiting
4. Deploy to Railway or Render
5. Create monitoring dashboard

## Key Takeaways
✅ FastAPI for production APIs  
✅ Redis for caching  
✅ Proper logging and monitoring  
✅ Security with authentication  
✅ Docker for containerization  
✅ Cloud deployment options

## 🎓 COURSE COMPLETE!

**Congratulations!** You've built a complete production-ready AI application!

### What You Learned:
1. ✅ LLM fundamentals and API usage
2. ✅ Advanced prompt engineering
3. ✅ RAG systems from scratch
4. ✅ Function calling and tool use
5. ✅ AI agents with memory
6. ✅ Multi-agent orchestration
7. ✅ Evaluation and testing
8. ✅ Production deployment

### Your Capstone Project:
**SupportGenie** - Complete AI customer support platform
- Autonomous agent with tools
- RAG-powered knowledge base
- Multi-agent architecture
- Production deployment
- Monitoring and optimization

### Next Steps:
1. Complete the capstone project
2. Add your own features
3. Deploy to production
4. Build your portfolio
5. Apply to real projects!

**You're now ready to build production Gen AI applications!** 🚀
