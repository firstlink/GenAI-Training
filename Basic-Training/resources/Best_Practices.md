# Best Practices for Production Gen AI Applications

## Table of Contents
1. [Prompt Engineering Best Practices](#prompt-engineering-best-practices)
2. [Error Handling](#error-handling)
3. [Security Considerations](#security-considerations)
4. [Cost Optimization](#cost-optimization)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring and Logging](#monitoring-and-logging)

---

## Prompt Engineering Best Practices

### 1. Be Specific and Clear

**❌ Bad:**
```python
prompt = "Summarize this"
```

**✅ Good:**
```python
prompt = """Summarize the following text in 2-3 sentences.
Focus on the main points and key takeaways.

Text: {text}"""
```

### 2. Use System Messages Effectively

```python
messages = [
    {
        "role": "system",
        "content": """You are a helpful customer service assistant.
        - Be polite and professional
        - Keep responses concise (under 100 words)
        - If you don't know something, say so
        - Never make up information"""
    },
    {
        "role": "user",
        "content": user_query
    }
]
```

### 3. Provide Examples (Few-Shot Learning)

```python
prompt = """Extract the company name from the text.

Examples:
Text: "Apple Inc. released new iPhone"
Company: Apple Inc.

Text: "Microsoft announces Azure updates"
Company: Microsoft

Text: "{new_text}"
Company:"""
```

### 4. Use Delimiters

```python
prompt = f"""Analyze the following text delimited by triple backticks.

Text to analyze:
```
{user_text}
```

Provide:
1. Main topic
2. Sentiment
3. Key entities"""
```

---

## Error Handling

### 1. Handle API Failures Gracefully

```python
from openai import OpenAI
import time

def call_llm_with_retry(messages, max_retries=3):
    """Call LLM with exponential backoff retry"""
    client = OpenAI()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                timeout=30.0
            )
            return response

        except openai.APIConnectionError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            print(f"Connection error. Retrying in {wait_time}s...")
            time.sleep(wait_time)

        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** (attempt + 1)
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)

        except openai.APIError as e:
            print(f"API Error: {e}")
            raise
```

### 2. Validate Outputs

```python
def validate_json_response(response_text):
    """Validate and parse JSON from LLM"""
    try:
        import json
        data = json.loads(response_text)

        # Validate required fields
        required_fields = ['name', 'email', 'age']
        for field in required_fields:
            if field not in data:
                return None, f"Missing required field: {field}"

        return data, None

    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
```

### 3. Set Timeouts

```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    timeout=30.0  # 30 second timeout
)
```

---

## Security Considerations

### 1. Never Expose API Keys

**❌ Bad:**
```python
# NEVER do this!
api_key = "sk-1234567890abcdef"  # Hardcoded key
```

**✅ Good:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")
```

### 2. Sanitize User Input

```python
def sanitize_input(user_input, max_length=1000):
    """Sanitize and validate user input"""

    # Remove leading/trailing whitespace
    cleaned = user_input.strip()

    # Limit length
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]

    # Remove potential injection attempts
    dangerous_patterns = ['<script>', 'javascript:', 'onerror=']
    for pattern in dangerous_patterns:
        if pattern.lower() in cleaned.lower():
            raise ValueError(f"Potentially dangerous input detected: {pattern}")

    return cleaned
```

### 3. Implement Content Filtering

```python
def check_content_safety(text):
    """Check content for safety concerns"""
    from openai import OpenAI
    client = OpenAI()

    # Use OpenAI's moderation endpoint
    response = client.moderations.create(input=text)

    if response.results[0].flagged:
        categories = response.results[0].categories
        flagged = [cat for cat, flagged in categories if flagged]
        return False, f"Content flagged for: {', '.join(flagged)}"

    return True, None
```

### 4. Use Environment-Specific Configs

```python
import os

class Config:
    ENV = os.getenv('ENVIRONMENT', 'development')

    if ENV == 'production':
        DEBUG = False
        MAX_TOKENS = 500
        TIMEOUT = 30
    else:
        DEBUG = True
        MAX_TOKENS = 1000
        TIMEOUT = 60
```

---

## Cost Optimization

### 1. Use Appropriate Models

```python
def choose_model(task_complexity):
    """Choose model based on task complexity"""

    if task_complexity == 'simple':
        # Use cheaper model for simple tasks
        return "gpt-3.5-turbo"  # $0.50/$1.50 per 1M tokens
    elif task_complexity == 'moderate':
        return "gpt-4-turbo"  # $10/$30 per 1M tokens
    else:  # complex
        return "gpt-4"  # $30/$60 per 1M tokens
```

### 2. Limit max_tokens

```python
# Prevent unexpectedly long responses
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=150,  # Limit response length
    temperature=0.3
)
```

### 3. Cache Responses

```python
import hashlib
import json
from functools import lru_cache

class ResponseCache:
    def __init__(self):
        self.cache = {}

    def get_cache_key(self, messages):
        """Generate cache key from messages"""
        message_str = json.dumps(messages, sort_keys=True)
        return hashlib.md5(message_str.encode()).hexdigest()

    def get(self, messages):
        """Get cached response"""
        key = self.get_cache_key(messages)
        return self.cache.get(key)

    def set(self, messages, response):
        """Cache response"""
        key = self.get_cache_key(messages)
        self.cache[key] = response

cache = ResponseCache()

def call_llm_with_cache(messages):
    # Check cache first
    cached = cache.get(messages)
    if cached:
        print("Using cached response")
        return cached

    # Call API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Cache result
    cache.set(messages, response)
    return response
```

### 4. Batch Processing

```python
def process_batch(texts, batch_size=10):
    """Process texts in batches to reduce API calls"""
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Combine into single prompt
        combined_prompt = "Process each text:\n\n"
        for idx, text in enumerate(batch):
            combined_prompt += f"{idx+1}. {text}\n"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": combined_prompt}]
        )

        results.append(response)

    return results
```

### 5. Monitor Costs

```python
import tiktoken

def estimate_cost(text, model="gpt-3.5-turbo"):
    """Estimate API call cost"""

    # Get tokenizer for model
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(text))

    # Pricing per 1M tokens (as of Dec 2024)
    pricing = {
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00}
    }

    input_cost = (num_tokens / 1_000_000) * pricing[model]["input"]

    # Estimate output tokens (assume 2x input for safety)
    output_cost = (num_tokens * 2 / 1_000_000) * pricing[model]["output"]

    total_cost = input_cost + output_cost

    return {
        "tokens": num_tokens,
        "estimated_cost": f"${total_cost:.6f}"
    }

# Example usage
text = "Your long prompt here..."
cost_info = estimate_cost(text)
print(f"Estimated tokens: {cost_info['tokens']}")
print(f"Estimated cost: {cost_info['estimated_cost']}")
```

---

## Performance Optimization

### 1. Use Streaming for Long Responses

```python
def stream_response(messages):
    """Stream response for better UX"""

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    return full_response
```

### 2. Parallel Processing

```python
import concurrent.futures

def process_documents_parallel(documents, max_workers=5):
    """Process multiple documents in parallel"""

    def process_single(doc):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Summarize: {doc}"}]
        )
        return response.choices[0].message.content

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single, documents))

    return results
```

### 3. Optimize Embeddings

```python
from sentence_transformers import SentenceTransformer

# Load model once at startup
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings_batch(texts, batch_size=32):
    """Generate embeddings efficiently in batches"""

    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # For faster similarity search
    )

    return embeddings
```

---

## Monitoring and Logging

### 1. Structured Logging

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger('genai_app')
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler('app.log')
        self.logger.addHandler(handler)

    def log_api_call(self, model, prompt_tokens, completion_tokens, latency):
        """Log API call metrics"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "api_call",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency
        }
        self.logger.info(json.dumps(log_data))

    def log_error(self, error_type, error_message, context=None):
        """Log errors with context"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "error",
            "error_type": error_type,
            "error_message": error_message,
            "context": context
        }
        self.logger.error(json.dumps(log_data))

logger = StructuredLogger()
```

### 2. Track Key Metrics

```python
import time

class MetricsTracker:
    def __init__(self):
        self.metrics = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_latency": 0.0,
            "errors": 0
        }

    def track_call(self, tokens, cost, latency):
        """Track API call metrics"""
        self.metrics["total_calls"] += 1
        self.metrics["total_tokens"] += tokens
        self.metrics["total_cost"] += cost

        # Update average latency
        n = self.metrics["total_calls"]
        current_avg = self.metrics["avg_latency"]
        self.metrics["avg_latency"] = ((current_avg * (n-1)) + latency) / n

    def get_metrics(self):
        """Get current metrics"""
        return self.metrics

metrics = MetricsTracker()

# Usage in API calls
start_time = time.time()
response = client.chat.completions.create(...)
latency = (time.time() - start_time) * 1000  # ms

metrics.track_call(
    tokens=response.usage.total_tokens,
    cost=calculate_cost(response),
    latency=latency
)
```

---

## RAG-Specific Best Practices

### 1. Chunk Size Optimization

```python
# Test different chunk sizes
chunk_sizes = [200, 400, 800]

for size in chunk_sizes:
    chunks = chunk_documents(documents, chunk_size=size)
    # Evaluate retrieval quality
    evaluate_retrieval(chunks)
```

### 2. Hybrid Search

```python
def hybrid_search(query, alpha=0.5):
    """Combine semantic and keyword search"""

    # Semantic search
    semantic_results = vector_db.search(query, top_k=10)

    # Keyword search (BM25)
    keyword_results = bm25_search(query, top_k=10)

    # Combine with weighted score
    combined = combine_results(semantic_results, keyword_results, alpha)

    return combined[:5]  # Return top 5
```

### 3. Reranking

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query, results):
    """Rerank results using cross-encoder"""

    pairs = [[query, doc] for doc in results]
    scores = cross_encoder.predict(pairs)

    # Sort by score
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked]
```

---

## Summary

✅ **Prompts**: Be specific, use examples, add structure
✅ **Errors**: Retry with backoff, validate outputs, set timeouts
✅ **Security**: Protect keys, sanitize input, filter content
✅ **Costs**: Choose right models, cache, batch, monitor
✅ **Performance**: Stream responses, parallel process, optimize embeddings
✅ **Monitoring**: Log structured data, track metrics

Following these best practices will help you build robust, secure, and cost-effective Gen AI applications!
