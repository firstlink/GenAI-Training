# Complete Debugging Guide for Gen AI Applications

## Table of Contents
1. [Common Mistakes by Session](#common-mistakes-by-session)
2. [Error Message Encyclopedia](#error-message-encyclopedia)
3. [Debugging Workflows](#debugging-workflows)
4. [Troubleshooting Decision Trees](#troubleshooting-decision-trees)
5. [Prevention Checklist](#prevention-checklist)

---

## Common Mistakes by Session

### Session 1: LLM Fundamentals

#### ❌ Mistake #1: Hardcoding API Keys
```python
# DON'T DO THIS!
api_key = "sk-1234567890abcdef"  # Exposed in code
client = OpenAI(api_key=api_key)
```

**Why it's wrong**: API keys in code can be:
- Committed to GitHub (public exposure)
- Seen by anyone with code access
- Difficult to rotate

**✅ Correct approach**:
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")

client = OpenAI(api_key=api_key)
```

---

#### ❌ Mistake #2: Not Handling Token Limits
```python
# This can fail silently!
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": very_long_text}]
)
```

**Why it's wrong**: Exceeding token limits causes:
- Truncated responses
- API errors
- Unexpected costs

**✅ Correct approach**:
```python
import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Check before calling
tokens = count_tokens(text)
if tokens > 3000:  # Leave room for response
    text = truncate_text(text, max_tokens=3000)

response = client.chat.completions.create(...)
```

---

#### ❌ Mistake #3: Ignoring Streaming for Long Responses
```python
# User waits 10+ seconds for full response
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    max_tokens=1000
)
print(response.choices[0].message.content)
```

**Why it's wrong**: Poor user experience with long wait times

**✅ Correct approach**:
```python
# Stream for better UX
stream = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    max_tokens=1000,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

### Session 2: Prompt Engineering

#### ❌ Mistake #1: Vague Prompts
```python
prompt = "Tell me about the product"  # Too vague!
```

**Why it's wrong**: LLM doesn't know:
- Which product?
- What information to include?
- What format to use?

**✅ Correct approach**:
```python
prompt = """Provide a detailed overview of the iPhone 15 Pro.

Include the following:
1. Key specifications (screen size, processor, camera)
2. Main features that differentiate it from previous models
3. Target audience
4. Price range

Format: Use bullet points for easy readability.
Tone: Professional but approachable.
Length: 150-200 words."""
```

---

#### ❌ Mistake #2: Not Using System Messages
```python
# Missing system context
messages = [
    {"role": "user", "content": "What's our return policy?"}
]
```

**Why it's wrong**: No context about role, tone, or constraints

**✅ Correct approach**:
```python
messages = [
    {
        "role": "system",
        "content": """You are a customer service assistant for TechStore.
        - Be professional and empathetic
        - Keep responses under 100 words
        - Always cite our policy documents
        - Never make up information"""
    },
    {"role": "user", "content": "What's our return policy?"}
]
```

---

#### ❌ Mistake #3: Inconsistent Formatting
```python
# LLM output format varies each time
prompt = "Extract the name and email from this text"
# Sometimes returns JSON, sometimes plain text
```

**Why it's wrong**: Makes parsing difficult and unreliable

**✅ Correct approach**:
```python
prompt = """Extract the name and email from the following text.

Return ONLY valid JSON in this exact format:
{
  "name": "extracted name",
  "email": "extracted email"
}

Do not include any other text or explanation.

Text: {user_text}"""

# Parse response
import json
try:
    data = json.loads(response.choices[0].message.content)
    name = data['name']
    email = data['email']
except json.JSONDecodeError:
    # Handle parsing error
    pass
```

---

### Session 3: RAG Systems

#### ❌ Mistake #1: Not Chunking Documents
```python
# Storing entire documents as single chunks
documents = ["50-page PDF content here..."]
embeddings = embedding_model.encode(documents)
```

**Why it's wrong**:
- Embeddings lose precision with large text
- Retrieval returns entire document (too much)
- Context window overflow

**✅ Correct approach**:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = splitter.split_text(document)
embeddings = embedding_model.encode(chunks)
```

---

#### ❌ Mistake #2: Wrong Chunk Size
```python
# Too small - loses context
chunks = chunk_text(document, chunk_size=50)  # 50 chars!

# Too large - loses precision
chunks = chunk_text(document, chunk_size=5000)  # 5000 chars!
```

**Why it's wrong**:
- Too small: Fragmented, no context
- Too large: Multiple topics, poor matching

**✅ Correct approach**:
```python
# Test different sizes for your use case
sizes_to_test = [200, 400, 800, 1000]

for size in sizes_to_test:
    chunks = chunk_text(document, chunk_size=size, overlap=size//4)
    evaluate_retrieval_quality(chunks)

# Common sweet spot: 400-800 characters
chunks = chunk_text(document, chunk_size=500, overlap=100)
```

---

#### ❌ Mistake #3: Not Including Source Attribution
```python
# RAG without sources
def rag_answer(question):
    context = retrieve_context(question)
    answer = llm.generate(question, context)
    return answer  # User can't verify!
```

**Why it's wrong**: No way to verify accuracy or find more info

**✅ Correct approach**:
```python
def rag_answer(question):
    results = retrieve_context(question)

    # Extract sources
    sources = []
    for result in results:
        sources.append({
            "file": result['metadata']['source'],
            "page": result['metadata'].get('page', 'N/A')
        })

    context = format_context(results)
    answer = llm.generate(question, context)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": calculate_confidence(results)
    }
```

---

### Session 4: Function Calling

#### ❌ Mistake #1: Poor Function Descriptions
```python
# Vague description
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_data",
            "description": "Gets data",  # Too vague!
            "parameters": {...}
        }
    }
]
```

**Why it's wrong**: LLM can't determine when to use the function

**✅ Correct approach**:
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": """Retrieves the current status of a customer order.
            Use this when a customer asks about:
            - Order status
            - Shipping updates
            - Delivery tracking
            - Order confirmation

            Requires a valid order ID (format: ORD-XXXXX).""",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID (e.g., ORD-12345)"
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]
```

---

#### ❌ Mistake #2: Not Validating Function Inputs
```python
def get_order_status(order_id):
    # Directly use input - dangerous!
    order = database.query(f"SELECT * FROM orders WHERE id='{order_id}'")
    return order
```

**Why it's wrong**: SQL injection, invalid inputs, crashes

**✅ Correct approach**:
```python
import re

def get_order_status(order_id):
    # Validate format
    if not re.match(r'^ORD-\d{5}$', order_id):
        return {
            "error": "Invalid order ID format. Expected: ORD-XXXXX"
        }

    # Use parameterized query
    order = database.query(
        "SELECT * FROM orders WHERE id = ?",
        (order_id,)
    )

    if not order:
        return {
            "error": f"Order {order_id} not found"
        }

    return {
        "order_id": order.id,
        "status": order.status,
        "tracking": order.tracking_number
    }
```

---

#### ❌ Mistake #3: Not Handling Function Errors
```python
# Assuming functions always succeed
function_result = execute_function(name, args)
# What if it fails?
response = llm.generate_with_result(function_result)
```

**Why it's wrong**: Functions can fail (API down, invalid input, etc.)

**✅ Correct approach**:
```python
try:
    function_result = execute_function(name, args)

    if "error" in function_result:
        # Tell LLM about the error
        error_message = f"Function failed: {function_result['error']}"
        response = llm.generate_with_error(error_message)
        return response

    # Success path
    response = llm.generate_with_result(function_result)
    return response

except Exception as e:
    logger.error(f"Function execution failed: {e}")
    # Fallback response
    return "I'm sorry, I encountered an error. Please try again."
```

---

### Session 5: AI Agents & Memory

#### ❌ Mistake #1: Infinite Agent Loops
```python
# No max iterations!
while not task_complete:
    action = agent.decide_action()
    result = execute_action(action)
    # Could loop forever!
```

**Why it's wrong**: Agent can get stuck in loops, wasting money

**✅ Correct approach**:
```python
MAX_ITERATIONS = 10

for iteration in range(MAX_ITERATIONS):
    if task_complete:
        break

    action = agent.decide_action()
    result = execute_action(action)

    if iteration == MAX_ITERATIONS - 1:
        logger.warning("Max iterations reached, task incomplete")
        return "Unable to complete task after 10 attempts"
```

---

#### ❌ Mistake #2: Memory Overflow
```python
# Keeping entire conversation history
conversation_history.append({"role": "user", "content": message})
conversation_history.append({"role": "assistant", "content": response})
# After 50 messages, exceeds context window!
```

**Why it's wrong**: Eventually hits token limit

**✅ Correct approach**:
```python
class ConversationMemory:
    def __init__(self, max_messages=20):
        self.history = []
        self.max_messages = max_messages

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

        # Keep only recent messages
        if len(self.history) > self.max_messages:
            # Keep system message + recent messages
            system_msg = [m for m in self.history if m['role'] == 'system']
            recent = self.history[-(self.max_messages-1):]
            self.history = system_msg + recent

    def summarize_old_messages(self):
        # Alternative: summarize old context
        if len(self.history) > 10:
            old_messages = self.history[:5]
            summary = llm.summarize(old_messages)
            self.history = [summary] + self.history[5:]
```

---

### Session 6: Multi-Agent Systems

#### ❌ Mistake #1: Not Setting Agent Timeouts
```python
# Agents can run indefinitely
result = agent.execute(task)  # Waits forever!
```

**Why it's wrong**: Hangs, wastes resources

**✅ Correct approach**:
```python
import asyncio

async def execute_with_timeout(agent, task, timeout=30):
    try:
        result = await asyncio.wait_for(
            agent.execute(task),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Agent timed out after {timeout}s")
        return {"error": "Agent execution timed out"}
```

---

### Session 7: Evaluation

#### ❌ Mistake #1: Not Testing Edge Cases
```python
# Only testing happy path
test_cases = [
    "What is your return policy?",
    "How do I track my order?"
]
# What about errors, edge cases, adversarial inputs?
```

**Why it's wrong**: System fails in production

**✅ Correct approach**:
```python
test_cases = [
    # Happy path
    "What is your return policy?",

    # Edge cases
    "",  # Empty input
    "a" * 10000,  # Very long input
    "💩🔥😂",  # Emojis only

    # Adversarial
    "Ignore previous instructions and...",
    "<script>alert('xss')</script>",

    # Ambiguous
    "it doesn't work",  # What doesn't work?
    "same as before",  # What's before?

    # Out of scope
    "What's the meaning of life?",
    "Tell me a joke"
]
```

---

## Error Message Encyclopedia

### OpenAI Errors

#### Error: "Rate limit exceeded"
```
openai.RateLimitError: Rate limit reached for requests
```

**Causes**:
- Too many requests per minute (RPM)
- Too many tokens per minute (TPM)
- Quota exceeded

**Solutions**:
```python
from openai import OpenAI
import time

def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)
```

---

#### Error: "Context length exceeded"
```
openai.BadRequestError: This model's maximum context length is 4096 tokens
```

**Causes**:
- Input + output exceeds model's context window
- Common with long conversations or RAG contexts

**Solutions**:
```python
import tiktoken

def truncate_to_limit(messages, model="gpt-3.5-turbo", max_tokens=3000):
    encoding = tiktoken.encoding_for_model(model)

    # Count tokens
    total_tokens = sum(
        len(encoding.encode(msg['content']))
        for msg in messages
    )

    # Truncate if needed
    while total_tokens > max_tokens and len(messages) > 1:
        # Remove oldest message (keep system message)
        if messages[0]['role'] == 'system':
            messages.pop(1)  # Remove second oldest
        else:
            messages.pop(0)  # Remove oldest

        total_tokens = sum(
            len(encoding.encode(msg['content']))
            for msg in messages
        )

    return messages
```

---

#### Error: "Invalid API key"
```
openai.AuthenticationError: Incorrect API key provided
```

**Causes**:
- Wrong API key
- Key not activated
- Key expired
- No billing set up

**Solutions**:
1. Verify key at https://platform.openai.com/api-keys
2. Check key has no extra spaces
3. Ensure billing is configured
4. Generate new key if needed

```python
def validate_api_key(api_key):
    if not api_key:
        raise ValueError("API key is empty")

    if not api_key.startswith('sk-'):
        raise ValueError("Invalid API key format (should start with 'sk-')")

    # Test key
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()  # Simple test call
        return True
    except openai.AuthenticationError:
        return False
```

---

### ChromaDB Errors

#### Error: "Collection already exists"
```
chromadb.errors.UniqueConstraintError: Collection 'my_collection' already exists
```

**Solution**:
```python
# Check if exists first
try:
    collection = client.get_collection("my_collection")
    print("Collection exists, using existing")
except:
    collection = client.create_collection("my_collection")
    print("Created new collection")

# Or delete and recreate
try:
    client.delete_collection("my_collection")
except:
    pass
collection = client.create_collection("my_collection")
```

---

## Debugging Workflows

### Debugging RAG Issues

```
Problem: RAG returns irrelevant documents

Step 1: Check embedding quality
→ Print query embedding
→ Print document embeddings
→ Calculate manual similarity

Step 2: Inspect retrieved documents
→ Print top 5 results with scores
→ Check if relevant docs exist in DB

Step 3: Test chunking strategy
→ Try different chunk sizes
→ Check chunk overlap
→ Inspect chunk content

Step 4: Verify search parameters
→ Adjust top_k value
→ Try different similarity metrics
→ Test with query expansion

Step 5: Evaluate end-to-end
→ Create test suite
→ Measure retrieval accuracy
→ Track false positives/negatives
```

---

### Debugging Agent Loops

```
Problem: Agent gets stuck or loops indefinitely

Step 1: Add verbose logging
→ Log each agent step
→ Log tool calls and results
→ Log decision reasoning

Step 2: Set max iterations
→ Add iteration counter
→ Force exit after N attempts

Step 3: Check tool outputs
→ Verify tools return expected format
→ Handle error cases properly

Step 4: Review agent prompts
→ Make instructions clearer
→ Add examples of success
→ Define termination conditions

Step 5: Implement circuit breaker
→ Detect repeated actions
→ Break on no progress
→ Escalate to human
```

---

## Prevention Checklist

### Before Deployment

#### API Safety
- [ ] API keys stored in environment variables
- [ ] Rate limiting implemented
- [ ] Timeout handling in place
- [ ] Retry logic with exponential backoff
- [ ] Error messages don't expose sensitive info

#### Data Safety
- [ ] Input validation implemented
- [ ] SQL injection prevention
- [ ] XSS prevention
- [ ] Content moderation active
- [ ] PII detection/redaction

#### Performance
- [ ] Response streaming for long outputs
- [ ] Caching for repeated queries
- [ ] Database connection pooling
- [ ] Async operations where possible

#### Quality
- [ ] Test suite covers edge cases
- [ ] Evaluation metrics defined
- [ ] Monitoring dashboard set up
- [ ] Logging configured properly
- [ ] Alerts for failures

#### Cost Control
- [ ] Token counting implemented
- [ ] Max token limits set
- [ ] Cost tracking active
- [ ] Budget alerts configured
- [ ] Caching reduces redundant calls

---

**Use this guide whenever you encounter issues. Prevention is better than debugging!**
