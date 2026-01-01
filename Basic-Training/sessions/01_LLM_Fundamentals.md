# Session 1: LLM Fundamentals & API Usage

**Duration**: 60 minutes
**Difficulty**: Beginner
**Colab Notebook**: [01_LLM_Fundamentals.ipynb](../notebooks/01_LLM_Fundamentals.ipynb)

---

## Learning Objectives

By the end of this session, you will:
- 🎯 Understand how Large Language Models work (high-level)
- 🎯 Make your first API calls to OpenAI, Claude, and Gemini
- 🎯 Understand tokens and how to count them
- 🎯 Master key parameters: temperature, max_tokens, top_p, and top_k
- 🎯 Implement streaming responses
- 🎯 Calculate and optimize API costs
- 🎯 Build your first AI chatbot

---

## Capstone Project: Session 1 Build

**What You'll Build**: Basic chatbot for SupportGenie
- Simple question-answering interface
- Token management and cost tracking
- Streaming responses for better UX
- Basic conversation handling

---

## Part 1: How LLMs Work (High-Level)

### What is a Large Language Model?

**Simple Definition**: An LLM is a neural network trained on massive amounts of text to predict the next word in a sequence.

### The Training Process (Simplified)

```
1. PRE-TRAINING
   Input: Billions of words from books, websites, articles
   Task: Predict the next word
   Result: Model learns grammar, facts, reasoning

2. FINE-TUNING
   Input: High-quality instruction-response pairs
   Task: Follow instructions accurately
   Result: Model becomes a helpful assistant

3. ALIGNMENT
   Input: Human feedback on responses
   Task: Be helpful, harmless, honest
   Result: Safe, reliable AI assistant
```

### How Text Generation Works

```
User: "The weather today is"

LLM Process:
1. Tokenize: ["The", "weather", "today", "is"]
2. Predict next token probabilities:
   - "sunny" (30%)
   - "nice" (25%)
   - "cloudy" (20%)
   - "rainy" (15%)
   - ... (10%)

3. Sample based on temperature
4. Add chosen token to sequence
5. Repeat until done
```

**Key Insight**: LLMs don't "know" facts—they predict statistically likely continuations.

---

## Part 2: Understanding Tokens

### What Are Tokens?

**Tokens** are the basic units that LLMs process. Not quite words, not quite characters.

**Rule of Thumb**:
- 1 token ≈ 4 characters in English
- 1 token ≈ ¾ of a word
- 100 tokens ≈ 75 words

### Examples

```
Text: "Hello, world!"
Tokens: ["Hello", ",", " world", "!"]
Count: 4 tokens

Text: "ChatGPT is amazing"
Tokens: ["Chat", "G", "PT", " is", " amazing"]
Count: 5 tokens

Text: "OpenAI"
Tokens: ["Open", "AI"]
Count: 2 tokens
```

### Why Tokens Matter

1. **Context Limits**: Models have max token limits (e.g., 4K, 16K, 128K)
2. **Cost**: APIs charge per token
3. **Performance**: More tokens = slower response

### Counting Tokens with Tiktoken

```python
import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text for a specific model"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

# Examples
print(count_tokens("Hello, world!"))  # 4
print(count_tokens("The quick brown fox jumps"))  # 5
```

---

## Part 3: Making Your First API Call

### OpenAI API - GPT Models

#### Basic Chat Completion

```python
from openai import OpenAI
import os

# Initialize client
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Make API call
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=50,
    temperature=0.7
)

# Extract answer
answer = response.choices[0].message.content
print(answer)

# Check token usage
print(f"Tokens used: {response.usage.total_tokens}")
```

**Output**:
```
The capital of France is Paris.
Tokens used: 28
```

---

### Message Roles Explained

```python
messages = [
    {
        "role": "system",
        "content": "You are a customer support agent for TechStore."
    },
    {
        "role": "user",
        "content": "How do I return a product?"
    },
    {
        "role": "assistant",
        "content": "To return a product, please contact..."
    },
    {
        "role": "user",
        "content": "How long does it take?"
    }
]
```

**Role Purposes**:
- **system**: Sets the AI's behavior, tone, and constraints
- **user**: The human's messages
- **assistant**: The AI's previous responses (for conversation history)

---

### Anthropic Claude API

```python
from anthropic import Anthropic
import os

client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

message = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(message.content[0].text)
print(f"Tokens used: Input={message.usage.input_tokens}, Output={message.usage.output_tokens}")
```

**Key Differences from OpenAI**:
- Uses `messages.create()` instead of `chat.completions.create()`
- System message is separate parameter (not in messages list)
- Returns structured `Message` object

---

### Google Gemini API

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content('What is the capital of France?')

print(response.text)
```

**Key Differences**:
- Simpler API for basic use
- Generous free tier
- Different model naming

---

## Part 4: Key Parameters

### Temperature (0.0 - 2.0)

Controls randomness/creativity of responses by adjusting the probability distribution over tokens.

**How it works**:
- Temperature modifies the "sharpness" of the probability distribution
- **Low temperature (near 0)**: Sharpens distribution → model picks highest probability tokens
- **High temperature (near 2)**: Flattens distribution → model considers lower probability tokens

**Mathematical Effect**:
```
Original probabilities: [0.6, 0.25, 0.1, 0.05]

Temperature = 0.1 (Sharp):
→ [0.95, 0.04, 0.008, 0.002]  # Heavily favors top choice

Temperature = 1.0 (Normal):
→ [0.6, 0.25, 0.1, 0.05]      # Unchanged

Temperature = 2.0 (Flat):
→ [0.4, 0.3, 0.2, 0.1]        # More balanced distribution
```

**Practical Examples**:

```python
# Temperature = 0 (Deterministic)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Name a color"}],
    temperature=0.0
)
# Likely output: "Blue" (consistent every time)
# Use case: Data extraction, classification, factual Q&A

# Temperature = 0.7 (Balanced)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Name a color"}],
    temperature=0.7
)
# Varied output: "Blue", "Red", "Green", "Yellow"
# Use case: General conversations, customer support

# Temperature = 1.5 (Creative)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Name a color"}],
    temperature=1.5
)
# Creative output: "Cerulean", "Magenta", "Chartreuse"
# Use case: Creative writing, brainstorming, marketing copy
```

**When to use each range**:
- **0.0**: Deterministic outputs (same input → same output)
  - Math problems, code generation, JSON parsing
  - When consistency is critical
- **0.0-0.3**: Mostly factual, minimal variation
  - Data extraction, summarization, translation
  - Technical documentation
- **0.3-0.7**: Slight creativity, natural variation
  - Customer support, Q&A systems
  - Educational content
- **0.7-1.0**: Balanced creativity and coherence
  - General chatbots, content writing
  - Email drafting, casual conversation
- **1.0-1.5**: High creativity, diverse outputs
  - Creative writing, storytelling
  - Marketing slogans, ideation
- **1.5-2.0**: Maximum creativity (may lose coherence)
  - Experimental outputs, poetry
  - Brainstorming unusual ideas

**Important Notes**:
- Default is typically `1.0` if not specified
- Temperature doesn't affect the model's "knowledge"—only how it samples from its predictions
- For production apps, start with lower values (0.3-0.7) for reliability

---

### max_tokens

Limits the maximum number of tokens in the generated response (completion).

**What it controls**:
- Sets an upper bound on response length
- Model stops generating when hitting this limit OR when it naturally completes
- Does NOT include input tokens (only output/completion tokens)

**Understanding Token Limits**:

Each model has a **context window** (total tokens allowed):
- GPT-3.5-turbo: 4,096 tokens (newer versions: 16K)
- GPT-4: 8,192 tokens (turbo: 128K)
- Claude 3: 200,000 tokens
- Gemini Pro: 32,768 tokens

**Formula**:
```
Input tokens + max_tokens ≤ Context Window

Example:
- Prompt: 500 tokens
- max_tokens: 1000
- Total needed: 1500 tokens
- GPT-3.5-turbo (4K): ✅ Fits
- GPT-3.5-turbo (4K) with 3800 prompt: ❌ Exceeds limit
```

**Practical Examples**:

```python
# Short response (50 tokens ≈ 37 words)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=50
)
# Output: "Quantum computing uses quantum bits (qubits) that can exist
# in multiple states simultaneously, enabling..."
# Note: Might cut off mid-sentence!

# Medium response (200 tokens ≈ 150 words)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=200
)
# Output: Complete paragraph with key concepts

# Long response (500 tokens ≈ 375 words)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=500
)
# Output: Detailed explanation with examples
```

**Important Behaviors**:

1. **Early Stopping**: Model stops when naturally done (even if under limit)
```python
max_tokens=1000
# Question: "What is 2+2?"
# Response: "4" (only ~1 token, not 1000)
```

2. **Truncation**: Response cuts off if limit reached mid-sentence
```python
max_tokens=10
# Question: "Write a story about a dragon"
# Response: "Once upon a time, there was a mighty dragon who lived in..."
# (cuts off abruptly)
```

3. **No tokens left**: Error if prompt uses entire context window
```python
prompt_tokens = 4090  # Almost full context (4096)
max_tokens = 500      # Requests 500 more
# Error: Would exceed 4096 token limit
```

**Best Practices**:

- **Cost Control**: Set reasonable limits to avoid runaway costs
  ```python
  max_tokens=100   # Caps cost per request
  ```

- **User Experience**: Allow enough for complete thoughts
  ```python
  max_tokens=50    # ❌ Too short, likely truncated
  max_tokens=200   # ✅ Good for most responses
  max_tokens=1000  # ✅ Good for detailed explanations
  ```

- **Context Window Math**: Leave room for conversation history
  ```python
  # For chatbots with history:
  # context_limit = 4096
  # conversation_history = ~2000 tokens
  # max_tokens = 500
  # Safety: 2000 + 500 = 2500 < 4096 ✅
  ```

- **Default Values**: If omitted, models use their defaults
  - OpenAI: No default (can generate to context limit)
  - Claude: Must be specified
  - Gemini: Uses model default

**Recommended Settings by Use Case**:
- **Chatbots**: 150-300 (conversational responses)
- **Summarization**: 100-200 (concise summaries)
- **Code Generation**: 500-1000 (complete functions)
- **Long-form Content**: 1000-2000 (articles, essays)
- **Simple Q&A**: 50-100 (short answers)

**Monitoring Token Usage**:
```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100
)

# Check actual usage
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
print(f"Stopped because: {response.choices[0].finish_reason}")
# finish_reason can be: "stop" (natural), "length" (hit max_tokens)
```

---

### top_p (Nucleus Sampling)

Alternative to temperature that limits token choices to a cumulative probability threshold.

**How it works**:
- Selects from the smallest set of tokens whose cumulative probability ≥ `top_p`
- Also called "nucleus sampling" (picks from the probability "nucleus")
- More dynamic than `top_k` - adapts set size based on probability distribution

**Visual Example**:

```
Token probabilities (sorted):
Token A: 40%  ███████████████████
Token B: 30%  ██████████████
Token C: 15%  ███████
Token D: 10%  ████
Token E: 3%   █
Token F: 2%   █

top_p = 0.5:  Only A and B (40% + 30% = 70% ≥ 50%)
top_p = 0.7:  A, B, and C (40% + 30% + 15% = 85% ≥ 70%)
top_p = 0.9:  A, B, C, and D (40% + 30% + 15% + 10% = 95% ≥ 90%)
top_p = 1.0:  All tokens considered
```

**Practical Examples**:

```python
# top_p = 0.1 (Very focused - only highest probability tokens)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Complete: The sky is"}],
    top_p=0.1
)
# Very predictable outputs: "blue", "clear"
# Use case: When you want consistency but not complete determinism

# top_p = 0.5 (Moderate - top half of probability mass)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Complete: The sky is"}],
    top_p=0.5
)
# Balanced outputs: "blue", "clear", "cloudy", "gray"
# Use case: Structured outputs with some variety

# top_p = 0.9 (Broad - most of probability mass)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Complete: The sky is"}],
    top_p=0.9
)
# More varied: "blue", "cloudy", "beautiful", "vast", "endless"
# Use case: Creative tasks while avoiding very unlikely tokens

# top_p = 1.0 (All tokens - no filtering)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Complete: The sky is"}],
    top_p=1.0
)
# Maximum variety: Any token possible
# Use case: Maximum creativity (rare in production)
```

**top_p vs Temperature**:

| Aspect | top_p | temperature |
|--------|-------|-------------|
| **Mechanism** | Filters tokens by probability | Reshapes probability distribution |
| **Effect** | Dynamic cutoff | Global scaling |
| **Predictability** | Adapts to context | Consistent behavior |
| **Best for** | Controlling quality floor | Controlling creativity level |

**When to use top_p**:

- **0.1**: Very conservative outputs
  - Factual Q&A, data extraction
  - When wrong answers are costly

- **0.5**: Moderate diversity
  - Customer service responses
  - Structured content generation

- **0.9-0.95**: High diversity (most common)
  - General chatbots
  - Creative writing with quality control

- **1.0**: No filtering
  - Experimental purposes
  - Maximum creativity (use with caution)

**Combining with Other Parameters**:

```python
# ❌ AVOID: Using both temperature and top_p
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.8,
    top_p=0.9  # Conflicting sampling strategies
)

# ✅ RECOMMENDED: Use one or the other
# Option 1: Temperature only
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)

# Option 2: top_p only
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    top_p=0.9,
    temperature=1  # Keep at 1 when using top_p
)
```

**Default Values**:
- OpenAI default: `top_p=1.0`
- Most production apps use: `top_p=0.9` or `temperature=0.7`

**Key Insight**:
`top_p` is often preferred over `temperature` in production because it provides better quality control—it filters out very unlikely (potentially nonsensical) tokens while still allowing creativity within the reasonable probability range.

---

### top_k (Top-K Sampling)

Limits the model to consider only the top K most likely tokens at each step.

**How it works**:
- Sorts all possible tokens by probability
- Keeps only the top K tokens
- Samples from this reduced set
- More straightforward than `top_p`, but less adaptive

**Visual Example**:

```
All token probabilities (sorted):
Token A: 40%  ███████████████████
Token B: 30%  ██████████████
Token C: 15%  ███████
Token D: 10%  ████
Token E: 3%   █
Token F: 2%   █
... (thousands more)

top_k = 2:  Only consider A and B
top_k = 4:  Only consider A, B, C, and D
top_k = 10: Consider top 10 tokens
top_k = 50: Consider top 50 tokens (common default)
```

**Practical Examples**:

```python
# Note: Not all APIs support top_k
# Google Gemini and some other models support it
# OpenAI GPT models do NOT support top_k (use top_p instead)

# Example with Google Gemini
import google.generativeai as genai

model = genai.GenerativeModel('gemini-pro')

# top_k = 1 (Always pick the most likely token - deterministic)
response = model.generate_content(
    'Complete: The sky is',
    generation_config=genai.types.GenerationConfig(
        top_k=1
    )
)
# Output: "blue" (always the same, most probable)

# top_k = 10 (Consider only top 10 tokens)
response = model.generate_content(
    'Complete: The sky is',
    generation_config=genai.types.GenerationConfig(
        top_k=10
    )
)
# Output: Limited variety from top 10 choices

# top_k = 40 (Common setting - good balance)
response = model.generate_content(
    'Complete: The sky is',
    generation_config=genai.types.GenerationConfig(
        top_k=40
    )
)
# Output: Good variety while filtering extreme outliers
```

**top_k vs top_p Comparison**:

| Feature | top_k | top_p |
|---------|-------|-------|
| **Fixed Set Size** | ✅ Always K tokens | ❌ Variable based on probabilities |
| **Adaptivity** | ❌ Same K for all contexts | ✅ Adapts to probability distribution |
| **Simplicity** | ✅ Easy to understand | ❌ More complex concept |
| **Quality Control** | ⚠️ Can include low-prob tokens | ✅ Better at filtering unlikely tokens |
| **Predictability** | ✅ Consistent behavior | ⚠️ Variable set size |

**The Problem with top_k**:

```
Scenario 1 - Clear winner:
Token A: 95%  (clear choice)
Token B: 3%
Token C: 1%
Token D: 0.5%
Token E: 0.5%

top_k=5: Includes E (0.5%) - probably shouldn't

Scenario 2 - Evenly distributed:
Token A: 15%
Token B: 14%
Token C: 13%
Token D: 12%
Token E: 11%
... (20 more tokens around 5-10%)

top_k=5: Might miss good options (F-Z)

→ top_k doesn't adapt to context!
→ top_p solves this by using probability mass instead
```

**When to use top_k**:

- **1**: Deterministic output (like temperature=0)
  - Same result every time
  - Good for testing, reproducible results

- **10-20**: Conservative, focused outputs
  - Factual tasks
  - When quality matters more than variety

- **40-50**: Balanced (common default)
  - General purpose tasks
  - Good middle ground

- **100+**: Very diverse outputs
  - Creative tasks
  - Brainstorming

**API Support**:

| Provider | Supports top_k? | Default Value | Recommendation |
|----------|----------------|---------------|----------------|
| **OpenAI (GPT)** | ❌ No | N/A | Use `top_p` instead |
| **Anthropic (Claude)** | ❌ No | N/A | Use `top_p` instead |
| **Google (Gemini)** | ✅ Yes | 40 | Can use both `top_k` and `top_p` |
| **Cohere** | ✅ Yes | None | Use `top_k` or `top_p` |
| **Open-source (HuggingFace)** | ✅ Yes | Varies | Usually available |

**Example with Multiple Parameters (Gemini)**:

```python
# Combining top_k, top_p, and temperature
response = model.generate_content(
    'Write a creative story opening',
    generation_config=genai.types.GenerationConfig(
        top_k=40,          # Limit to top 40 tokens
        top_p=0.95,        # Then apply nucleus sampling
        temperature=0.9,   # Finally, adjust randomness
        max_output_tokens=200
    )
)

# Processing order:
# 1. Filter to top_k tokens (40)
# 2. Further filter by top_p (95% cumulative probability)
# 3. Apply temperature to remaining set
# 4. Sample one token
```

**Best Practices**:

1. **Choose the right parameter for your API**:
   ```python
   # OpenAI/Claude - use top_p
   response = client.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=[{"role": "user", "content": "Hello"}],
       top_p=0.9
   )

   # Gemini - can use top_k or top_p (or both)
   response = model.generate_content(
       'Hello',
       generation_config=genai.types.GenerationConfig(
           top_k=40
       )
   )
   ```

2. **Don't overthink it**:
   - Most production apps do fine with just `temperature` or `top_p`
   - Only adjust `top_k` if you're using an API that supports it and have specific needs

3. **Start with defaults**:
   - Gemini default (`top_k=40`) works well for most cases
   - Only tune if you have specific quality issues

**Summary**:

`top_k` is a simpler but less adaptive alternative to `top_p`. It's useful when available, but most modern LLM APIs (OpenAI, Claude) prefer `top_p` for its superior adaptivity and quality control. If your API supports both, `top_p` is generally the better choice for production applications.

---

## Part 5: Streaming Responses

### Why Stream?

**Problem**: Long responses take 10+ seconds. User sees nothing until complete.

**Solution**: Stream tokens as they're generated. Better UX!

### Implementation

```python
def stream_chat(message):
    """Stream a chat response token by token"""

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}],
        stream=True,  # Enable streaming
        max_tokens=200
    )

    full_response = ""
    print("Assistant: ", end="", flush=True)

    for chunk in stream:
        # Check if chunk has content
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    print()  # New line at end
    return full_response

# Test
response = stream_chat("Tell me a short story about a robot")
```

**Output** (appears gradually):
```
Assistant: Once upon a time, there was a small robot named Bolt...
(text appears word by word in real-time)
```

---

## Part 6: Cost Calculation

### Pricing (as of December 2024)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-3.5-turbo | $0.50 | $1.50 |
| GPT-4-turbo | $10.00 | $30.00 |
| GPT-4 | $30.00 | $60.00 |
| Claude Haiku | $0.25 | $1.25 |
| Claude Sonnet | $3.00 | $15.00 |
| Claude Opus | $15.00 | $75.00 |
| Gemini Pro | $0.125 | $0.375 |

### Cost Calculator

```python
def calculate_cost(prompt, response, model="gpt-3.5-turbo"):
    """Calculate cost of an API call"""

    # Pricing per 1M tokens
    pricing = {
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
    }

    # Count tokens
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)

    # Calculate cost
    input_cost = (input_tokens / 1_000_000) * pricing[model]["input"]
    output_cost = (output_tokens / 1_000_000) * pricing[model]["output"]

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }

# Example
prompt = "Explain machine learning in simple terms"
response = "Machine learning is a way for computers to learn..."

cost_info = calculate_cost(prompt, response)
print(f"Total cost: ${cost_info['total_cost']:.6f}")
```

---

## Part 7: Building Your First Chatbot

### Simple Chatbot Class

```python
class SimpleChatbot:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history = []
        self.total_tokens = 0
        self.total_cost = 0.0

    def set_system_message(self, message):
        """Set the system prompt"""
        self.conversation_history = [
            {"role": "system", "content": message}
        ]

    def chat(self, user_message, stream=False):
        """Send a message and get response"""

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        if stream:
            return self._stream_response()
        else:
            return self._complete_response()

    def _complete_response(self):
        """Get complete response at once"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            temperature=0.7,
            max_tokens=500
        )

        assistant_message = response.choices[0].message.content

        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        # Track usage
        self.total_tokens += response.usage.total_tokens
        self.total_cost += self._calculate_call_cost(response.usage)

        return assistant_message

    def _stream_response(self):
        """Stream response token by token"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            temperature=0.7,
            max_tokens=500,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        print()  # New line

        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })

        return full_response

    def _calculate_call_cost(self, usage):
        """Calculate cost for this call"""
        pricing = {"gpt-3.5-turbo": {"input": 0.50, "output": 1.50}}

        input_cost = (usage.prompt_tokens / 1_000_000) * pricing[self.model]["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * pricing[self.model]["output"]

        return input_cost + output_cost

    def get_stats(self):
        """Get usage statistics"""
        return {
            "total_tokens": self.total_tokens,
            "total_cost": f"${self.total_cost:.6f}",
            "messages": len([m for m in self.conversation_history if m['role'] == 'user'])
        }

    def clear_history(self):
        """Reset conversation"""
        system_msg = [m for m in self.conversation_history if m['role'] == 'system']
        self.conversation_history = system_msg
```

### Using the Chatbot

```python
# Initialize
bot = SimpleChatbot(api_key=os.environ.get('OPENAI_API_KEY'))

# Set personality
bot.set_system_message(
    "You are a helpful and friendly AI assistant. "
    "Keep responses concise (under 100 words)."
)

# Have a conversation
print("Chatbot ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nChat Statistics:")
        print(bot.get_stats())
        break

    print("Assistant: ", end="")
    bot.chat(user_input, stream=True)
    print()
```

---

## Part 8: Capstone - SupportGenie v0.1

Let's build the first version of SupportGenie!

```python
class SupportGenieV1(SimpleChatbot):
    """
    SupportGenie - AI Customer Support Assistant
    Version 0.1: Basic chatbot with professional tone
    """

    def __init__(self, api_key):
        super().__init__(api_key, model="gpt-3.5-turbo")

        # Set professional customer service personality
        self.set_system_message("""
You are SupportGenie, an AI customer support assistant for TechStore.

Guidelines:
- Be professional, empathetic, and helpful
- Keep responses under 100 words
- If you don't know something, be honest
- Always offer to escalate to a human agent if needed
- Use a friendly but professional tone

Response Format:
1. Acknowledge the customer's concern
2. Provide helpful information or solution
3. Ask if there's anything else you can help with
        """)

    def welcome(self):
        """Display welcome message"""
        print("="* 60)
        print("    SupportGenie v0.1 - AI Customer Support")
        print("="* 60)
        print("\nHello! I'm SupportGenie, your AI support assistant.")
        print("How can I help you today?\n")
        print("(Type 'quit' to exit, 'stats' for usage info)\n")

    def run(self):
        """Run the support chatbot"""
        self.welcome()

        while True:
            user_input = input("You: ")

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using SupportGenie!")
                stats = self.get_stats()
                print(f"\nSession Stats: {stats['messages']} messages, {stats['total_cost']}")
                break

            if user_input.lower() == 'stats':
                print(f"\n{self.get_stats()}\n")
                continue

            print("\nSupportGenie: ", end="")
            self.chat(user_input, stream=True)
            print()

# Run it!
if __name__ == "__main__":
    import os
    bot = SupportGenieV1(api_key=os.environ.get('OPENAI_API_KEY'))
    bot.run()
```

**Example Interaction**:
```
You: My order hasn't arrived yet
SupportGenie: I understand your concern about your order. I'd be happy
to help you track it down. To assist you better, could you please provide
your order number? It usually starts with "ORD-" followed by numbers.
Alternatively, I can escalate this to a human agent who can access your
account directly. What would you prefer?

You: I don't have my order number
SupportGenie: No problem! Let me connect you with one of our support
specialists who can look up your order using your email or account
information. They'll be able to provide you with a detailed tracking
update. Would you like me to create a support ticket for you?
```

---

## Common Mistakes & Debugging

### ❌ Mistake #1: Hardcoded API Keys
```python
# WRONG!
api_key = "sk-1234567890"  # Exposed in code
```

✅ **Correct**:
```python
import os
api_key = os.environ.get('OPENAI_API_KEY')
```

### ❌ Mistake #2: Not Handling Errors
```python
# WRONG!
response = client.chat.completions.create(...)  # Can fail
```

✅ **Correct**:
```python
try:
    response = client.chat.completions.create(...)
except openai.RateLimitError:
    print("Rate limit exceeded. Please wait.")
except openai.APIError as e:
    print(f"API error: {e}")
```

### ❌ Mistake #3: Ignoring Token Limits
```python
# WRONG!
messages.append(every_single_message)  # Eventually exceeds limit
```

✅ **Correct**:
```python
# Keep only recent messages
if len(messages) > 20:
    messages = messages[:1] + messages[-19:]  # Keep system + recent
```

---

## Exercises

### Exercise 1: Token Counter Tool
Create a function that estimates the cost of processing a document.

### Exercise 2: Multi-Model Comparison
Call the same prompt with GPT-3.5, GPT-4, and Claude. Compare:
- Response quality
- Speed
- Cost

### Exercise 3: Temperature Experiment
Test the same prompt with temperatures 0, 0.5, 1.0, 1.5, 2.0. Observe differences.

### Exercise 4: Enhance SupportGenie
Add features:
- Conversation history management
- Cost warnings ($0.10 limit)
- Response time tracking

---

## Key Takeaways

✅ **LLMs predict the next token** based on training data
✅ **Tokens are the basic unit** - ~4 chars or ¾ word
✅ **Temperature controls randomness** - lower = more predictable, higher = more creative
✅ **max_tokens limits response length** - plan for context window limits
✅ **top_p (nucleus sampling) filters by probability** - preferred for production quality control
✅ **top_k limits token choices** - available in some APIs (Gemini, not OpenAI/Claude)
✅ **Use temperature OR top_p, not both** - they serve similar purposes
✅ **Streaming improves UX** for long responses
✅ **Track costs carefully** - they add up quickly!
✅ **System messages set behavior** - use them to define personality and constraints

---

## Next Session Preview

In **Session 2: Prompt Engineering**, you'll learn:
- How to write effective prompts
- Few-shot learning techniques
- Chain-of-thought prompting
- Prompt templates and patterns
- How to make SupportGenie much smarter!

---

## Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude Docs](https://docs.anthropic.com/)
- [Tiktoken Library](https://github.com/openai/tiktoken)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)

---

**Session 1 Complete!** 🎉
**Next**: [Session 2: Prompt Engineering →](02_Prompt_Engineering.md)
