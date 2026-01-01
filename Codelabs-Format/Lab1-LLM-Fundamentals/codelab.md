# Lab 1: LLM Fundamentals & API Usage

## 🎯 What You'll Build

**SupportGenie v0.1** - A professional AI chatbot with:
- ✅ Multi-provider API support (OpenAI, Claude, Gemini)
- ✅ Streaming responses for better UX
- ✅ Token tracking and cost calculation
- ✅ Professional conversation handling

---

## 📋 Lab Overview

| Property | Value |
|----------|-------|
| **Duration** | 60 minutes |
| **Difficulty** | Beginner |
| **Prerequisites** | Basic Python, API key |
| **Environment** | Jupyter Notebook or Python script |

### What You'll Learn
- 🎯 How Large Language Models work (high-level)
- 🎯 Make your first API calls to OpenAI, Claude, and Gemini
- 🎯 Understand tokens and how to count them
- 🎯 Master key parameters: temperature, max_tokens, top_p
- 🎯 Implement streaming responses
- 🎯 Calculate and optimize API costs
- 🎯 Build your first AI chatbot

---

## 📚 Table of Contents

1. [Introduction](#1-introduction)
2. [Setup Your Environment](#2-setup-your-environment)
3. [How LLMs Work](#3-how-llms-work)
4. [Understanding Tokens](#4-understanding-tokens)
5. [Your First API Call](#5-your-first-api-call)
6. [Mastering Parameters](#6-mastering-parameters)
7. [Streaming Responses](#7-streaming-responses)
8. [Cost Calculation](#8-cost-calculation)
9. [Building Your Chatbot](#9-building-your-chatbot)
10. [Capstone: SupportGenie v0.1](#10-capstone-supportgenie-v01)
11. [Review & Next Steps](#11-review--next-steps)

---

## 1. Introduction

### Welcome to the Exciting World of AI! 🚀

Forget simple "Hello World" programs. In this lab, you'll build a **production-ready AI chatbot** that can:
- Answer questions naturally
- Stream responses in real-time
- Track costs automatically
- Handle errors gracefully

**Duration:** 5 minutes

### Why This Matters

Large Language Models are transforming every industry:
- 💼 Customer support automation
- 📊 Data analysis and insights
- 🤖 Personal AI assistants
- 📝 Content generation

By the end of this lab, you'll understand the fundamentals and have a working chatbot!

---

## 2. Setup Your Environment

**Duration:** 10 minutes

### 🛠️ Step 1: Install Required Packages

Open your terminal and run:

```bash
pip install openai anthropic google-generativeai tiktoken python-dotenv
```

**What these do:**
- `openai` - Access GPT models
- `anthropic` - Access Claude models
- `google-generativeai` - Access Gemini models
- `tiktoken` - Count tokens accurately
- `python-dotenv` - Manage API keys securely

---

### 🔑 Step 2: Get Your API Keys

You'll need at least ONE of these:

#### OpenAI API Key
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to API Keys
4. Create new secret key
5. Copy and save it securely

#### Anthropic API Key (Claude)
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up or log in
3. Navigate to API Keys
4. Create new key
5. Copy and save it securely

#### Google API Key (Gemini)
1. Go to [makersuite.google.com](https://makersuite.google.com)
2. Get API key
3. Copy and save it securely

---

### 🚨 CRITICAL: Secure Your API Keys

**NEVER** hardcode API keys in your code!

Create a `.env` file:

```bash
# .env file (DO NOT commit to Git!)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here
```

Add to `.gitignore`:
```bash
echo ".env" >> .gitignore
```

---

### ✅ Step 3: Verify Installation

Run this test script:

```python
# test_setup.py
import openai
import anthropic
import google.generativeai as genai
import tiktoken
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check if keys are loaded
print("✅ Packages imported successfully!")
print(f"OpenAI key loaded: {bool(os.getenv('OPENAI_API_KEY'))}")
print(f"Anthropic key loaded: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
print(f"Google key loaded: {bool(os.getenv('GOOGLE_API_KEY'))}")
```

**Expected output:**
```
✅ Packages imported successfully!
OpenAI key loaded: True
Anthropic key loaded: True
Google key loaded: True
```

---

## 3. How LLMs Work

**Duration:** 10 minutes

### What is a Large Language Model?

🧠 **Simple Definition:** An LLM is a neural network trained on massive amounts of text to predict the next word in a sequence.

Think of it like an **incredibly sophisticated autocomplete**.

---

### The Training Process (Simplified)

```
┌────────────────────────────────────────────────────┐
│  STEP 1: PRE-TRAINING                              │
│  ─────────────────────────────────────────         │
│  Input: Billions of words from books, websites    │
│  Task: Predict the next word                       │
│  Result: Model learns grammar, facts, reasoning    │
└────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────┐
│  STEP 2: FINE-TUNING                               │
│  ─────────────────────────────────────────         │
│  Input: High-quality instruction-response pairs    │
│  Task: Follow instructions accurately              │
│  Result: Model becomes a helpful assistant         │
└────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────┐
│  STEP 3: ALIGNMENT (RLHF)                          │
│  ─────────────────────────────────────────         │
│  Input: Human feedback on responses                │
│  Task: Be helpful, harmless, honest                │
│  Result: Safe, reliable AI assistant               │
└────────────────────────────────────────────────────┘
```

---

### How Text Generation Works

**Example:** User types "The weather today is"

```python
# LLM Process:
# 1. Tokenize input
tokens = ["The", "weather", "today", "is"]

# 2. Predict next token probabilities
probabilities = {
    "sunny": 30%,
    "nice": 25%,
    "cloudy": 20%,
    "rainy": 15%,
    "...": 10%
}

# 3. Sample based on temperature (more on this later!)
# 4. Add chosen token to sequence
# 5. Repeat until done
```

---

### 💡 Key Insight

**LLMs don't "know" facts** - they predict statistically likely continuations based on their training data.

This is why they can:
- ✅ Write convincingly
- ✅ Solve complex problems
- ❌ Sometimes "hallucinate" (make up facts)
- ❌ Get math wrong occasionally

---

### 🎯 Checkpoint

**Quick Quiz:** What does an LLM actually do?

<details>
<summary>Click to reveal answer</summary>

An LLM predicts the most likely next token (word/piece) based on the input sequence and its training data. It doesn't truly "understand" or "know" - it generates statistically probable text.

</details>

---

## 4. Understanding Tokens

**Duration:** 8 minutes

### What Are Tokens?

**Tokens** are the basic units that LLMs process. They're **not quite words, not quite characters**.

🎯 **Rule of Thumb:**
- 1 token ≈ **4 characters** in English
- 1 token ≈ **¾ of a word**
- 100 tokens ≈ **75 words**

---

### Examples

```python
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

---

### Why Tokens Matter

#### 1. **Context Limits**
Models have maximum token limits:
- GPT-3.5-turbo: 4,096 tokens (4K)
- GPT-4: 8,192 tokens (8K)
- GPT-4-turbo: 128,000 tokens (128K)
- Claude 3: 200,000 tokens (200K)

#### 2. **Cost**
APIs charge **per token** (input + output)

#### 3. **Performance**
More tokens = slower response time

---

### 🛠️ Hands-On: Counting Tokens

**Where to Find It:** Create a new Python file or Jupyter cell

```python
import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text for a specific model"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

# Test it out!
examples = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog",
    "OpenAI GPT-4 is a large language model",
]

for text in examples:
    token_count = count_tokens(text)
    print(f"Text: '{text}'")
    print(f"Tokens: {token_count}")
    print(f"Characters: {len(text)}")
    print(f"Ratio: {len(text)/token_count:.2f} chars/token\n")
```

**Expected Output:**
```
Text: 'Hello, world!'
Tokens: 4
Characters: 13
Ratio: 3.25 chars/token

Text: 'The quick brown fox jumps over the lazy dog'
Tokens: 9
Characters: 44
Ratio: 4.89 chars/token

Text: 'OpenAI GPT-4 is a large language model'
Tokens: 10
Characters: 39
Ratio: 3.90 chars/token
```

---

### 🎯 Checkpoint

**Exercise:** Count tokens in this sentence:
```
"Large Language Models are transforming artificial intelligence applications worldwide."
```

<details>
<summary>Click to see answer</summary>

```python
count_tokens("Large Language Models are transforming artificial intelligence applications worldwide.")
# Result: approximately 11 tokens
```

</details>

---

## 5. Your First API Call

**Duration:** 10 minutes

Let's make your first call to an LLM! We'll start with OpenAI's GPT.

---

### 🚀 OpenAI API - Basic Chat Completion

**Where to Find It:** Create `first_api_call.py`

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
print(f"Answer: {answer}")

# Check token usage
print(f"\nTokens used: {response.usage.total_tokens}")
print(f"  - Prompt: {response.usage.prompt_tokens}")
print(f"  - Completion: {response.usage.completion_tokens}")
```

**Expected Output:**
```
Answer: The capital of France is Paris.

Tokens used: 28
  - Prompt: 20
  - Completion: 8
```

---

### 📝 Understanding Message Roles

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

**Role Purposes:**
- 🎭 **system**: Sets the AI's behavior, tone, and constraints
- 👤 **user**: The human's messages
- 🤖 **assistant**: The AI's previous responses (for conversation history)

---

### 🧪 Try It: Anthropic Claude API

**Where to Find It:** Same file or new cell

```python
from anthropic import Anthropic
import os

client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

message = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(f"Answer: {message.content[0].text}")
print(f"\nTokens used:")
print(f"  - Input: {message.usage.input_tokens}")
print(f"  - Output: {message.usage.output_tokens}")
```

**Key Differences from OpenAI:**
- Uses `messages.create()` instead of `chat.completions.create()`
- System message is separate parameter (not in messages list)
- Returns structured `Message` object

---

### 🎯 Checkpoint

**Exercise:** Modify the code to ask: "Explain quantum computing in one sentence"

Run it with both OpenAI and Claude. Compare the responses!

---

## 6. Mastering Parameters

**Duration:** 12 minutes

Time to learn the "dials and knobs" that control LLM behavior!

---

### 🌡️ Temperature (0.0 - 2.0)

Controls **randomness/creativity** of responses.

```
┌─────────────────────────────────────────────────┐
│  Temperature Effect on Token Selection          │
├─────────────────────────────────────────────────┤
│                                                  │
│  Original probabilities: [60%, 25%, 10%, 5%]    │
│                                                  │
│  Temperature = 0.1 (Sharp):                     │
│  → [95%, 4%, 0.8%, 0.2%] ← Heavily favor top   │
│                                                  │
│  Temperature = 1.0 (Normal):                    │
│  → [60%, 25%, 10%, 5%] ← Unchanged              │
│                                                  │
│  Temperature = 2.0 (Flat):                      │
│  → [40%, 30%, 20%, 10%] ← More balanced         │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

### 🛠️ Hands-On: Temperature Experiment

**Where to Find It:** Create `temperature_test.py`

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

prompt = "Name a color"

temperatures = [0.0, 0.7, 1.5]

for temp in temperatures:
    print(f"\n{'='*50}")
    print(f"Temperature: {temp}")
    print('='*50)

    # Run 5 times to see variation
    for i in range(5):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=10
        )
        print(f"  {i+1}. {response.choices[0].message.content}")
```

**Expected Output:**
```
==================================================
Temperature: 0.0
==================================================
  1. Blue
  2. Blue
  3. Blue
  4. Blue
  5. Blue

==================================================
Temperature: 0.7
==================================================
  1. Blue
  2. Red
  3. Green
  4. Blue
  5. Yellow

==================================================
Temperature: 1.5
==================================================
  1. Cerulean
  2. Magenta
  3. Periwinkle
  4. Chartreuse
  5. Crimson
```

---

### 📊 When to Use Each Temperature

| Range | Use Case | Example |
|-------|----------|---------|
| **0.0** | Deterministic, consistent | Math problems, code generation, JSON parsing |
| **0.0-0.3** | Mostly factual | Data extraction, summarization, translation |
| **0.3-0.7** | Balanced | Customer support, chatbots, Q&A |
| **0.7-1.0** | Creative but coherent | Content writing, email drafting |
| **1.0-1.5** | High creativity | Creative writing, brainstorming, marketing |
| **1.5-2.0** | Maximum creativity | Experimental, poetry (may lose coherence) |

---

### 🎯 max_tokens

Limits the **maximum response length** (output only).

**Formula:**
```
Input tokens + max_tokens ≤ Context Window

Example:
- Prompt: 500 tokens
- max_tokens: 1000
- Total needed: 1500 tokens
- GPT-3.5-turbo (4K): ✅ Fits
```

---

### 🛠️ Hands-On: max_tokens Experiment

```python
response_lengths = [50, 200, 500]

for max_tok in response_lengths:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Explain quantum computing"}],
        max_tokens=max_tok
    )

    content = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason

    print(f"\nmax_tokens: {max_tok}")
    print(f"Finish reason: {finish_reason}")
    print(f"Actual tokens: {response.usage.completion_tokens}")
    print(f"Response: {content[:100]}...")
```

**finish_reason values:**
- `"stop"` - Natural completion
- `"length"` - Hit max_tokens limit (truncated)

---

### 🎯 top_p (Nucleus Sampling)

Alternative to temperature. Limits tokens to cumulative probability threshold.

```
Token probabilities (sorted):
Token A: 40%  ███████████████████
Token B: 30%  ██████████████
Token C: 15%  ███████
Token D: 10%  ████
Token E: 3%   █
Token F: 2%   █

top_p = 0.5:  Only A and B (40% + 30% = 70% ≥ 50%)
top_p = 0.9:  A, B, C, and D (95% ≥ 90%)
```

**🎯 Pro Tip:** Use temperature **OR** top_p, not both!
- Production apps often use `top_p=0.9`
- Provides better quality control than temperature

---

### 🎯 Checkpoint

**Exercise:** Experiment with these combinations:

1. `temperature=0, max_tokens=50` - What happens?
2. `temperature=2.0, max_tokens=50` - What happens?
3. `top_p=0.1, max_tokens=100` - What happens?

---

## 7. Streaming Responses

**Duration:** 8 minutes

### Why Stream?

❌ **Problem:** Long responses take 10+ seconds. User sees nothing until complete.
✅ **Solution:** Stream tokens as they're generated. Better UX!

---

### 🛠️ Hands-On: Implement Streaming

**Where to Find It:** Create `streaming_demo.py`

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def stream_chat(message):
    """Stream a chat response token by token"""

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}],
        stream=True,  # ← Enable streaming!
        max_tokens=200
    )

    full_response = ""
    print("Assistant: ", end="", flush=True)

    for chunk in stream:
        # Check if chunk has content
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)  # ← Print immediately
            full_response += content

    print()  # New line at end
    return full_response

# Test it!
response = stream_chat("Tell me a short story about a robot")
```

**Expected Output:**
```
Assistant: Once upon a time, there was a small robot named Bolt...
(text appears word by word in real-time)
```

---

### 🎯 Checkpoint

**Exercise:** Modify the streaming function to also count tokens as they arrive.

<details>
<summary>Hint</summary>

Add a counter and increment it for each chunk received!

</details>

---

## 8. Cost Calculation

**Duration:** 5 minutes

Understanding costs is crucial for production applications!

### 💰 Pricing (December 2024)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-3.5-turbo | $0.50 | $1.50 |
| GPT-4-turbo | $10.00 | $30.00 |
| GPT-4 | $30.00 | $60.00 |
| Claude Haiku | $0.25 | $1.25 |
| Claude Sonnet | $3.00 | $15.00 |
| Claude Opus | $15.00 | $75.00 |
| Gemini Pro | $0.125 | $0.375 |

---

### 🛠️ Hands-On: Cost Calculator

**Where to Find It:** Create `cost_calculator.py`

```python
import tiktoken

def calculate_cost(prompt, response, model="gpt-3.5-turbo"):
    """Calculate cost of an API call"""

    # Pricing per 1M tokens
    pricing = {
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
    }

    # Count tokens
    encoding = tiktoken.encoding_for_model(model)
    input_tokens = len(encoding.encode(prompt))
    output_tokens = len(encoding.encode(response))

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

# Test it
prompt = "Explain machine learning in simple terms"
response = "Machine learning is a way for computers to learn from data..."

cost_info = calculate_cost(prompt, response)
print(f"Input tokens: {cost_info['input_tokens']}")
print(f"Output tokens: {cost_info['output_tokens']}")
print(f"Total cost: ${cost_info['total_cost']:.6f}")
```

---

## 9. Building Your Chatbot

**Duration:** 10 minutes

Now let's combine everything into a reusable chatbot class!

---

### 🛠️ Hands-On: SimpleChatbot Class

**Where to Find It:** Create `chatbot.py`

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

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


# Test it!
if __name__ == "__main__":
    bot = SimpleChatbot(api_key=os.getenv('OPENAI_API_KEY'))

    bot.set_system_message("You are a helpful assistant. Keep responses under 100 words.")

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

### 🎯 Try It Out!

Run the chatbot and have a conversation:
```bash
python chatbot.py
```

---

## 10. Capstone: SupportGenie v0.1

**Duration:** 15 minutes

Let's build a professional customer support chatbot!

---

### 🎯 Project Requirements

Build **SupportGenie** with:
- ✅ Professional, empathetic tone
- ✅ Streaming responses
- ✅ Token and cost tracking
- ✅ Session statistics
- ✅ Error handling

---

### 🛠️ Hands-On: Build SupportGenie

**Where to Find It:** Create `supportgenie.py`

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class SupportGenieV1:
    """
    SupportGenie - AI Customer Support Assistant
    Version 0.1: Basic chatbot with professional tone
    """

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"
        self.conversation_history = []
        self.total_tokens = 0
        self.total_cost = 0.0

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

    def set_system_message(self, message):
        """Set the system prompt"""
        self.conversation_history = [
            {"role": "system", "content": message}
        ]

    def chat(self, user_message, stream=True):
        """Send a message and get response"""

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Stream response
        stream_obj = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            temperature=0.7,
            max_tokens=500,
            stream=True
        )

        full_response = ""
        for chunk in stream_obj:
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

    def get_stats(self):
        """Get usage statistics"""
        return {
            "messages": len([m for m in self.conversation_history if m['role'] == 'user']),
            "total_cost": f"${self.total_cost:.4f}"
        }

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
                print(f"\nSession Stats: {stats['messages']} messages")
                break

            if user_input.lower() == 'stats':
                print(f"\n{self.get_stats()}\n")
                continue

            print("\nSupportGenie: ", end="")
            self.chat(user_input, stream=True)
            print()


# Run it!
if __name__ == "__main__":
    bot = SupportGenieV1(api_key=os.getenv('OPENAI_API_KEY'))
    bot.run()
```

---

### 🎯 Test Your SupportGenie

Try these example interactions:

```
You: My order hasn't arrived yet
SupportGenie: I understand your concern about your order...

You: I don't have my order number
SupportGenie: No problem! Let me connect you with...
```

---

### 🎯 Challenge: Enhance It!

Add these features:
1. **Cost warnings** - Alert when cost exceeds $0.10
2. **Response time tracking** - Show how long each response took
3. **Error handling** - Gracefully handle API errors
4. **History export** - Save conversation to file

---

## 11. Review & Next Steps

**Duration:** 5 minutes

### 🎉 Congratulations!

You've completed Lab 1! You now know:
- ✅ How LLMs work (prediction, not magic!)
- ✅ Tokens and why they matter
- ✅ Making API calls to OpenAI, Claude, Gemini
- ✅ Key parameters (temperature, max_tokens, top_p)
- ✅ Streaming for better UX
- ✅ Cost calculation and optimization
- ✅ Building production chatbots

---

### 📊 Key Takeaways

✅ **LLMs predict the next token** based on training data
✅ **Tokens are the basic unit** - ~4 chars or ¾ word
✅ **Temperature controls randomness** - lower = predictable, higher = creative
✅ **max_tokens limits response length** - plan for context window
✅ **top_p filters by probability** - preferred for production
✅ **Streaming improves UX** for long responses
✅ **Track costs carefully** - they add up quickly!
✅ **System messages set behavior** - use them wisely

---

### 🚀 Next Steps

**Ready for more?**

👉 [Lab 2: Prompt Engineering](../Lab2-Prompt-Engineering/codelab.md)

Learn to write prompts that get exceptional results every time!

---

### 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude Docs](https://docs.anthropic.com/)
- [Tiktoken Library](https://github.com/openai/tiktoken)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)

---

### 🤝 Need Help?

- Check the troubleshooting section
- Join our Discord community
- Review the code examples
- Ask questions in discussions

---

**You're now ready to build AI applications! 🎉**

---

## 📌 Quick Reference

### Common Code Snippets

**Basic API Call:**
```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=100
)
```

**Streaming:**
```python
stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**Count Tokens:**
```python
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokens = len(encoding.encode(text))
```

---

**End of Lab 1** ✅
