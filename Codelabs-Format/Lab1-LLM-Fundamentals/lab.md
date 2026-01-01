# Lab 1: LLM Fundamentals & API Usage
## 🛠️ Hands-On Lab

> **Purpose:** Apply what you learned through practical coding exercises

---

## 📋 Lab Overview

| Property | Value |
|----------|-------|
| **Duration** | 60-90 minutes (coding) |
| **Difficulty** | Beginner |
| **Prerequisites** | Completed [learning.md](learning.md) |
| **What You'll Build** | SupportGenie v0.1 - Professional AI chatbot |

---

## 📖 Table of Contents

1. [Setup Your Environment](#1-setup-your-environment)
2. [Exercise 1: First API Call](#2-exercise-1-first-api-call)
3. [Exercise 2: Token Counting](#3-exercise-2-token-counting)
4. [Exercise 3: Temperature Experiments](#4-exercise-3-temperature-experiments)
5. [Exercise 4: Parameter Comparison](#5-exercise-4-parameter-comparison)
6. [Exercise 5: Streaming Implementation](#6-exercise-5-streaming-implementation)
7. [Exercise 6: Cost Calculator](#7-exercise-6-cost-calculator)
8. [Exercise 7: Build SimpleChatbot](#8-exercise-7-build-simplechatbot)
9. [Capstone: SupportGenie v0.1](#9-capstone-supportgenie-v01)
10. [Challenges & Extensions](#10-challenges--extensions)

---

## 1. Setup Your Environment

### 🛠️ Step 1.1: Install Required Packages

Open your terminal and run:

```bash
pip install openai anthropic google-generativeai tiktoken python-dotenv
```

**Verify installation:**
```bash
python -c "import openai, anthropic, tiktoken; print('✅ All packages installed!')"
```

---

### 🔑 Step 1.2: Configure API Keys

Create a `.env` file in your project directory:

```bash
# .env file
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here
```

**🚨 CRITICAL:** Add `.env` to your `.gitignore`:
```bash
echo ".env" >> .gitignore
```

---

### ✅ Step 1.3: Test Your Setup

Create `test_setup.py`:

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Check API keys
openai_key = os.getenv('OPENAI_API_KEY')
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
google_key = os.getenv('GOOGLE_API_KEY')

print("API Key Status:")
print(f"✅ OpenAI: {'Loaded' if openai_key else '❌ Missing'}")
print(f"✅ Anthropic: {'Loaded' if anthropic_key else '❌ Missing'}")
print(f"✅ Google: {'Loaded' if google_key else '❌ Missing'}")

# You need at least ONE key to proceed
if openai_key or anthropic_key or google_key:
    print("\n🎉 Setup complete! You're ready to code.")
else:
    print("\n⚠️ No API keys found. Please add at least one key to .env file.")
```

**Run it:**
```bash
python test_setup.py
```

**Expected output:**
```
API Key Status:
✅ OpenAI: Loaded
✅ Anthropic: Loaded
✅ Google: Loaded

🎉 Setup complete! You're ready to code.
```

---

## 2. Exercise 1: First API Call

**Duration:** 10 minutes
**Objective:** Make your first successful API call to an LLM

---

### 🎯 Task 1.1: OpenAI API Call

Create `exercise1_openai.py`:

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

# Extract and display answer
answer = response.choices[0].message.content
print(f"Answer: {answer}")

# Show token usage
print(f"\nToken Usage:")
print(f"  Prompt: {response.usage.prompt_tokens}")
print(f"  Completion: {response.usage.completion_tokens}")
print(f"  Total: {response.usage.total_tokens}")
```

**Run it:**
```bash
python exercise1_openai.py
```

**Expected output:**
```
Answer: The capital of France is Paris.

Token Usage:
  Prompt: 20
  Completion: 8
  Total: 28
```

---

### 🎯 Task 1.2: Claude API Call

Create `exercise1_claude.py`:

```python
from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

message = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(f"Answer: {message.content[0].text}")
print(f"\nToken Usage:")
print(f"  Input: {message.usage.input_tokens}")
print(f"  Output: {message.usage.output_tokens}")
```

**Run it:**
```bash
python exercise1_claude.py
```

---

### 🎯 Task 1.3: Gemini API Call

Create `exercise1_gemini.py`:

```python
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content('What is the capital of France?')

print(f"Answer: {response.text}")
```

**Run it:**
```bash
python exercise1_gemini.py
```

---

### ✅ Checkpoint

**Verify:** You should have successfully called at least ONE LLM API.

**Troubleshooting:**
- **401 Error:** Check your API key is correct
- **Import Error:** Reinstall packages with `pip install`
- **No output:** Check your internet connection

---

## 3. Exercise 2: Token Counting

**Duration:** 10 minutes
**Objective:** Understand how tokens work in practice

---

### 🎯 Task 2.1: Basic Token Counter

Create `exercise2_tokens.py`:

```python
import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text for a specific model"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

# Test with various texts
test_texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog",
    "OpenAI GPT-4 is a large language model",
    "Machine learning is transforming artificial intelligence",
]

print("Token Analysis:")
print("=" * 70)

for text in test_texts:
    token_count = count_tokens(text)
    char_count = len(text)
    ratio = char_count / token_count if token_count > 0 else 0

    print(f"\nText: '{text}'")
    print(f"  Tokens: {token_count}")
    print(f"  Characters: {char_count}")
    print(f"  Ratio: {ratio:.2f} chars/token")
```

**Run it:**
```bash
python exercise2_tokens.py
```

---

### 🎯 Task 2.2: Token Breakdown Visualizer

Add this to `exercise2_tokens.py`:

```python
def visualize_tokens(text, model="gpt-3.5-turbo"):
    """Show how text is tokenized"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    token_strings = [encoding.decode([token]) for token in tokens]

    print(f"\nText: '{text}'")
    print(f"Tokens ({len(tokens)}): {token_strings}")
    print("Visual breakdown:")
    for i, token_str in enumerate(token_strings, 1):
        print(f"  {i}. '{token_str}'")

# Test it
visualize_tokens("ChatGPT is amazing")
visualize_tokens("Hello, world!")
visualize_tokens("The weather today is sunny")
```

---

### 🎯 Task 2.3: Your Turn!

**Challenge:** Count tokens for this paragraph:

```
"Large Language Models have revolutionized natural language processing.
They can understand context, generate creative content, and assist with
complex reasoning tasks."
```

**Questions:**
1. How many tokens?
2. How many characters?
3. What's the chars/token ratio?

<details>
<summary>Click for solution</summary>

```python
text = """Large Language Models have revolutionized natural language processing.
They can understand context, generate creative content, and assist with
complex reasoning tasks."""

tokens = count_tokens(text)
chars = len(text)
ratio = chars / tokens

print(f"Tokens: {tokens}")  # ~35-40 tokens
print(f"Characters: {chars}")  # ~180 characters
print(f"Ratio: {ratio:.2f}")  # ~4.5 chars/token
```

</details>

---

## 4. Exercise 3: Temperature Experiments

**Duration:** 15 minutes
**Objective:** See how temperature affects output

---

### 🎯 Task 3.1: Temperature Comparison

Create `exercise3_temperature.py`:

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def test_temperature(prompt, temperature, runs=5):
    """Test a prompt at specific temperature multiple times"""
    print(f"\n{'='*60}")
    print(f"Temperature: {temperature}")
    print('='*60)

    responses = []
    for i in range(runs):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=10
        )
        answer = response.choices[0].message.content
        responses.append(answer)
        print(f"  {i+1}. {answer}")

    # Check uniqueness
    unique = len(set(responses))
    print(f"\nUnique responses: {unique}/{runs}")
    return responses

# Test with different temperatures
prompt = "Name a color"

test_temperature(prompt, temperature=0.0)
test_temperature(prompt, temperature=0.7)
test_temperature(prompt, temperature=1.5)
```

**Run it:**
```bash
python exercise3_temperature.py
```

**Expected behavior:**
- **Temp 0.0:** Same response every time (e.g., all "Blue")
- **Temp 0.7:** Some variation (Blue, Red, Green, Yellow)
- **Temp 1.5:** High variation (Cerulean, Magenta, Chartreuse)

---

### 🎯 Task 3.2: Find Your Sweet Spot

**Challenge:** Test these prompts with different temperatures:

```python
prompts = [
    "What is 2+2?",  # Should use low temp
    "Write a creative opening line for a story",  # Should use high temp
    "Translate 'Hello' to Spanish",  # Should use low temp
    "Generate a unique business name for a coffee shop"  # Should use high temp
]
```

**Questions:**
1. Which prompts work best at temp=0?
2. Which need higher temperature?
3. Why?

<details>
<summary>Click for answer</summary>

- **Math (2+2):** temp=0 (factual, one correct answer)
- **Creative story:** temp=1.2-1.5 (want variety)
- **Translation:** temp=0-0.3 (factual, consistent)
- **Business name:** temp=0.9-1.3 (creative but coherent)

</details>

---

## 5. Exercise 4: Parameter Comparison

**Duration:** 15 minutes
**Objective:** Compare temperature, top_p, and max_tokens

---

### 🎯 Task 4.1: max_tokens Effects

Create `exercise4_parameters.py`:

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def test_max_tokens(prompt, max_tokens_values):
    """Test different max_tokens values"""
    for max_tok in max_tokens_values:
        print(f"\n{'='*60}")
        print(f"max_tokens: {max_tok}")
        print('='*60)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tok
        )

        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        actual_tokens = response.usage.completion_tokens

        print(f"Finish reason: {finish_reason}")
        print(f"Actual tokens: {actual_tokens}")
        print(f"Response: {content}")

        if finish_reason == "length":
            print("⚠️ Response was truncated!")

# Test it
prompt = "Explain what machine learning is"
test_max_tokens(prompt, [20, 50, 150])
```

---

### 🎯 Task 4.2: top_p Experiment

Add this to `exercise4_parameters.py`:

```python
def test_top_p(prompt, top_p_values, runs=3):
    """Test different top_p values"""
    for top_p in top_p_values:
        print(f"\n{'='*60}")
        print(f"top_p: {top_p}")
        print('='*60)

        for i in range(runs):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                top_p=top_p,
                max_tokens=20
            )
            print(f"  {i+1}. {response.choices[0].message.content}")

# Test it
test_top_p("Complete this: The sky is", [0.1, 0.5, 0.9])
```

---

### ✅ Checkpoint

**Verify you understand:**
1. What happens when max_tokens is too low?
2. How does top_p=0.1 differ from top_p=0.9?
3. When would you use temperature vs top_p?

---

## 6. Exercise 5: Streaming Implementation

**Duration:** 15 minutes
**Objective:** Implement real-time streaming responses

---

### 🎯 Task 5.1: Basic Streaming

Create `exercise5_streaming.py`:

```python
from openai import OpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def stream_response(prompt):
    """Stream a response token by token"""
    print("Assistant: ", end="", flush=True)

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=200
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    print()  # New line at end
    return full_response

# Test it
print("Testing streaming...\n")
stream_response("Tell me a short joke about programming")
```

**Run it:**
```bash
python exercise5_streaming.py
```

**You should see:** Text appearing word-by-word in real-time!

---

### 🎯 Task 5.2: Streaming with Token Counter

**Challenge:** Modify the streaming function to count tokens as they arrive.

<details>
<summary>Click for solution</summary>

```python
def stream_with_counter(prompt):
    """Stream and count tokens"""
    print("Assistant: ", end="", flush=True)

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=200
    )

    full_response = ""
    chunk_count = 0

    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
            chunk_count += 1

    print(f"\n\n[Received {chunk_count} chunks]")
    return full_response

stream_with_counter("Explain Python in one paragraph")
```

</details>

---

### 🎯 Task 5.3: Compare Streaming vs Non-Streaming

**Challenge:** Time both approaches and compare.

<details>
<summary>Click for solution</summary>

```python
import time

def non_streaming(prompt):
    """Regular API call"""
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    elapsed = time.time() - start
    print(f"Non-streaming took {elapsed:.2f}s")
    return response.choices[0].message.content

def streaming(prompt):
    """Streaming API call"""
    start = time.time()
    first_token_time = None

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=200
    )

    full = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time() - start
            full += chunk.choices[0].delta.content

    elapsed = time.time() - start
    print(f"Streaming: First token in {first_token_time:.2f}s, Total {elapsed:.2f}s")
    return full

prompt = "Explain machine learning in detail"
non_streaming(prompt)
streaming(prompt)
```

</details>

---

## 7. Exercise 6: Cost Calculator

**Duration:** 10 minutes
**Objective:** Calculate API costs accurately

---

### 🎯 Task 6.1: Build Cost Calculator

Create `exercise6_costs.py`:

```python
import tiktoken

def calculate_cost(input_text, output_text, model="gpt-3.5-turbo"):
    """Calculate the cost of an API call"""

    # Pricing per 1M tokens (as of Dec 2024)
    pricing = {
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
    }

    if model not in pricing:
        raise ValueError(f"Model {model} not in pricing table")

    # Count tokens
    encoding = tiktoken.encoding_for_model(model)
    input_tokens = len(encoding.encode(input_text))
    output_tokens = len(encoding.encode(output_text))

    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * pricing[model]["input"]
    output_cost = (output_tokens / 1_000_000) * pricing[model]["output"]
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

# Test it
prompt = "Explain quantum computing in simple terms"
response = "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, enabling parallel processing of information. Unlike classical bits that are either 0 or 1, qubits can be both at once through superposition."

result = calculate_cost(prompt, response)

print("Cost Analysis:")
print("=" * 50)
print(f"Model: {result['model']}")
print(f"\nTokens:")
print(f"  Input:  {result['input_tokens']}")
print(f"  Output: {result['output_tokens']}")
print(f"  Total:  {result['total_tokens']}")
print(f"\nCosts:")
print(f"  Input:  ${result['input_cost']:.6f}")
print(f"  Output: ${result['output_cost']:.6f}")
print(f"  Total:  ${result['total_cost']:.6f}")
```

**Run it:**
```bash
python exercise6_costs.py
```

---

### 🎯 Task 6.2: Compare Model Costs

**Challenge:** Calculate costs for the same conversation across different models.

<details>
<summary>Click for solution</summary>

```python
models = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"]

print("\nModel Cost Comparison:")
print("=" * 60)

for model in models:
    result = calculate_cost(prompt, response, model)
    print(f"\n{model}:")
    print(f"  Total cost: ${result['total_cost']:.6f}")
    print(f"  For 1,000 requests: ${result['total_cost'] * 1000:.2f}")
    print(f"  For 10,000 requests: ${result['total_cost'] * 10000:.2f}")
```

</details>

---

## 8. Exercise 7: Build SimpleChatbot

**Duration:** 20 minutes
**Objective:** Create a reusable chatbot class

---

### 🎯 Task 7.1: Basic Chatbot Class

Create `exercise7_chatbot.py`:

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
        """Set the AI's personality and behavior"""
        self.conversation_history = [
            {"role": "system", "content": message}
        ]

    def chat(self, user_message, stream=False):
        """Send a message and get response"""
        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Get response
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
        self.total_cost += self._calculate_cost(response.usage)

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

    def _calculate_cost(self, usage):
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

    bot.set_system_message("You are a helpful assistant. Keep responses under 50 words.")

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

**Run it:**
```bash
python exercise7_chatbot.py
```

---

### ✅ Checkpoint

Test your chatbot with these conversations:
1. Multi-turn conversation (3+ messages)
2. Check statistics after
3. Clear history and start new conversation

---

## 9. Capstone: SupportGenie v0.1

**Duration:** 30 minutes
**Objective:** Build a production-ready customer support chatbot

---

### 🎯 Requirements

Build **SupportGenie** with:
- ✅ Professional, empathetic customer service tone
- ✅ Streaming responses for better UX
- ✅ Token and cost tracking
- ✅ Session statistics
- ✅ Response format guidelines
- ✅ Escalation to human agents when needed

---

### 🎯 Implementation

Create `capstone_supportgenie.py`:

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class SupportGenieV1:
    """
    SupportGenie - AI Customer Support Assistant
    Version 0.1: Professional chatbot with best practices
    """

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"
        self.conversation_history = []
        self.total_tokens = 0
        self.total_cost = 0.0

        # Set professional system message
        self.set_system_message("""
You are SupportGenie, an AI customer support assistant for TechStore.

Guidelines:
- Be professional, empathetic, and helpful
- Keep responses under 100 words unless more detail is needed
- If you don't know something, be honest and offer to escalate
- Always offer to escalate to a human agent if the issue is complex
- Use a friendly but professional tone
- Acknowledge emotions when appropriate

Response Format:
1. Acknowledge the customer's concern
2. Provide helpful information or next steps
3. Ask if there's anything else you can help with
        """)

    def set_system_message(self, message):
        """Set the system prompt"""
        self.conversation_history = [
            {"role": "system", "content": message}
        ]

    def chat(self, user_message):
        """Send a message and get streamed response"""
        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Stream response
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

    def get_stats(self):
        """Get session statistics"""
        user_messages = len([m for m in self.conversation_history if m['role'] == 'user'])
        return {
            "messages": user_messages,
            "total_cost": f"${self.total_cost:.4f}"
        }

    def welcome(self):
        """Display welcome message"""
        print("=" * 60)
        print("    SupportGenie v0.1 - AI Customer Support")
        print("=" * 60)
        print("\n👋 Hello! I'm SupportGenie, your AI support assistant.")
        print("How can I help you today?\n")
        print("💡 Tips:")
        print("  - Type 'quit' to exit")
        print("  - Type 'stats' for session info")
        print("  - Be as specific as possible for best results\n")

    def run(self):
        """Run the interactive chatbot"""
        self.welcome()

        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n✅ Thank you for using SupportGenie!")
                stats = self.get_stats()
                print(f"📊 Session Stats: {stats['messages']} messages exchanged")
                break

            if user_input.lower() == 'stats':
                stats = self.get_stats()
                print(f"\n📊 Current Session:")
                print(f"  Messages: {stats['messages']}")
                print(f"  Estimated cost: {stats['total_cost']}\n")
                continue

            print("\nSupportGenie: ", end="")
            self.chat(user_input)
            print()


# Run it!
if __name__ == "__main__":
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        exit(1)

    bot = SupportGenieV1(api_key=api_key)
    bot.run()
```

**Run it:**
```bash
python capstone_supportgenie.py
```

---

### 🎯 Test Scenarios

Try these customer support scenarios:

1. **Order Inquiry:**
   ```
   You: My order hasn't arrived yet and it's been 2 weeks
   ```

2. **Product Question:**
   ```
   You: Does the XPS 15 laptop come with a warranty?
   ```

3. **Return Request:**
   ```
   You: I want to return my headphones, they don't fit
   ```

4. **Technical Issue:**
   ```
   You: My laptop won't turn on after the update
   ```

5. **Edge Case:**
   ```
   You: asdkfjasldkfj
   ```

---

### ✅ Success Criteria

Your SupportGenie should:
- ✅ Respond professionally and empathetically
- ✅ Stream responses smoothly
- ✅ Offer to escalate when appropriate
- ✅ Handle unclear questions gracefully
- ✅ Keep responses concise but helpful
- ✅ Track session statistics

---

## 10. Challenges & Extensions

**Want to go further? Try these challenges!**

---

### 🏆 Challenge 1: Cost Warnings

Add automatic alerts when session cost exceeds $0.10:

```python
def check_cost_warning(self):
    """Alert if cost is getting high"""
    if self.total_cost > 0.10:
        print(f"\n⚠️ Warning: Session cost is ${self.total_cost:.4f}")
```

---

### 🏆 Challenge 2: Response Time Tracking

Track and display how long each response takes:

```python
import time

# In chat method:
start_time = time.time()
# ... generate response ...
elapsed = time.time() - start_time
print(f"\n⏱️ Response time: {elapsed:.2f}s")
```

---

### 🏆 Challenge 3: Conversation Export

Save conversation history to a file:

```python
def export_conversation(self, filename="conversation.txt"):
    """Export conversation to file"""
    with open(filename, 'w') as f:
        for msg in self.conversation_history:
            if msg['role'] != 'system':
                f.write(f"{msg['role'].upper()}: {msg['content']}\n\n")
    print(f"✅ Conversation saved to {filename}")
```

---

### 🏆 Challenge 4: Multi-Language Support

Add automatic language detection and response:

```python
# Detect language in system message
"""
Detect the user's language and respond in the same language.
If the user writes in Spanish, respond in Spanish.
If the user writes in French, respond in French.
"""
```

---

### 🏆 Challenge 5: Sentiment Analysis

Track customer sentiment (happy, frustrated, neutral):

```python
def analyze_sentiment(self, user_message):
    """Quick sentiment check"""
    frustrated_words = ['angry', 'frustrated', 'terrible', 'awful']
    happy_words = ['great', 'thanks', 'perfect', 'excellent']

    msg_lower = user_message.lower()
    if any(word in msg_lower for word in frustrated_words):
        return "frustrated"
    elif any(word in msg_lower for word in happy_words):
        return "happy"
    return "neutral"
```

---

## 🎉 Congratulations!

You've completed Lab 1! You now have:

✅ **Practical Skills:**
- Made API calls to OpenAI, Claude, and Gemini
- Counted and visualized tokens
- Experimented with temperature and parameters
- Implemented streaming responses
- Calculated API costs
- Built a complete chatbot

✅ **Production Code:**
- SimpleChatbot class (reusable)
- SupportGenie v0.1 (production-ready)
- Cost calculator (essential tool)
- Token counter (debugging helper)

✅ **Real Understanding:**
- How LLMs work (prediction engines)
- Why tokens matter (costs, limits, performance)
- When to use different parameters
- How to optimize for cost and quality

---

## 🚀 Next Steps

**Ready for more advanced topics?**

👉 **[Lab 2: Prompt Engineering →](../Lab2-Prompt-Engineering/learning.md)**

Learn to write prompts that get exceptional results every time!

---

## 📚 Additional Practice

### Optional Exercises:

1. **Build a Translation Bot**
   - Use temp=0 for consistency
   - Track costs per translation
   - Support 3+ languages

2. **Create a Story Generator**
   - Use temp=1.2-1.5 for creativity
   - Implement streaming for drama
   - Save stories to files

3. **Make a Quiz Bot**
   - Ask random trivia questions
   - Track score across session
   - Use temp=0 for consistent questions

---

## 🤝 Need Help?

**Common Issues:**

**API Error 401:**
- Check your API key in `.env`
- Make sure `.env` is in the same directory
- Verify key has credits

**Slow Responses:**
- Check your internet connection
- Try a smaller max_tokens
- Consider using GPT-3.5 instead of GPT-4

**Import Errors:**
- Reinstall packages: `pip install --upgrade openai anthropic`
- Check Python version (need 3.8+)

---

**End of Lab 1** ✅

**Estimated completion time:** 60-90 minutes
**Actual time taken:** _____ minutes

**Self-Assessment:**
- [ ] I can make API calls to at least one LLM
- [ ] I understand how tokens work
- [ ] I can use temperature and max_tokens effectively
- [ ] I implemented streaming successfully
- [ ] I built SupportGenie v0.1
- [ ] I'm ready for Lab 2!
