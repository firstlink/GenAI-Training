# Lab 1: LLM Fundamentals & API Usage
## 📚 Learning Material

> **Purpose:** Understand the theory and concepts before you code

---

## 📋 Overview

| Property | Value |
|----------|-------|
| **Duration** | 30 minutes (reading) |
| **Difficulty** | Beginner |
| **Prerequisites** | Basic Python knowledge |
| **Next Step** | [Hands-On Lab →](lab.md) |

---

## 📖 Table of Contents

1. [Introduction](#1-introduction)
2. [How LLMs Work](#2-how-llms-work)
3. [Understanding Tokens](#3-understanding-tokens)
4. [API Basics](#4-api-basics)
5. [Key Parameters Explained](#5-key-parameters-explained)
6. [Streaming Responses](#6-streaming-responses)
7. [Cost Calculation](#7-cost-calculation)
8. [Review & Key Takeaways](#8-review--key-takeaways)

---

## 1. Introduction

### Welcome to the World of LLMs! 🚀

Large Language Models (LLMs) are transforming every industry:
- 💼 **Customer support** - Automated, intelligent responses
- 📊 **Data analysis** - Natural language queries
- 🤖 **Personal assistants** - Context-aware help
- 📝 **Content generation** - Articles, emails, code

By understanding how LLMs work and how to use their APIs, you'll be able to build production-ready AI applications.

---

### What You'll Learn in This Module

- 🎯 How Large Language Models actually work
- 🎯 What tokens are and why they matter
- 🎯 How to make API calls to different providers
- 🎯 Understanding key parameters that control behavior
- 🎯 When and how to use streaming
- 🎯 How to calculate and optimize costs

---

## 2. How LLMs Work

### What is a Large Language Model?

🧠 **Simple Definition:**
> An LLM is a neural network trained on massive amounts of text to predict the most likely next word (or token) in a sequence.

Think of it as an **incredibly sophisticated autocomplete system**.

---

### The Training Process

LLMs go through three main training stages:

```
┌────────────────────────────────────────────────────┐
│  STAGE 1: PRE-TRAINING                             │
│  ───────────────────────────────────────────────   │
│                                                     │
│  Input:  Billions of words from books, websites,   │
│          articles, code repositories                │
│                                                     │
│  Task:   Predict the next word in a sentence       │
│                                                     │
│  Result: Model learns grammar, facts, patterns,    │
│          and reasoning from vast amounts of data    │
│                                                     │
│  Example: "The capital of France is ___"           │
│           Model learns → "Paris"                    │
└────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────┐
│  STAGE 2: FINE-TUNING                              │
│  ───────────────────────────────────────────────   │
│                                                     │
│  Input:  High-quality instruction-response pairs   │
│          curated by humans                          │
│                                                     │
│  Task:   Learn to follow instructions accurately   │
│          and generate helpful responses             │
│                                                     │
│  Result: Model becomes a helpful assistant that    │
│          can follow complex instructions            │
│                                                     │
│  Example: "Explain photosynthesis simply"          │
│           Model generates clear explanation         │
└────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────┐
│  STAGE 3: ALIGNMENT (RLHF)                         │
│  ───────────────────────────────────────────────   │
│                                                     │
│  Input:  Human feedback on model responses         │
│          (thumbs up/down, rankings)                 │
│                                                     │
│  Task:   Be helpful, harmless, and honest          │
│          Avoid harmful or biased outputs            │
│                                                     │
│  Result: Safe, reliable, and aligned AI assistant  │
│          that follows ethical guidelines            │
│                                                     │
│  Example: Refuses harmful requests, admits          │
│           uncertainty, avoids making things up      │
└────────────────────────────────────────────────────┘
```

---

### How Text Generation Actually Works

Let's see what happens when you ask an LLM to complete a sentence:

**User Input:** "The weather today is"

```
Step 1: TOKENIZATION
────────────────────
Input text → ["The", "weather", "today", "is"]

Step 2: PROBABILITY CALCULATION
────────────────────────────────
Model calculates probability for EVERY possible next token:
  "sunny"      → 30%  ████████████████
  "nice"       → 25%  █████████████
  "cloudy"     → 20%  ██████████
  "rainy"      → 15%  ████████
  "beautiful"  → 5%   ███
  "terrible"   → 3%   ██
  ... (thousands more with tiny probabilities)

Step 3: SAMPLING (based on temperature)
────────────────────────────────────────
- Temperature = 0.0  → Always pick "sunny" (most likely)
- Temperature = 0.7  → Sample from top choices with variation
- Temperature = 2.0  → Consider many options, very creative

Step 4: ADD TOKEN & REPEAT
───────────────────────────
Selected token "sunny" is added to sequence
New sequence: ["The", "weather", "today", "is", "sunny"]
Process repeats for next token until done
```

---

### 💡 Critical Insight

**LLMs don't truly "know" facts or "understand" meaning.**

They're **statistical prediction engines** that:
- ✅ Generate highly convincing text based on patterns
- ✅ Can solve complex problems through learned patterns
- ✅ Produce creative and coherent responses
- ❌ Sometimes "hallucinate" (confidently state false information)
- ❌ Don't have real-time knowledge (training data has a cutoff)
- ❌ Can't truly reason like humans (simulate reasoning patterns)

This is why we need techniques like:
- **RAG (Retrieval-Augmented Generation)** - Ground responses in facts
- **Prompt engineering** - Guide the model effectively
- **Guardrails** - Prevent harmful outputs

---

## 3. Understanding Tokens

### What Are Tokens?

**Tokens** are the fundamental units that LLMs process.

🎯 **Not characters:** "Hello" could be 1 token, not 5
🎯 **Not words:** "ChatGPT" is often 2+ tokens
🎯 **Subword units:** Somewhere between characters and words

---

### The Token Economy

**Rule of Thumb (English):**
```
1 token ≈ 4 characters
1 token ≈ ¾ of a word
100 tokens ≈ 75 words
1,000 tokens ≈ 750 words
```

---

### Real Examples

```
┌─────────────────────────────────────────────┐
│  TEXT: "Hello, world!"                      │
│  TOKENS: ["Hello", ",", " world", "!"]      │
│  COUNT: 4 tokens                             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  TEXT: "ChatGPT is amazing"                 │
│  TOKENS: ["Chat", "G", "PT", " is",         │
│           " amazing"]                        │
│  COUNT: 5 tokens                             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  TEXT: "OpenAI GPT-4"                       │
│  TOKENS: ["Open", "AI", " G", "PT", "-4"]  │
│  COUNT: 5 tokens                             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  TEXT: "The quick brown fox"                │
│  TOKENS: ["The", " quick", " brown", " fox"]│
│  COUNT: 4 tokens                             │
└─────────────────────────────────────────────┘
```

---

### Why Tokens Matter

#### 1. **Context Window Limits**

Every model has a maximum token limit (input + output):

| Model | Context Window | Practical Limit |
|-------|---------------|-----------------|
| GPT-3.5-turbo | 4,096 tokens | ~3,000 words |
| GPT-3.5-turbo-16k | 16,384 tokens | ~12,000 words |
| GPT-4 | 8,192 tokens | ~6,000 words |
| GPT-4-turbo | 128,000 tokens | ~96,000 words |
| Claude 3 | 200,000 tokens | ~150,000 words |
| Gemini Pro | 32,768 tokens | ~24,000 words |

**What happens when you exceed the limit?**
```
❌ Error: "This model's maximum context length is 4096 tokens"
```

---

#### 2. **Cost Per Token**

APIs charge based on token usage:

```
Example Pricing (GPT-3.5-turbo):
─────────────────────────────────
Input:  $0.50 per 1M tokens
Output: $1.50 per 1M tokens

Sample Conversation:
────────────────────
User prompt: "Explain photosynthesis" (3 tokens)
System message: 50 tokens
Model response: 150 tokens

Cost calculation:
Input tokens:  53 tokens × $0.50/1M = $0.0000265
Output tokens: 150 tokens × $1.50/1M = $0.0002250
Total cost: $0.0002515 (~$0.00025)

For 1,000 similar requests:
Total cost ≈ $0.25
```

---

#### 3. **Performance Impact**

```
Token Count → Processing Time

Input: 100 tokens, Output: 50 tokens
Response time: ~0.5 seconds

Input: 1,000 tokens, Output: 500 tokens
Response time: ~3-5 seconds

Input: 10,000 tokens, Output: 2,000 tokens
Response time: ~10-20 seconds
```

**More tokens = Slower responses + Higher costs**

---

### Token Encoding Differences

Different models use different tokenization:

```
Text: "Hello, 世界!" (Hello, World! in Chinese/Japanese)

GPT (cl100k_base encoding):
  ["Hello", ",", " ", "世", "界", "!"]
  6 tokens

Claude (similar encoding):
  ["Hello", ",", " 世界", "!"]
  4 tokens

Why different? Different tokenizers handle:
- Non-English languages differently
- Special characters uniquely
- Common phrases as single tokens
```

**Lesson:** Always count tokens for YOUR specific model!

---

## 4. API Basics

### Understanding API Providers

Three major LLM API providers:

```
┌──────────────────────────────────────────────────┐
│  OPENAI (ChatGPT)                                │
├──────────────────────────────────────────────────┤
│  Models: GPT-3.5-turbo, GPT-4, GPT-4-turbo      │
│  Strengths: Fast, reliable, widely adopted       │
│  Best for: General purpose, production apps      │
│  API style: OpenAI standard                      │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│  ANTHROPIC (Claude)                              │
├──────────────────────────────────────────────────┤
│  Models: Claude 3 Haiku, Sonnet, Opus           │
│  Strengths: Long context (200K), nuanced         │
│  Best for: Complex reasoning, long documents     │
│  API style: Similar but slightly different       │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│  GOOGLE (Gemini)                                 │
├──────────────────────────────────────────────────┤
│  Models: Gemini Pro, Gemini Ultra               │
│  Strengths: Multimodal, generous free tier       │
│  Best for: Experimentation, cost-conscious       │
│  API style: Google's generative AI API          │
└──────────────────────────────────────────────────┘
```

---

### Message Structure

All chat-based LLM APIs use a **message array** format:

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "What is the capital of France?"
    },
    {
        "role": "assistant",
        "content": "The capital of France is Paris."
    },
    {
        "role": "user",
        "content": "What about Germany?"
    }
]
```

---

### Role Explanations

| Role | Purpose | When to Use |
|------|---------|-------------|
| **system** | Sets AI behavior, personality, constraints | First message, defines how AI should act |
| **user** | Human messages/questions | Every user input |
| **assistant** | AI responses | Conversation history, few-shot examples |

---

### System Message Best Practices

```python
# ❌ WEAK System Message
"You are helpful."

# ✅ STRONG System Message
"""
You are a professional customer support agent for TechStore.

Guidelines:
- Be empathetic and patient
- Keep responses under 100 words
- If you don't know, say so and offer to escalate
- Never make up product information
- Use a friendly but professional tone

Response format:
1. Acknowledge the customer's concern
2. Provide solution or next steps
3. Ask if there's anything else needed
"""
```

---

## 5. Key Parameters Explained

### Temperature (0.0 - 2.0)

**Controls randomness and creativity** in responses.

```
┌────────────────────────────────────────────────┐
│  HOW TEMPERATURE WORKS                         │
├────────────────────────────────────────────────┤
│                                                 │
│  Original token probabilities:                 │
│    Token A: 60%  ████████████████              │
│    Token B: 25%  ███████                       │
│    Token C: 10%  ███                           │
│    Token D: 5%   ██                            │
│                                                 │
│  Temperature = 0.1 (Very focused):             │
│    Token A: 95%  ███████████████████████       │
│    Token B: 4%   █                             │
│    Token C: 0.8% ▌                             │
│    Token D: 0.2% ▌                             │
│                                                 │
│  Temperature = 1.0 (Balanced):                 │
│    Token A: 60%  ████████████████              │
│    Token B: 25%  ███████                       │
│    Token C: 10%  ███                           │
│    Token D: 5%   ██                            │
│                                                 │
│  Temperature = 2.0 (Very creative):            │
│    Token A: 40%  ██████████                    │
│    Token B: 30%  ████████                      │
│    Token C: 20%  █████                         │
│    Token D: 10%  ███                           │
│                                                 │
└────────────────────────────────────────────────┘
```

---

### Temperature Use Cases

| Temperature | Behavior | Best For | Example Use |
|-------------|----------|----------|-------------|
| **0.0** | Deterministic, always same output | Exact consistency needed | Data extraction, classification |
| **0.0-0.3** | Very focused, minimal variation | Factual tasks | Summarization, translation |
| **0.3-0.7** | Slight creativity, natural | General purpose | Chatbots, Q&A systems |
| **0.7-1.0** | Balanced creativity | Content generation | Email drafting, explanations |
| **1.0-1.5** | High creativity | Creative tasks | Story writing, brainstorming |
| **1.5-2.0** | Maximum creativity (may ramble) | Experimental | Poetry, unusual ideas |

---

### max_tokens

**Limits the maximum length** of the generated response.

```
┌────────────────────────────────────────────────┐
│  UNDERSTANDING max_tokens                      │
├────────────────────────────────────────────────┤
│                                                 │
│  Formula:                                      │
│  Input tokens + max_tokens ≤ Context Window   │
│                                                 │
│  Example with GPT-3.5-turbo (4K context):     │
│                                                 │
│  Scenario 1: ✅ Success                        │
│    Input: 500 tokens                           │
│    max_tokens: 1,000                           │
│    Total: 1,500 tokens < 4,096 ✅             │
│                                                 │
│  Scenario 2: ❌ Error                          │
│    Input: 3,800 tokens                         │
│    max_tokens: 1,000                           │
│    Total: 4,800 tokens > 4,096 ❌             │
│    Error: "exceeds maximum context length"     │
│                                                 │
│  Scenario 3: ⚠️ Truncation                    │
│    Input: 100 tokens                           │
│    max_tokens: 50 (very low!)                  │
│    Response: "Quantum computing uses qubi..."   │
│    (cut off mid-sentence)                      │
│                                                 │
└────────────────────────────────────────────────┘
```

---

### max_tokens Recommendations

| Use Case | Recommended max_tokens | Reason |
|----------|------------------------|--------|
| Simple Q&A | 50-100 | Short, concise answers |
| Chatbot responses | 150-300 | Conversational length |
| Explanations | 200-500 | Detailed but focused |
| Code generation | 500-1,000 | Complete functions |
| Long-form content | 1,000-2,000 | Articles, essays |
| Document summaries | 100-200 | Concise summaries |

---

### top_p (Nucleus Sampling)

**Alternative to temperature** - filters tokens by cumulative probability.

```
┌────────────────────────────────────────────────┐
│  HOW top_p WORKS (Nucleus Sampling)            │
├────────────────────────────────────────────────┤
│                                                 │
│  Token probabilities (sorted):                 │
│    Token A: 40%  ████████████████              │
│    Token B: 30%  ████████████                  │
│    Token C: 15%  ██████                        │
│    Token D: 10%  ████                          │
│    Token E: 3%   █                             │
│    Token F: 2%   █                             │
│                                                 │
│  top_p = 0.5:                                  │
│    Keep A and B only (40% + 30% = 70% ≥ 50%)  │
│    Very focused, predictable                   │
│                                                 │
│  top_p = 0.7:                                  │
│    Keep A, B, C (40%+30%+15% = 85% ≥ 70%)     │
│    Balanced variety                            │
│                                                 │
│  top_p = 0.9:                                  │
│    Keep A, B, C, D (95% ≥ 90%)                │
│    Good variety, filters outliers              │
│                                                 │
│  top_p = 1.0:                                  │
│    Keep all tokens                             │
│    No filtering                                │
│                                                 │
└────────────────────────────────────────────────┘
```

---

### top_p vs temperature

| Aspect | top_p | temperature |
|--------|-------|-------------|
| **Mechanism** | Filters tokens by probability mass | Reshapes probability distribution |
| **Adaptivity** | Adapts to each context | Same effect everywhere |
| **Quality control** | Better at filtering bad tokens | Can include very unlikely tokens |
| **Common values** | 0.9 (production standard) | 0.7 (general use) |
| **Best practice** | Use OR temperature, not both! | Use OR top_p, not both! |

**🎯 Production Recommendation:** Use `top_p=0.9` for most applications.

---

### top_k (Available in some APIs)

**Limits consideration to top K most likely tokens.**

```
Available in:
✅ Google Gemini (default: 40)
✅ Cohere
❌ OpenAI (use top_p instead)
❌ Anthropic Claude (use top_p instead)

Example:
  top_k = 1  → Always pick most likely (deterministic)
  top_k = 40 → Consider top 40 tokens (balanced)
  top_k = 100 → Consider top 100 (very creative)

Problem: Fixed size doesn't adapt to context
Solution: top_p is usually better!
```

---

## 6. Streaming Responses

### Why Streaming Matters

```
┌────────────────────────────────────────────────┐
│  WITHOUT STREAMING (Bad UX)                    │
├────────────────────────────────────────────────┤
│                                                 │
│  User: "Explain machine learning"              │
│                                                 │
│  [Loading for 8 seconds...]                    │
│  ⏳ User sees nothing...                        │
│  ⏳ User waits...                               │
│  ⏳ User gets frustrated...                     │
│                                                 │
│  Assistant: [Full response appears suddenly]    │
│                                                 │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│  WITH STREAMING (Good UX)                      │
├────────────────────────────────────────────────┤
│                                                 │
│  User: "Explain machine learning"              │
│                                                 │
│  Assistant: Machine                            │
│  Assistant: Machine learning                   │
│  Assistant: Machine learning is               │
│  Assistant: Machine learning is a              │
│  Assistant: Machine learning is a way...       │
│                                                 │
│  ✅ User sees progress immediately              │
│  ✅ Feels faster (perception)                   │
│  ✅ Can interrupt if not relevant               │
│                                                 │
└────────────────────────────────────────────────┘
```

---

### When to Use Streaming

✅ **Use streaming when:**
- Responses typically > 50 tokens
- User experience matters
- Building chat interfaces
- Long-form content generation

❌ **Don't use streaming when:**
- Responses are very short (<20 tokens)
- You need the complete response before processing
- Building APIs (usually want complete response)
- Cost calculation needs to be done upfront

---

## 7. Cost Calculation

### Understanding API Pricing

```
┌────────────────────────────────────────────────┐
│  PRICING MODEL (All providers)                 │
├────────────────────────────────────────────────┤
│                                                 │
│  Cost = (Input tokens × Input price) +        │
│         (Output tokens × Output price)         │
│                                                 │
│  Input:  Your prompt + conversation history    │
│  Output: Model's generated response            │
│                                                 │
│  ⚠️ Output tokens typically cost MORE!         │
│                                                 │
└────────────────────────────────────────────────┘
```

---

### Current Pricing (December 2024)

| Provider | Model | Input (per 1M tokens) | Output (per 1M tokens) |
|----------|-------|----------------------|------------------------|
| **OpenAI** | GPT-3.5-turbo | $0.50 | $1.50 |
| | GPT-4-turbo | $10.00 | $30.00 |
| | GPT-4 | $30.00 | $60.00 |
| **Anthropic** | Claude 3 Haiku | $0.25 | $1.25 |
| | Claude 3 Sonnet | $3.00 | $15.00 |
| | Claude 3 Opus | $15.00 | $75.00 |
| **Google** | Gemini Pro | $0.125 | $0.375 |

---

### Real-World Cost Examples

```
Example 1: Customer Support Chatbot
────────────────────────────────────
Model: GPT-3.5-turbo
Average conversation:
  - Input: 200 tokens (history + new question)
  - Output: 100 tokens (response)

Cost per conversation:
  (200 × $0.50/1M) + (100 × $1.50/1M)
  = $0.0001 + $0.00015
  = $0.00025 per conversation

1,000 conversations = $0.25
10,000 conversations = $2.50
100,000 conversations = $25.00

─────────────────────────────────────

Example 2: Document Summarization
────────────────────────────────────
Model: GPT-4-turbo
Long document:
  - Input: 5,000 tokens (document)
  - Output: 500 tokens (summary)

Cost per summary:
  (5,000 × $10/1M) + (500 × $30/1M)
  = $0.05 + $0.015
  = $0.065 per summary

100 summaries = $6.50

─────────────────────────────────────

Example 3: Code Generation
────────────────────────────────────
Model: Claude 3 Sonnet
Code request:
  - Input: 500 tokens (requirements)
  - Output: 800 tokens (code + explanation)

Cost per generation:
  (500 × $3/1M) + (800 × $15/1M)
  = $0.0015 + $0.012
  = $0.0135 per generation

1,000 generations = $13.50
```

---

### Cost Optimization Strategies

#### 1. **Choose the Right Model**
```
Question: "What's 2+2?"
❌ GPT-4: Overkill, expensive
✅ GPT-3.5-turbo: Perfect, cheap

Question: "Analyze this legal contract for risks"
❌ GPT-3.5-turbo: May miss nuances
✅ GPT-4 or Claude Opus: Worth the cost
```

#### 2. **Limit max_tokens Appropriately**
```
❌ max_tokens=2000 for simple Q&A (wasteful)
✅ max_tokens=100 for simple Q&A (efficient)
```

#### 3. **Manage Conversation History**
```
❌ Sending entire 50-message history every time
✅ Keep only last 10 messages + system message
✅ Summarize old messages periodically
```

#### 4. **Cache Responses**
```
✅ Cache common questions
✅ Cache system prompts
✅ Reuse responses when possible
```

#### 5. **Monitor Usage**
```
✅ Track costs per user/session
✅ Set spending limits
✅ Alert on unusual usage
```

---

## 8. Review & Key Takeaways

### 🎯 What You've Learned

✅ **How LLMs Work**
- Statistical prediction engines, not true intelligence
- Trained in 3 stages: pre-training, fine-tuning, alignment
- Generate text by predicting next token probabilities

✅ **Tokens**
- Basic units of LLM processing (~4 chars, ¾ word)
- Critical for costs, context limits, and performance
- Different encodings for different models

✅ **API Basics**
- Three major providers: OpenAI, Anthropic, Google
- Message structure: system, user, assistant roles
- System messages define behavior

✅ **Key Parameters**
- **temperature** (0-2): Controls creativity/randomness
- **max_tokens**: Limits response length
- **top_p** (0-1): Filters by probability (preferred)
- **top_k**: Fixed token limit (some APIs)

✅ **Streaming**
- Better UX for long responses
- Shows progress immediately
- Can be interrupted

✅ **Costs**
- Charged per token (input + output)
- Output typically costs more
- Optimize by choosing right model, limiting tokens, caching

---

### 🎓 Conceptual Knowledge Check

Before moving to the hands-on lab, make sure you understand:

1. **What does an LLM actually do?**
   <details>
   <summary>Answer</summary>
   Predicts the next most likely token based on input and training data. It's a statistical model, not truly intelligent.
   </details>

2. **Why do tokens matter?**
   <details>
   <summary>Answer</summary>
   They determine: (1) context limits, (2) API costs, (3) response speed
   </details>

3. **When would you use temperature=0 vs temperature=1.5?**
   <details>
   <summary>Answer</summary>
   Temp 0: Consistent, factual tasks (data extraction, classification)
   Temp 1.5: Creative tasks (writing, brainstorming, marketing)
   </details>

4. **What's the difference between temperature and top_p?**
   <details>
   <summary>Answer</summary>
   Temperature reshapes probabilities globally. top_p filters tokens by cumulative probability (more adaptive). Use one or the other, not both.
   </details>

5. **Why does streaming matter?**
   <details>
   <summary>Answer</summary>
   Better user experience - users see progress immediately rather than waiting 10+ seconds for complete response.
   </details>

---

### 🚀 Ready for Hands-On Practice?

Now that you understand the theory, it's time to **write actual code**!

👉 **[Continue to Hands-On Lab →](lab.md)**

In the lab, you'll:
- ✅ Set up your environment and API keys
- ✅ Make your first API calls
- ✅ Experiment with different parameters
- ✅ Implement streaming responses
- ✅ Build a complete chatbot (SupportGenie v0.1)

---

### 📚 Additional Reading (Optional)

Want to go deeper? Check out:
- [How GPT-3 Works - Visualizations](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)
- [OpenAI Tokenizer Tool](https://platform.openai.com/tokenizer)
- [Anthropic's Model Context Protocol](https://www.anthropic.com/research)
- [Google's AI Principles](https://ai.google/responsibility/principles/)

---

**Next:** [Hands-On Lab →](lab.md)
