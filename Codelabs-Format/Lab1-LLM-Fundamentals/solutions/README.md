# Lab 1 Solutions: LLM Fundamentals & API Usage

## 📚 Overview

This directory contains complete, well-documented solutions for all Lab 1 exercises. Each solution includes:
- ✅ Complete working code
- ✅ Detailed comments and explanations
- ✅ Best practices and production patterns
- ✅ Error handling
- ✅ Additional learning resources

---

## 📁 Files Included

### Setup & Configuration
- **`test_setup.py`** - Verify API key configuration

### Core Exercises (1-7)
- **`exercise1_openai.py`** - First API call with OpenAI
- **`exercise1_claude.py`** - First API call with Claude
- **`exercise1_gemini.py`** - First API call with Gemini
- **`exercise2_tokens.py`** - Token counting and cost calculation
- **`exercise3_temperature.py`** - Temperature parameter experiments
- **`exercise4_parameters.py`** - LLM parameter comparison
- **`exercise5_streaming.py`** - Streaming response implementation
- **`exercise6_cost_calculator.py`** - Cost tracking and monitoring
- **`exercise7_chatbot.py`** - Simple chatbot with conversation history

### Capstone Project
- **`capstone_supportgenie_v01.py`** - Professional customer support chatbot (SupportGenie v0.1)

---

## 🚀 Quick Start

### 1. Prerequisites

Install required packages:
```bash
pip install openai anthropic google-generativeai tiktoken python-dotenv
```

### 2. Configure API Keys

Create a `.env` file in this directory:
```bash
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here
```

**Note:** You need at least ONE API key to run the exercises.

### 3. Test Your Setup

```bash
python test_setup.py
```

Expected output:
```
API Key Status:
✅ OpenAI: Loaded
✅ Anthropic: Loaded
✅ Google: Loaded

🎉 Setup complete! You're ready to code.
```

---

## 📖 Exercise Guide

### Exercise 1: First API Call
**Files:** `exercise1_openai.py`, `exercise1_claude.py`, `exercise1_gemini.py`

**What you'll learn:**
- Making your first LLM API call
- Understanding request/response structure
- Extracting token usage
- Basic cost calculation

**Run it:**
```bash
# OpenAI (recommended for beginners)
python exercise1_openai.py

# Or try Claude
python exercise1_claude.py

# Or try Gemini (free tier)
python exercise1_gemini.py
```

**Key concepts:**
- System vs user messages
- Model selection
- Token tracking
- API response structure

---

### Exercise 2: Token Counting
**File:** `exercise2_tokens.py`

**What you'll learn:**
- Accurate token counting with tiktoken
- Character-to-token ratio
- Token visualization
- Cost estimation

**Run it:**
```bash
python exercise2_tokens.py
```

**Sample output:**
```
Text: 'Hello, world!'
  Tokens: 4
  Characters: 13
  Ratio: 3.25 chars/token
```

**Key concepts:**
- Tokens ≠ words
- ~4 characters per token (English)
- Special characters are separate tokens
- Always use tiktoken for accuracy

---

### Exercise 3: Temperature Experiments
**File:** `exercise3_temperature.py`

**What you'll learn:**
- Temperature effects on creativity
- Deterministic vs random outputs
- When to use high/low temperature
- Temperature selection guide

**Run it:**
```bash
python exercise3_temperature.py
```

**Key findings:**
- **Temperature 0.0** → Deterministic (same every time)
- **Temperature 0.7** → Balanced (default)
- **Temperature 1.0+** → Creative (different every time)

**Use cases:**
| Task | Recommended Temperature |
|------|------------------------|
| Math/Facts | 0.0 |
| Code generation | 0.2-0.3 |
| General chat | 0.7 |
| Creative writing | 0.9-1.2 |

---

### Exercise 4: Parameter Comparison
**File:** `exercise4_parameters.py`

**What you'll learn:**
- max_tokens effects
- top_p vs temperature
- presence_penalty and frequency_penalty
- Optimal parameter combinations

**Run it:**
```bash
python exercise4_parameters.py
```

**Parameters covered:**
- **max_tokens** - Limit response length
- **temperature** - Control randomness
- **top_p** - Nucleus sampling (alternative to temperature)
- **presence_penalty** - Encourage new topics
- **frequency_penalty** - Reduce repetition

---

### Exercise 5: Streaming Implementation
**File:** `exercise5_streaming.py`

**What you'll learn:**
- Streaming vs non-streaming
- Real-time token display
- Stream chunk handling
- Performance comparison

**Run it:**
```bash
python exercise5_streaming.py
```

**When to use streaming:**
- ✅ Chatbots and conversational AI
- ✅ Long-form content generation
- ✅ Interactive applications
- ✅ Better user experience

**When to skip streaming:**
- ⏭️ Batch processing
- ⏭️ Backend services
- ⏭️ When you need complete response first

---

### Exercise 6: Cost Calculator
**File:** `exercise6_cost_calculator.py`

**What you'll learn:**
- Calculate per-call costs
- Track cumulative spending
- Estimate project costs
- Budget monitoring

**Run it:**
```bash
python exercise6_cost_calculator.py
```

**Features:**
- Cost calculation for multiple models
- Project cost estimation
- Budget alerts
- Cost optimization tips

**Sample cost comparison:**
| Model | Input (1K tokens) | Output (1K tokens) |
|-------|-------------------|-------------------|
| GPT-3.5-turbo | $0.0015 | $0.002 |
| GPT-4 | $0.03 | $0.06 |
| GPT-4o-mini | $0.00015 | $0.0006 |

---

### Exercise 7: Build SimpleChatbot
**File:** `exercise7_chatbot.py`

**What you'll learn:**
- Conversation history management
- Context tracking
- Interactive chat loop
- Different chatbot personalities

**Run it:**
```bash
python exercise7_chatbot.py
```

**Features:**
- Maintains conversation context
- Configurable system messages
- Token and cost tracking
- Interactive mode

**Try it:**
The script includes an interactive mode! When prompted, type 'y' to start chatting.

---

### 🏆 Capstone: SupportGenie v0.1
**File:** `capstone_supportgenie_v01.py`

**What you'll build:**
A professional, production-ready customer support chatbot with:
- ✅ Multiple support modes (support, sales, technical)
- ✅ Conversation history tracking
- ✅ Cost and token monitoring
- ✅ Session management
- ✅ Conversation export to JSON
- ✅ Error handling

**Run it:**
```bash
python capstone_supportgenie_v01.py
```

**Features:**

1. **Three Support Modes:**
   - Customer Support - General help
   - Sales Assistant - Product recommendations
   - Technical Support - Troubleshooting

2. **Commands:**
   - `quit` - End conversation with stats
   - `clear` - Clear history
   - `stats` - Show session statistics
   - `export` - Save conversation to JSON
   - `mode <name>` - Switch modes

3. **Tracking:**
   - Session ID and duration
   - Message count
   - Token usage
   - Total cost

**Example usage:**
```
Choose mode:
  1. Demo (automated)
  2. Interactive chat
  3. Exit

Choice: 2

Select support mode:
  1. Customer Support
  2. Sales Assistant
  3. Technical Support

Choice (1-3): 1
Enter company name (default: TechCorp): MyCompany

You: I can't reset my password
MyCompany Agent: I'd be happy to help you reset your password...
```

**Export feature:**
Exports conversation to JSON with:
- Complete message history
- Session metadata
- Usage statistics

---

## 💡 Learning Path

### Recommended Order:
1. **Setup** → `test_setup.py`
2. **First API Call** → `exercise1_openai.py`
3. **Tokens** → `exercise2_tokens.py`
4. **Temperature** → `exercise3_temperature.py`
5. **Parameters** → `exercise4_parameters.py`
6. **Streaming** → `exercise5_streaming.py`
7. **Costs** → `exercise6_cost_calculator.py`
8. **Chatbot** → `exercise7_chatbot.py`
9. **Capstone** → `capstone_supportgenie_v01.py`

---

## 🎯 Key Takeaways

### Core Concepts Mastered:
1. ✅ **API Integration** - OpenAI, Claude, Gemini
2. ✅ **Token Management** - Counting and optimization
3. ✅ **Parameter Tuning** - Temperature, top_p, penalties
4. ✅ **Cost Control** - Tracking and budgeting
5. ✅ **Conversation Handling** - History and context
6. ✅ **Production Patterns** - Error handling, logging

### Best Practices:
- ✅ Always use environment variables for API keys
- ✅ Track token usage for cost management
- ✅ Set max_tokens to prevent runaway costs
- ✅ Choose the right temperature for your use case
- ✅ Use streaming for better UX
- ✅ Maintain conversation history for context
- ✅ Handle errors gracefully

---

## 🔧 Troubleshooting

### Common Issues:

**1. API Key Errors (401)**
```
Problem: Invalid API key
Solution: Check your .env file and verify keys are correct
```

**2. Import Errors**
```
Problem: Module not found
Solution: pip install openai anthropic tiktoken python-dotenv
```

**3. Rate Limit Errors**
```
Problem: Too many requests
Solution: Add delays between calls or upgrade API tier
```

**4. Token Limit Exceeded**
```
Problem: Response too long
Solution: Reduce max_tokens or clear conversation history
```

---

## 📚 Additional Resources

### Documentation:
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Anthropic Claude Docs](https://docs.anthropic.com/)
- [Google Gemini Docs](https://ai.google.dev/docs)
- [Tiktoken Documentation](https://github.com/openai/tiktoken)

### Next Steps:
- **Lab 2**: Advanced Prompt Engineering
- **Lab 3**: Document Processing & Embeddings
- **Lab 5**: RAG Pipeline (SupportGenie v3.0)

---

## 🎓 Grading Criteria (If Submitting)

Your solution should demonstrate:
- ✅ Working code that runs without errors
- ✅ Proper API key management (.env)
- ✅ Token tracking and cost awareness
- ✅ Error handling
- ✅ Clean, readable code
- ✅ Understanding of key concepts

---

## 💻 Code Quality Notes

All solutions include:
- Detailed docstrings
- Inline comments
- Type hints where applicable
- Error handling
- Production-ready patterns
- Educational explanations

---

## 🚀 Going Further

### Challenges:
1. Modify SupportGenie to support multiple languages
2. Add conversation persistence (save/load)
3. Implement rate limiting
4. Add sentiment analysis to conversations
5. Create a web interface (Flask/FastAPI)

### Ideas:
- Build a coding tutor chatbot
- Create a creative writing assistant
- Make a technical documentation helper
- Develop a language learning bot

---

## ⚖️ License & Usage

These solutions are provided for educational purposes. Feel free to:
- ✅ Learn from the code
- ✅ Modify for your projects
- ✅ Use as templates
- ✅ Share with others

---

## 🤝 Need Help?

If you get stuck:
1. Review the lab.md instructions
2. Check the learning.md concepts
3. Review solution comments
4. Test with simpler examples
5. Ask in course discussions

---

**Happy Coding! 🎉**

*These solutions demonstrate production-ready patterns while remaining educational and easy to understand.*
