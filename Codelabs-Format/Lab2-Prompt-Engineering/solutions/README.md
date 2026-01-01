# Lab 2 Solutions: Prompt Engineering

## 📚 Overview

Complete, production-ready solutions for Lab 2: Prompt Engineering. Each solution demonstrates advanced prompt engineering techniques with real-world applications.

---

## 📁 Files Included

### Core Exercises
- **`exercise1_prompt_quality.py`** - Vague vs specific prompts, 7-part structure
- **`exercise2_system_messages.py`** - System message control, AI personalities
- **`exercise3_few_shot_learning.py`** - Few-shot classification, sentiment analysis
- **`exercise4_chain_of_thought.py`** - Step-by-step reasoning, CoT prompting
- **`exercise5_prompt_templates.py`** - Reusable templates, variable substitution
- **`exercise6_edge_cases.py`** - Edge case handling, defensive prompting
- **`exercise7_tone_style.py`** - Tone control, formality levels, brand voice

### Capstone Project
- **`capstone_supportgenie_v02.py`** - SupportGenie v0.2 with advanced prompting

---

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you completed Lab 1 setup
# Your .env file should have:
OPENAI_API_KEY=sk-your-key-here
```

### Run Any Exercise
```bash
python exercise1_prompt_quality.py
python exercise2_system_messages.py
python exercise3_few_shot_learning.py
# ... etc
```

### Run Capstone
```bash
python capstone_supportgenie_v02.py
```

---

## 📖 Exercise Guide

### Exercise 1: Prompt Quality (15 min)
**File:** `exercise1_prompt_quality.py`

**What you'll learn:**
- Impact of specificity on output quality
- 7-part prompt structure
- Context and constraints

**Key concepts:**
1. Task/Role
2. Context
3. Instructions
4. Examples
5. Constraints
6. Output Format
7. Tone/Style

**Run it:**
```bash
python exercise1_prompt_quality.py
```

---

### Exercise 2: System Messages (15 min)
**File:** `exercise2_system_messages.py`

**What you'll learn:**
- Set AI personality and behavior
- Create specialized assistants
- Control output format
- Set behavioral constraints

**Features:**
- 5 different AI personalities
- Output format control (JSON, Markdown, etc.)
- Behavior constraints (do's and don'ts)

**Example:**
```python
system_message = """You are a patient teacher.
Use simple language and real-world examples.
Always encourage the learner."""
```

---

### Exercise 3: Few-Shot Learning (20 min)
**File:** `exercise3_few_shot_learning.py`

**What you'll learn:**
- Zero-shot vs few-shot comparison
- Sentiment analysis with examples
- Intent classification
- Structured data extraction

**Applications:**
- ✅ Sentiment analyzer (positive/negative/neutral)
- ✅ Intent classifier (question/complaint/purchase/etc.)
- ✅ Category classification
- ✅ Contact info extraction

**Sample output:**
```
😊 'This is fantastic!' → positive
😞 'Worst ever!' → negative
😐 'It arrived yesterday' → neutral
```

---

### Exercise 4: Chain-of-Thought (15 min)
**File:** `exercise4_chain_of_thought.py`

**What you'll learn:**
- Step-by-step reasoning
- Improve accuracy on complex tasks
- Transparent problem-solving
- CoT + few-shot combination

**Use cases:**
- Math word problems
- Logical reasoning
- Multi-step analysis

**Magic phrase:**
> "Let's solve this step by step..."

---

### Exercise 5: Prompt Templates (20 min)
**File:** `exercise5_prompt_templates.py`

**What you'll learn:**
- Build reusable templates
- Variable substitution
- Template library organization

**Includes:**
- Customer support template
- Code review template
- Email writer template
- Meeting summary template
- Social media template

**Usage:**
```python
template = PromptTemplate(
    "You are a {role}. Help with {task}."
)
prompt = template.format(role="developer", task="debugging")
```

---

### Exercise 6: Edge Case Handling (15 min)
**File:** `exercise6_edge_cases.py`

**What you'll learn:**
- Identify common edge cases
- Build defensive prompts
- Handle unexpected inputs
- Create fallback responses

**Edge cases covered:**
- Empty/blank input
- Invalid data types
- Out of range values
- Inappropriate content
- Off-topic requests

---

### Exercise 7: Tone and Style (15 min)
**File:** `exercise7_tone_style.py`

**What you'll learn:**
- Control conversational tone
- Adjust formality levels
- Create consistent brand voice
- Switch writing styles

**Tone variations:**
- Professional
- Friendly
- Empathetic
- Urgent
- Casual

**Formality levels:**
- Very Casual → Very Formal

---

### 🏆 Capstone: SupportGenie v0.2
**File:** `capstone_supportgenie_v02.py`

**NEW in v0.2:**
- ✅ Few-shot intent classification
- ✅ 7-part structured system messages
- ✅ Chain-of-thought reasoning for complex issues
- ✅ Comprehensive edge case handling
- ✅ Context-aware tone control
- ✅ Reusable prompt templates

**Run demo:**
```bash
python capstone_supportgenie_v02.py
```

**Features:**
1. **Intent Classification** - Automatically detects: question, complaint, purchase, cancel, praise
2. **Edge Case Handling** - Handles empty, short, and inappropriate inputs
3. **Enhanced System Messages** - Structured with role, expertise, constraints
4. **Three Modes** - Support, Sales, Technical (each with specialized prompts)

**Example interaction:**
```
👤 Customer: "I can't log into my account"
🏷️  Intent: question
🤖 Agent: I understand you're having trouble logging in. Let me help you...
```

---

## 💡 Key Techniques Learned

### 7-Part Prompt Structure
```
1. Task/Role - "You are a..."
2. Context - Background information
3. Instructions - What to do
4. Examples - Show desired output
5. Constraints - Limits and requirements
6. Output Format - How to structure response
7. Tone/Style - Voice and personality
```

### Few-Shot Pattern
```
Examples:
Input: [example 1] → Output: [result 1]
Input: [example 2] → Output: [result 2]
Input: [example 3] → Output: [result 3]

Now:
Input: [actual input] → Output:
```

### Chain-of-Thought
```
"Let's solve this step by step:
1. First, identify...
2. Then, calculate...
3. Finally, verify..."
```

### Edge Case Handling
```
EDGE CASES:
- If empty → "Please provide input"
- If invalid → "Invalid input, please try again"
- If out of scope → "That's outside my expertise"
```

---

## 🎯 Best Practices Summary

### DO:
✅ Be specific and detailed
✅ Provide context and examples
✅ Set clear constraints
✅ Define desired tone/format
✅ Handle edge cases
✅ Use system messages for behavior
✅ Test with diverse inputs

### DON'T:
❌ Use vague prompts
❌ Skip context
❌ Ignore edge cases
❌ Forget to specify tone
❌ Over-complicate unnecessarily
❌ Mix too many techniques at once

---

## 📊 Comparison: v0.1 vs v0.2

| Feature | v0.1 (Lab 1) | v0.2 (Lab 2) |
|---------|--------------|--------------|
| System Messages | Basic | 7-part structure |
| Intent Detection | None | Few-shot classification |
| Edge Cases | Minimal | Comprehensive |
| Reasoning | Direct | Chain-of-thought |
| Prompts | Ad-hoc | Template-based |
| Tone Control | Basic | Context-aware |
| Response Quality | Good | Excellent |

---

## 🧪 Testing

All solutions have been syntax-validated:

```bash
# Test syntax
for file in *.py; do
    python3 -m py_compile "$file" && echo "✅ $file"
done
```

---

## 🔧 Troubleshooting

**Issue:** Responses too generic
**Solution:** Add more specific context and constraints

**Issue:** Inconsistent outputs
**Solution:** Use few-shot examples and temperature=0

**Issue:** Edge cases not handled
**Solution:** Add explicit edge case instructions to prompt

---

## 📚 Further Reading

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Few-Shot Learning Paper](https://arxiv.org/abs/2005.14165)
- [Chain-of-Thought Paper](https://arxiv.org/abs/2201.11903)

---

## 🎓 Next Steps

After completing Lab 2:
1. **Lab 3:** Document Processing & Embeddings
2. **Lab 4:** Semantic Search & Retrieval
3. **Lab 5:** Complete RAG Pipeline (SupportGenie v3.0)

---

## 💻 Code Quality

All solutions include:
- ✅ Comprehensive docstrings
- ✅ Inline comments
- ✅ Real-world examples
- ✅ Error handling
- ✅ Best practices
- ✅ Educational explanations

---

**Happy Prompting! 🎯**

*Master prompt engineering to unlock the full potential of LLMs*
