# Lab 1 Solutions

Complete, tested solutions for all Lab 1 exercises.

---

## 📁 Files Included

| File | Exercise | Description |
|------|----------|-------------|
| `exercise1_openai.py` | Exercise 1 | First API call to OpenAI |
| `exercise1_claude.py` | Exercise 1 | First API call to Claude |
| `exercise1_gemini.py` | Exercise 1 | First API call to Gemini |
| `exercise2_tokens.py` | Exercise 2 | Token counting and visualization |
| `exercise3_temperature.py` | Exercise 3 | Temperature experiments |
| `exercise4_parameters.py` | Exercise 4 | Parameter comparison (max_tokens, top_p) |
| `exercise5_streaming.py` | Exercise 5 | Streaming implementation |
| `exercise6_costs.py` | Exercise 6 | Cost calculator |
| `exercise7_chatbot.py` | Exercise 7 | SimpleChatbot class |
| `capstone_supportgenie.py` | Capstone | Complete SupportGenie v0.1 |

---

## 🚀 How to Use

### Run Individual Solutions:

```bash
# Make sure you have .env file configured
cd solutions/lab1

# Run any solution
python exercise1_openai.py
python capstone_supportgenie.py
```

### Run All Tests:

```bash
# Test all solutions
python test_all.py
```

---

## 📝 Solution Notes

### Best Practices Demonstrated:
- ✅ Environment variable usage
- ✅ Error handling
- ✅ Clear docstrings
- ✅ Type hints where appropriate
- ✅ Modular code structure
- ✅ Inline comments for complex logic

### Alternative Approaches:
Each solution includes comments about alternative implementations where applicable.

---

## 🎯 Learning from Solutions

**Tips:**
1. Don't just copy - understand WHY each line works
2. Try modifying parameters and observing changes
3. Compare your solution to provided one
4. Run with different inputs to see behavior
5. Read inline comments for explanations

**Common Mistakes Addressed:**
- Hardcoding API keys (solution: use .env)
- Not handling errors (solution: try-except blocks)
- Inefficient token counting (solution: use tiktoken)
- Not tracking costs (solution: calculate per call)

---

## ✅ Verified

All solutions:
- [x] Run without errors
- [x] Produce expected output
- [x] Handle edge cases
- [x] Follow best practices
- [x] Include detailed comments
- [x] Tested with Python 3.8+

---

**Questions?** Review the inline comments or refer back to learning.md for theory.
