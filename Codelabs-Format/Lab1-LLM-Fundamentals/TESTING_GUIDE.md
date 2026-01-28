# Testing Guide for Lab1-LLM-Fundamentals

This guide provides comprehensive instructions for testing your lab to ensure everything works correctly.

## 📋 Table of Contents

1. [Quick Test](#quick-test)
2. [Complete Test Suite](#complete-test-suite)
3. [Manual Testing Steps](#manual-testing-steps)
4. [Testing Checklist](#testing-checklist)
5. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Test

Run this single command to test everything at once:

```bash
python3 test_lab.py
```

This will check:
- ✅ Python version compatibility
- ✅ Required package installation
- ✅ File syntax errors
- ✅ Project structure
- ✅ Documentation completeness
- ✅ Configuration files

---

## 📦 Complete Test Suite

### Step 1: Environment Setup

1. **Check Python version:**
   ```bash
   python3 --version
   ```
   Required: Python 3.8 or higher

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installations:**
   ```bash
   pip list | grep -E "(openai|anthropic|google-generativeai|tiktoken|python-dotenv)"
   ```

### Step 2: Configuration Setup

1. **Create .env file:**
   ```bash
   cp .env.example .env
   ```

2. **Add your API keys to .env:**
   ```bash
   # Edit .env and add at least one API key
   nano .env  # or use your preferred editor
   ```

3. **Test API key loading:**
   ```bash
   cd solutions
   python3 test_setup.py
   ```

   Expected output:
   ```
   ✅ Packages imported successfully!
   OpenAI key loaded: True
   Anthropic key loaded: True
   Google key loaded: True
   ```

### Step 3: Syntax Validation

**Test all Python files compile without errors:**

```bash
cd solutions
for file in *.py; do
    echo "Testing $file..."
    python3 -m py_compile "$file" && echo "✅ OK" || echo "❌ FAILED"
done
```

### Step 4: Exercise Testing

Test each exercise individually:

#### Exercise 1: First API Call
```bash
# OpenAI (requires OPENAI_API_KEY)
python3 exercise1_openai.py

# Claude (requires ANTHROPIC_API_KEY)
python3 exercise1_claude.py

# Gemini (requires GOOGLE_API_KEY)
python3 exercise1_gemini.py
```

**Expected output:**
- Question and answer displayed
- Token usage shown
- Cost calculation
- No errors

#### Exercise 2: Token Counting
```bash
python3 exercise2_tokens.py
```

**Expected output:**
- Token counts for sample texts
- Character-to-token ratios
- Visual token breakdown

#### Exercise 3: Temperature Experiments
```bash
python3 exercise3_temperature.py
```

**Expected output:**
- Multiple responses at different temperatures
- Variation analysis
- Temperature recommendations

#### Exercise 4: Parameter Comparison
```bash
python3 exercise4_parameters.py
```

**Expected output:**
- Parameter effect demonstrations
- Side-by-side comparisons
- Best practices summary

#### Exercise 5: Streaming
```bash
python3 exercise5_streaming.py
```

**Expected output:**
- Token-by-token streaming output
- Performance comparison
- Streaming vs non-streaming demo

#### Exercise 6: Cost Calculator
```bash
python3 exercise6_cost_calculator.py
```

**Expected output:**
- Cost calculations for different models
- Budget estimation tools
- Cost optimization tips

#### Exercise 7: Simple Chatbot
```bash
python3 exercise7_chatbot.py
```

**Expected output:**
- Interactive chat mode (if enabled)
- Conversation history tracking
- Session statistics

#### Capstone: SupportGenie
```bash
python3 capstone_supportgenie_v01.py
```

**Expected output:**
- Mode selection menu
- Interactive chat interface
- Commands working (stats, export, quit)
- Session tracking

---

## 🔍 Manual Testing Steps

### Test 1: Notebook Functionality

1. **Open notebook:**
   ```bash
   jupyter notebook Lab1-LLM-Fundamentals.ipynb
   ```

2. **Run all cells sequentially:**
   - Click "Cell" → "Run All"
   - Watch for errors
   - Verify outputs appear

3. **Check for:**
   - ✅ All cells execute without errors
   - ✅ Markdown renders correctly
   - ✅ Code outputs display properly
   - ✅ Interactive elements work

### Test 2: Code Quality

1. **Check for syntax errors:**
   ```bash
   find . -name "*.py" -exec python3 -m py_compile {} \;
   ```

2. **Check imports:**
   ```bash
   grep -r "^import\|^from" solutions/*.py
   ```

3. **Check for hardcoded API keys (security):**
   ```bash
   grep -r "sk-" solutions/*.py
   grep -r "API_KEY.*=" solutions/*.py | grep -v "os.getenv"
   ```
   Should return no results!

### Test 3: Student Experience

Simulate a student going through the lab:

1. **Start fresh:**
   ```bash
   rm -rf __pycache__
   rm .env  # If testing fresh setup
   ```

2. **Follow setup instructions:**
   - Install dependencies
   - Configure .env
   - Run test_setup.py

3. **Complete each exercise in order:**
   - Read the instructions
   - Run the code
   - Verify output matches expectations

4. **Test error handling:**
   - Try with missing API keys
   - Try with invalid keys
   - Test with network issues (disconnect)

---

## ✅ Testing Checklist

Use this checklist to ensure comprehensive testing:

### Environment Setup
- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] .env file created with API keys
- [ ] .gitignore prevents .env from being committed

### File Structure
- [ ] All exercise files present (exercise1-7)
- [ ] Capstone file present
- [ ] test_setup.py present
- [ ] README.md present
- [ ] Notebook file present

### Code Quality
- [ ] All .py files compile without syntax errors
- [ ] No hardcoded API keys in code
- [ ] All imports are present and correct
- [ ] Code follows Python best practices

### Functionality
- [ ] test_setup.py runs successfully
- [ ] Exercise 1 (OpenAI) works
- [ ] Exercise 1 (Claude) works (if key provided)
- [ ] Exercise 1 (Gemini) works (if key provided)
- [ ] Exercise 2 (tokens) works without API
- [ ] Exercise 3 (temperature) works
- [ ] Exercise 4 (parameters) works
- [ ] Exercise 5 (streaming) works
- [ ] Exercise 6 (cost calculator) works
- [ ] Exercise 7 (chatbot) works
- [ ] Capstone (SupportGenie) works

### Notebook
- [ ] Notebook opens without errors
- [ ] All cells execute successfully
- [ ] Markdown renders correctly
- [ ] Code outputs display properly
- [ ] Interactive elements work

### Documentation
- [ ] README.md is clear and complete
- [ ] Solutions README.md is helpful
- [ ] Code comments are informative
- [ ] Learning objectives are clear

### User Experience
- [ ] Instructions are easy to follow
- [ ] Error messages are helpful
- [ ] Examples are relevant
- [ ] Progression is logical

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Issue: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'openai'
```

**Solution:**
```bash
pip install openai anthropic google-generativeai tiktoken python-dotenv
```

#### Issue: API Key Error
```
openai.error.AuthenticationError: Invalid API key
```

**Solution:**
1. Check .env file exists
2. Verify API key is correct
3. Ensure no extra spaces in .env
4. Check key hasn't expired

#### Issue: Token Limit Exceeded
```
Error: This model's maximum context length is 4096 tokens
```

**Solution:**
1. Reduce max_tokens parameter
2. Shorten the input prompt
3. Clear conversation history

#### Issue: Rate Limit Error
```
openai.error.RateLimitError: Rate limit exceeded
```

**Solution:**
1. Wait a minute and retry
2. Check your API usage quota
3. Upgrade API tier if needed

#### Issue: Import Error in Notebook
```
ImportError: cannot import name 'OpenAI'
```

**Solution:**
1. Install packages in notebook:
   ```python
   !pip install openai
   ```
2. Restart kernel
3. Re-run cells

---

## 🧪 Advanced Testing

### Load Testing

Test with high volume:

```python
# Test 100 rapid API calls
for i in range(100):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Count: {i}"}],
        max_tokens=10
    )
    print(f"Call {i}: Success")
```

### Error Handling Testing

Test error scenarios:

```python
# Test with invalid API key
client = OpenAI(api_key="invalid-key")
# Should raise AuthenticationError

# Test with empty message
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": ""}]
)
# Should handle gracefully

# Test with excessive max_tokens
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hi"}],
    max_tokens=100000
)
# Should raise error
```

### Performance Testing

Measure response times:

```python
import time

start = time.time()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
end = time.time()

print(f"Response time: {end - start:.2f} seconds")
```

---

## 📊 Test Report Template

Use this template to document your testing:

```markdown
# Lab1 Test Report

**Date:** YYYY-MM-DD
**Tester:** Your Name

## Environment
- Python Version: 3.x.x
- OS: macOS/Linux/Windows
- Dependencies: All installed ✅

## Test Results

### Setup Tests
- [ ] Dependencies installed
- [ ] API keys configured
- [ ] test_setup.py passed

### Exercise Tests
- [ ] Exercise 1 (OpenAI): PASS/FAIL
- [ ] Exercise 1 (Claude): PASS/FAIL
- [ ] Exercise 1 (Gemini): PASS/FAIL
- [ ] Exercise 2: PASS/FAIL
- [ ] Exercise 3: PASS/FAIL
- [ ] Exercise 4: PASS/FAIL
- [ ] Exercise 5: PASS/FAIL
- [ ] Exercise 6: PASS/FAIL
- [ ] Exercise 7: PASS/FAIL
- [ ] Capstone: PASS/FAIL

### Notebook Tests
- [ ] All cells execute: PASS/FAIL
- [ ] Outputs correct: PASS/FAIL

## Issues Found
1. [Issue description]
   - Severity: High/Medium/Low
   - Solution: [How to fix]

2. [Issue description]
   - Severity: High/Medium/Low
   - Solution: [How to fix]

## Overall Assessment
- Pass Rate: X/Y tests passed (Z%)
- Ready for students: YES/NO
- Recommended fixes: [List]
```

---

## 🎓 Student Testing

Ask students to test and provide feedback on:

1. **Clarity**: Are instructions clear?
2. **Difficulty**: Is the progression appropriate?
3. **Errors**: Did they encounter any errors?
4. **Time**: How long did it take?
5. **Learning**: Did they understand the concepts?

---

## ✨ Final Verification

Before releasing the lab:

1. **Run full test suite:**
   ```bash
   python3 test_lab.py
   ```

2. **Test on clean environment:**
   ```bash
   # Create fresh virtual environment
   python3 -m venv test_env
   source test_env/bin/activate
   pip install -r requirements.txt
   python3 test_lab.py
   ```

3. **Have peer review:**
   - Another instructor tests
   - Provides feedback
   - Confirms clarity

4. **Student pilot test:**
   - 2-3 students complete lab
   - Collect feedback
   - Make adjustments

---

## 🎯 Success Criteria

Your lab is ready when:

- ✅ All automated tests pass
- ✅ All exercises run without errors
- ✅ Notebook executes completely
- ✅ Documentation is clear and complete
- ✅ Students can complete independently
- ✅ Learning objectives are met
- ✅ Code follows best practices
- ✅ No security issues (hardcoded keys)

---

**Happy Testing! 🧪**

*Last Updated: 2026-01-19*
