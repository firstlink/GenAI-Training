# Session 0: Setup & Environment Configuration

## Overview
Duration: 15 minutes
Difficulty: Beginner

In this setup session, you'll configure your development environment and verify that all tools and API keys are working correctly.

---

## Learning Objectives

By the end of this session, you will:
- ✅ Have all required API keys configured
- ✅ Understand how to use Google Colab notebooks
- ✅ Successfully test API connections
- ✅ Be ready to start building Gen AI applications

---

## Part 1: Obtain API Keys

### OpenAI API Key (Required)

**Step 1**: Go to [OpenAI Platform](https://platform.openai.com/)

**Step 2**: Sign up or log in to your account

**Step 3**: Navigate to **API Keys**
- Click on your profile icon (top right)
- Select "API keys" or visit https://platform.openai.com/api-keys

**Step 4**: Create a new secret key
- Click **"Create new secret key"**
- Give it a name like "GenAI-Course"
- Click **"Create secret key"**

**Step 5**: Copy and save your key
- ⚠️ **IMPORTANT**: You can only see this key once!
- Copy it immediately and save it in a secure location
- Never share this key publicly or commit it to GitHub

**Billing Setup**:
- OpenAI requires a paid account for API access
- Add a payment method in the billing section
- Set spending limits to control costs (recommended: $10/month for learning)

**Free Credits**:
- New accounts may receive $5-18 in free credits
- Check your usage at https://platform.openai.com/usage

---

### Anthropic Claude API Key (Optional but Recommended)

**Step 1**: Visit [Anthropic Console](https://console.anthropic.com/)

**Step 2**: Sign up for an account

**Step 3**: Navigate to **API Keys**

**Step 4**: Click **"Create Key"**
- Name it "GenAI-Course"
- Copy and save securely

**Billing Setup**:
- Add payment method in console
- Anthropic provides $5 in free credits for new users
- Usage is pay-as-you-go

**Why use Claude?**:
- Excellent for longer context (200K tokens)
- Strong at following instructions
- Good for tool use and agentic workflows

---

### Google Gemini API Key (Optional)

**Step 1**: Go to [Google AI Studio](https://makersuite.google.com/app/apikey)

**Step 2**: Sign in with your Google account

**Step 3**: Click **"Get API Key"**

**Step 4**: Create API key
- Select "Create API key in new project" or choose existing project
- Copy your key

**Free Tier**:
- Gemini offers a generous free tier
- 15 requests per minute
- 1 million tokens per month free

---

## Part 2: Google Colab Setup

### What is Google Colab?

Google Colaboratory (Colab) is a free Jupyter notebook environment that:
- Runs in the cloud (no local setup needed)
- Provides free GPU access
- Comes with many libraries pre-installed
- Allows easy sharing and collaboration

### Opening Your First Notebook

**Step 1**: Open the setup notebook
- Navigate to `notebooks/00_Setup.ipynb`
- Right-click and select "Open with Google Colaboratory"
- OR upload to your Google Drive and open with Colab

**Step 2**: Understand the Colab interface
- **Code cells**: Contain Python code (gray background)
- **Text cells**: Contain markdown documentation
- **Run button**: ▶️ icon to execute cells
- **Runtime**: The Python environment running your code

**Step 3**: Configure runtime (if needed)
- Click **Runtime → Change runtime type**
- For this course: Python 3, No GPU/TPU needed (use CPU)
- For heavy models later: Can use GPU

### Colab Tips
- Use `Ctrl+Enter` (Windows/Linux) or `Cmd+Enter` (Mac) to run cells
- Use `Shift+Enter` to run cell and move to next
- Access file menu to download, save to Drive, etc.
- Can install packages with `!pip install package-name`

---

## Part 3: Configure API Keys in Colab

### Method 1: Using Colab Secrets (Recommended)

This is the secure way to store API keys in Colab:

**Step 1**: In your Colab notebook, click the 🔑 **Secrets** icon in the left sidebar

**Step 2**: Click **"+ Add new secret"**

**Step 3**: Add each API key:
```
Name: OPENAI_API_KEY
Value: sk-...your-actual-key...
```

**Step 4**: Enable notebook access
- Toggle the switch to allow this notebook to access the secret

**Step 5**: Repeat for other keys:
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

**Step 6**: Access in code:
```python
from google.colab import userdata
import os

# Get API keys from Colab secrets
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

### Method 2: Direct Input (Quick but Less Secure)

For quick testing only:

```python
import os
from getpass import getpass

# Prompt for API key (won't show on screen)
api_key = getpass('Enter your OpenAI API key: ')
os.environ['OPENAI_API_KEY'] = api_key
```

⚠️ **Warning**: Don't hardcode keys directly in notebooks you plan to share!

---

## Part 4: Verify Installation

### Test OpenAI Connection

```python
# Install OpenAI library
!pip install openai

# Test connection
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Make a simple test call
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Say 'Setup successful!' if you can read this."}
    ],
    max_tokens=20
)

print(response.choices[0].message.content)
```

**Expected Output**:
```
Setup successful!
```

### Test Claude Connection (if configured)

```python
# Install Anthropic library
!pip install anthropic

from anthropic import Anthropic
import os

client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

# Make a test call
message = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=20,
    messages=[
        {"role": "user", "content": "Say 'Claude setup successful!' if you can read this."}
    ]
)

print(message.content[0].text)
```

**Expected Output**:
```
Claude setup successful!
```

### Test Gemini Connection (if configured)

```python
# Install Google AI library
!pip install google-generativeai

import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

# Make a test call
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content('Say "Gemini setup successful!" if you can read this.')

print(response.text)
```

**Expected Output**:
```
Gemini setup successful!
```

---

## Part 5: Install Course Dependencies

Run this cell to install all libraries needed for the course:

```python
# Install all required libraries
!pip install openai anthropic google-generativeai
!pip install langchain langchain-community langchain-openai
!pip install sentence-transformers chromadb
!pip install faiss-cpu
!pip install python-dotenv
!pip install requests beautifulsoup4
!pip install gradio
!pip install tiktoken

print("✅ All dependencies installed successfully!")
```

### Verify Key Libraries

```python
# Test imports
import openai
import anthropic
import google.generativeai as genai
import langchain
from sentence_transformers import SentenceTransformer
import chromadb

print("✅ All imports successful!")
print(f"OpenAI version: {openai.__version__}")
print(f"LangChain version: {langchain.__version__}")
```

---

## Part 6: Understanding Costs

### OpenAI Pricing (as of Dec 2024)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-4 Turbo | $10.00 | $30.00 |
| GPT-4 | $30.00 | $60.00 |
| GPT-3.5 Turbo | $0.50 | $1.50 |

**For this course**: Expect to spend $1-5 if you run all exercises.

### Anthropic Pricing

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude 3 Opus | $15.00 | $75.00 |
| Claude 3 Sonnet | $3.00 | $15.00 |
| Claude 3 Haiku | $0.25 | $1.25 |

### Google Gemini Pricing

| Model | Free Tier | Paid (per 1M tokens) |
|-------|-----------|----------------------|
| Gemini Pro | 15 RPM, 1M tokens/month | $0.125 / $0.375 |
| Gemini Pro Vision | 15 RPM | $0.125 / $0.375 |

**Tip**: Start with Gemini's free tier or GPT-3.5-turbo to minimize costs while learning.

---

## Part 7: Cost Control Best Practices

### Set Spending Limits

**OpenAI**:
1. Go to https://platform.openai.com/account/billing/limits
2. Set monthly spending limits
3. Enable email notifications

**Anthropic**:
1. Visit console billing settings
2. Set budget alerts

### Monitor Usage

```python
# Check OpenAI usage in code
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# This is an example - actual usage tracking requires API calls
# to the usage endpoint or checking the dashboard
print("Check usage at: https://platform.openai.com/usage")
```

### Development Tips to Save Money

1. **Use cheaper models for testing**:
   - GPT-3.5-turbo instead of GPT-4
   - Claude Haiku instead of Opus
   - Gemini free tier

2. **Limit max_tokens**:
   ```python
   response = client.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=[...],
       max_tokens=100  # Limit response length
   )
   ```

3. **Cache responses during development**:
   ```python
   import json

   # Save response to file
   with open('cached_response.json', 'w') as f:
       json.dump(response.model_dump(), f)

   # Load from cache instead of calling API again
   with open('cached_response.json', 'r') as f:
       cached_data = json.load(f)
   ```

4. **Use smaller test datasets** while developing

---

## Troubleshooting

### "Authentication Error" or "Invalid API Key"

**Solution**:
1. Double-check your API key is copied correctly (no extra spaces)
2. Verify key is active in the provider's dashboard
3. Ensure billing is set up
4. For OpenAI: Make sure you've added a payment method

### "Rate Limit Exceeded"

**Solution**:
1. Wait a minute before retrying
2. Reduce frequency of requests
3. Upgrade to higher tier (OpenAI)
4. Implement exponential backoff

### "Model not found" Error

**Solution**:
1. Check model name spelling
2. Verify you have access to that model
3. Use a different model (e.g., "gpt-3.5-turbo" instead of "gpt-4")

### Colab Disconnects

**Solution**:
1. Colab times out after 90 minutes of inactivity
2. Pro version provides longer sessions
3. Save work frequently
4. Use `Ctrl+S` or `Cmd+S` to save

### Package Installation Fails

**Solution**:
```python
# Force reinstall
!pip install --upgrade --force-reinstall package-name

# Or use specific version
!pip install package-name==1.2.3
```

---

## Checklist

Before proceeding to Session 1, verify you have:

- [ ] OpenAI API key obtained and tested
- [ ] (Optional) Claude API key obtained and tested
- [ ] (Optional) Gemini API key obtained and tested
- [ ] Successfully opened a Colab notebook
- [ ] Configured API keys using Colab Secrets
- [ ] Tested API connections successfully
- [ ] Installed all required libraries
- [ ] Set spending limits on API accounts
- [ ] Understand basic Colab interface

---

## Next Steps

Once you've completed this setup and all tests pass, you're ready for:

**[Session 1: LLM Fundamentals & API Usage →](01_LLM_Fundamentals.md)**

In Session 1, you'll learn:
- How to make API calls to different LLM providers
- Understanding tokens and context windows
- Managing parameters like temperature and top_p
- Handling streaming responses
- Cost calculation and optimization

---

## Additional Resources

- [OpenAI Documentation](https://platform.openai.com/docs)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [Google AI Documentation](https://ai.google.dev/)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

---

**Setup Complete!** 🎉

You're now ready to build amazing Gen AI applications!
