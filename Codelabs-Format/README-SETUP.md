# GenAI Training Labs - Python Setup

## Installation Complete ✅

All Python dependencies for Labs 1-7 have been successfully installed in a virtual environment.

## Virtual Environment Location

```
GenAI-Training/Codelabs-Format/venv/
```

## Installed Packages

### Core LLM API Clients
- **openai** (2.16.0) - OpenAI GPT models
- **anthropic** (0.76.0) - Anthropic Claude models
- **google-generativeai** (0.8.6) - Google Gemini models
- **tiktoken** (0.12.0) - OpenAI tokenizer

### Embeddings & Vector Database
- **sentence-transformers** (5.2.2) - Semantic embeddings
- **chromadb** (1.4.1) - Vector database

### Utilities
- **python-dotenv** (1.2.1) - Environment variables
- **numpy** (2.4.1) - Numerical computing

### Supporting Libraries
- **scipy** (1.17.0) - Scientific computing
- **scikit-learn** (1.8.0) - Machine learning utilities
- **torch** (2.10.0) - PyTorch (required by sentence-transformers)

## Usage Instructions

### 1. Activate the Virtual Environment

Before running any lab scripts, activate the virtual environment:

```bash
cd GenAI-Training/Codelabs-Format
source venv/bin/activate
```

You'll see `(venv)` appear in your terminal prompt.

### 2. Run Lab Scripts

Once activated, you can run any lab Python file:

```bash
# Lab 1 - LLM Fundamentals
python Lab1-LLM-Fundamentals/solutions/exercise1_openai.py
python Lab1-LLM-Fundamentals/solutions/exercise1_claude.py
python Lab1-LLM-Fundamentals/solutions/exercise1_gemini.py

# Lab 2 - Prompt Engineering
python Lab2-Prompt-Engineering/solutions/exercise1_prompt_quality.py

# Lab 3 - Document Processing
python Lab3-Document-Processing/solutions/exercise1_document_loading.py

# Lab 4 - Semantic Search
python Lab4-Semantic-Search/solutions/exercise1_basic_search.py

# Lab 5 - RAG Pipeline
python Lab5-RAG-Pipeline/solutions/exercise1_basic_rag.py

# Lab 6 - AI Agents
python Lab6-AI-Agents/solutions/exercise1_calculator_agents.py

# Lab 7 - Agent Memory
python Lab7-Agent-Memory/solutions/all_exercises.py
```

### 3. Environment Variables

Make sure to set up your API keys in a `.env` file for each lab that requires them:

```bash
# Example .env file
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
```

### 4. Deactivate the Virtual Environment

When you're done working:

```bash
deactivate
```

## Lab Dependencies by Lab

### Lab 1 - LLM Fundamentals
- openai, anthropic, google-generativeai, tiktoken, python-dotenv

### Lab 2 - Prompt Engineering
- openai, python-dotenv

### Lab 3 - Document Processing
- sentence-transformers, chromadb, numpy

### Lab 4 - Semantic Search
- sentence-transformers, chromadb, numpy

### Lab 5 - RAG Pipeline
- openai, anthropic, sentence-transformers, chromadb, numpy, python-dotenv

### Lab 6 - AI Agents
- openai, anthropic, python-dotenv

### Lab 7 - Agent Memory
- openai, chromadb, python-dotenv

## Reinstalling Dependencies

If you need to reinstall or update dependencies:

```bash
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

## Troubleshooting

### Virtual Environment Not Activating
Make sure you're in the correct directory:
```bash
cd /Users/firstlinkconsultingllc/udemy/Training/GenAI-Training/Codelabs-Format
```

### Import Errors
Always activate the virtual environment before running scripts:
```bash
source venv/bin/activate
```

### API Key Errors
Check that your `.env` file exists and contains valid API keys.
