# Interactive Quizzes & Code Challenges

## Overview

This document contains all quizzes, knowledge checks, and code challenges for the course. Each session has:
- **Knowledge Checks**: Quick questions after major concepts
- **Session Quiz**: 10 comprehensive questions
- **Code Challenges**: Practical coding exercises
- **Solutions**: Detailed explanations

---

## Session 1: LLM Fundamentals

### Knowledge Check 1.1: Understanding Tokens

**Question 1**: How many tokens is the text "Hello, world!" approximately?
- A) 2 tokens
- B) 3 tokens ✓
- C) 13 tokens
- D) 1 token

**Explanation**: "Hello", ",", " world" = 3 tokens. Punctuation and spaces often become separate tokens.

---

**Question 2**: Which parameter controls the randomness of LLM responses?
- A) max_tokens
- B) temperature ✓
- C) top_p
- D) frequency_penalty

**Explanation**: Temperature (0.0-2.0) controls randomness. Lower = more deterministic, Higher = more creative.

---

**Question 3**: What happens if you exceed the context window?
- A) The API automatically truncates input ✓
- B) The response is empty
- C) You get an error
- D) The model uses a larger window

**Explanation**: Most APIs will return an error (BadRequest). You need to manage context size manually.

---

### Session 1 Quiz (10 Questions)

**Q1**: What is a token in the context of LLMs?
- A) A security credential
- B) A unit of text processing ✓
- C) An API key
- D) A model parameter

**Answer**: B. Tokens are the smallest units that LLMs process (roughly 4 characters or 3/4 of a word in English).

---

**Q2**: Which model is most cost-effective for simple tasks?
- A) GPT-4
- B) GPT-4 Turbo
- C) GPT-3.5-turbo ✓
- D) Claude Opus

**Answer**: C. GPT-3.5-turbo at $0.50/$1.50 per 1M tokens is cheapest.

---

**Q3**: What is the purpose of the system message?
- A) To debug the application
- B) To set the behavior and persona of the AI ✓
- C) To count tokens
- D) To handle errors

**Answer**: B. System messages define how the AI should behave.

---

**Q4**: When should you use streaming responses?
- A) For very short responses
- B) For long responses to improve UX ✓
- C) Never, it's slower
- D) Only for voice applications

**Answer**: B. Streaming provides immediate feedback for long responses.

---

**Q5**: What is the typical context window for GPT-3.5-turbo?
- A) 1,024 tokens
- B) 4,096 tokens ✓
- C) 16,384 tokens
- D) 128,000 tokens

**Answer**: B. 4,096 tokens (though some versions support 16K).

---

**Q6**: How do you calculate API costs?
- A) Per request
- B) Per second
- C) Per token ✓
- D) Per word

**Answer**: C. Costs are calculated per token (input + output).

---

**Q7**: Which API parameter limits response length?
- A) context_length
- B) max_tokens ✓
- C) temperature
- D) top_p

**Answer**: B. max_tokens limits the number of tokens in the response.

---

**Q8**: What does temperature=0 mean?
- A) Fastest response
- B) Longest response
- C) Most deterministic response ✓
- D) Most creative response

**Answer**: C. Temperature=0 gives the most deterministic (predictable) responses.

---

**Q9**: What is the purpose of top_p (nucleus sampling)?
- A) To control response length
- B) To control which tokens are considered for generation ✓
- C) To set the temperature
- D) To enable streaming

**Answer**: B. top_p controls token selection probability.

---

**Q10**: How should API keys be stored?
- A) In the code
- B) In comments
- C) In environment variables ✓
- D) In a README file

**Answer**: C. Always use environment variables, never hardcode keys.

---

### Code Challenge 1.1: Token Counter

**Task**: Build a function that estimates API costs based on input text.

```python
# Your task: Complete this function
import tiktoken

def estimate_cost(text, model="gpt-3.5-turbo"):
    """
    Estimate the cost of an API call for given text.

    Args:
        text (str): Input text
        model (str): Model name

    Returns:
        dict: {
            "input_tokens": int,
            "estimated_output_tokens": int,
            "estimated_cost": float
        }
    """
    # TODO: Implement this function
    pass

# Test cases
assert estimate_cost("Hello world")["input_tokens"] == 2
assert estimate_cost("Hello world")["estimated_cost"] > 0
```

**Solution**:
```python
import tiktoken

def estimate_cost(text, model="gpt-3.5-turbo"):
    # Pricing per 1M tokens (as of Dec 2024)
    pricing = {
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00}
    }

    # Get tokenizer
    encoding = tiktoken.encoding_for_model(model)

    # Count input tokens
    input_tokens = len(encoding.encode(text))

    # Estimate output tokens (assume 2x input as rough estimate)
    estimated_output_tokens = input_tokens * 2

    # Calculate cost
    input_cost = (input_tokens / 1_000_000) * pricing[model]["input"]
    output_cost = (estimated_output_tokens / 1_000_000) * pricing[model]["output"]
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_cost": round(total_cost, 6)
    }

# Test
result = estimate_cost("Hello world")
print(f"Tokens: {result['input_tokens']}")
print(f"Estimated cost: ${result['estimated_cost']}")
```

---

### Code Challenge 1.2: Streaming Chat

**Task**: Implement a streaming chat interface.

```python
from openai import OpenAI
import os

def streaming_chat(message, history=None):
    """
    Create a streaming chat response.

    Args:
        message (str): User message
        history (list): Previous messages

    Yields:
        str: Chunks of the response
    """
    # TODO: Implement streaming chat
    pass

# Test
for chunk in streaming_chat("Tell me a short story"):
    print(chunk, end="", flush=True)
```

**Solution**:
```python
from openai import OpenAI
import os

def streaming_chat(message, history=None):
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    # Build messages
    messages = history or []
    messages.append({"role": "user", "content": message})

    # Stream response
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
        max_tokens=200
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            yield content

    # Update history
    messages.append({"role": "assistant", "content": full_response})

# Test
print("Assistant: ", end="")
for chunk in streaming_chat("Tell me a short story"):
    print(chunk, end="", flush=True)
print()
```

---

## Session 2: Prompt Engineering

### Knowledge Check 2.1: Prompt Components

**Question 1**: What are the three main components of an effective prompt?
- A) Role, Task, Examples ✓
- B) Name, Date, Output
- C) Model, Temperature, Tokens
- D) Input, Process, Output

**Answer**: A. Good prompts include the AI's role, the specific task, and examples (few-shot learning).

---

**Question 2**: What is few-shot learning?
- A) Training a model with little data
- B) Providing examples in the prompt ✓
- C) Using a small model
- D) Making multiple API calls

**Answer**: B. Few-shot learning provides examples to guide the model's responses.

---

### Code Challenge 2.1: Prompt Template System

**Task**: Build a reusable prompt template system.

```python
class PromptTemplate:
    def __init__(self, template, variables):
        """
        Create a prompt template.

        Args:
            template (str): Template string with {variable} placeholders
            variables (list): List of required variable names
        """
        # TODO: Implement
        pass

    def format(self, **kwargs):
        """
        Format the template with provided variables.

        Returns:
            str: Formatted prompt
        """
        # TODO: Implement
        pass

# Test
template = PromptTemplate(
    template="Translate the following {source_lang} to {target_lang}: {text}",
    variables=["source_lang", "target_lang", "text"]
)

prompt = template.format(
    source_lang="English",
    target_lang="Spanish",
    text="Hello, how are you?"
)

assert "English" in prompt
assert "Spanish" in prompt
```

**Solution**:
```python
class PromptTemplate:
    def __init__(self, template, variables):
        self.template = template
        self.variables = variables

    def format(self, **kwargs):
        # Validate all required variables provided
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Format template
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Invalid template variable: {e}")

# Usage
customer_service_template = PromptTemplate(
    template="""You are a {role} for {company}.

Customer question: {question}

Guidelines:
- Be {tone}
- Keep response under {max_words} words
- {additional_instructions}

Response:""",
    variables=["role", "company", "question", "tone", "max_words", "additional_instructions"]
)

prompt = customer_service_template.format(
    role="customer service representative",
    company="TechStore",
    question="How do I return a product?",
    tone="professional and empathetic",
    max_words=100,
    additional_instructions="Always cite our policy"
)

print(prompt)
```

---

## Session 3: RAG Systems

### Knowledge Check 3.1: RAG Concepts

**Question 1**: What does RAG stand for?
- A) Rapid AI Generation
- B) Retrieval-Augmented Generation ✓
- C) Random Access Gateway
- D) Recursive Algorithm Generator

**Answer**: B. Retrieval-Augmented Generation combines retrieval with generation.

---

**Question 2**: Why do we chunk documents for RAG?
- A) To save storage space
- B) To improve retrieval precision and manage context limits ✓
- C) To make embeddings faster
- D) To reduce API costs

**Answer**: B. Chunking improves retrieval precision and prevents context overflow.

---

**Question 3**: What is the purpose of embeddings?
- A) To compress text
- B) To represent text as numerical vectors for similarity search ✓
- C) To translate text
- D) To count words

**Answer**: B. Embeddings convert text to vectors for semantic similarity search.

---

### Session 3 Quiz

**Q1**: In a RAG system, when does retrieval happen?
- A) Before the user asks a question
- B) At query time, for each user question ✓
- C) Only once during setup
- D) Never, it's pre-computed

**Answer**: B. Retrieval happens dynamically for each query.

---

**Q2**: What is a good chunk size for most documents?
- A) 10-50 characters
- B) 400-800 characters ✓
- C) 5000+ characters
- D) Exactly 1 page

**Answer**: B. 400-800 characters balances context and precision.

---

**Q3**: Why include chunk overlap?
- A) To increase database size
- B) To maintain context between adjacent chunks ✓
- C) To reduce retrieval time
- D) To save money

**Answer**: B. Overlap ensures important information at chunk boundaries isn't lost.

---

**Q4**: What is semantic search?
- A) Searching by exact keyword match
- B) Searching by meaning using embeddings ✓
- C) Searching by date
- D) Searching by file size

**Answer**: B. Semantic search finds documents by meaning, not just keywords.

---

**Q5**: Which metric measures vector similarity?
- A) Euclidean distance
- B) Cosine similarity ✓
- C) Both A and B ✓
- D) Neither

**Answer**: C. Both are commonly used (cosine similarity is most popular).

---

**Q6**: What should RAG responses include?
- A) Just the answer
- B) Answer + source citations ✓
- C) Only the retrieved documents
- D) The embedding vectors

**Answer**: B. Always include sources for verification.

---

**Q7**: What is ChromaDB?
- A) A web browser
- B) A vector database ✓
- C) An embedding model
- D) An LLM

**Answer**: B. ChromaDB is a vector database for storing embeddings.

---

**Q8**: How do you improve RAG retrieval quality?
- A) Use larger chunks
- B) Add more documents
- C) Test different chunk sizes and use reranking ✓
- D) Use more expensive models

**Answer**: C. Optimization requires testing and advanced techniques.

---

**Q9**: What is a limitation of RAG?
- A) It's too fast
- B) Retrieved context might not contain the answer ✓
- C) It's too accurate
- D) It doesn't work with LLMs

**Answer**: B. RAG depends on having relevant information in the knowledge base.

---

**Q10**: When should you use RAG?
- A) For creative writing
- B) For factual Q&A about specific documents ✓
- C) For math calculations
- D) Never

**Answer**: B. RAG excels at document-based factual Q&A.

---

### Code Challenge 3.1: Build a Simple RAG System

**Task**: Create a complete RAG pipeline.

```python
class SimpleRAG:
    def __init__(self):
        """Initialize RAG system with vector database"""
        # TODO: Initialize embedding model and vector db
        pass

    def add_documents(self, documents):
        """
        Add documents to the knowledge base.

        Args:
            documents (list): List of document strings
        """
        # TODO: Chunk, embed, and store documents
        pass

    def query(self, question, top_k=3):
        """
        Answer a question using RAG.

        Args:
            question (str): User question
            top_k (int): Number of documents to retrieve

        Returns:
            dict: {
                "answer": str,
                "sources": list
            }
        """
        # TODO: Retrieve, format context, generate answer
        pass

# Test
rag = SimpleRAG()
rag.add_documents([
    "Our return policy allows 30 days for returns.",
    "Shipping takes 5-7 business days."
])

result = rag.query("What is your return policy?")
assert "30 days" in result["answer"].lower()
```

**Solution** (see Session 3 materials for complete implementation).

---

## Session 4: Function Calling

### Knowledge Check 4.1: Tool Use

**Question 1**: What is function calling?
- A) Making recursive functions
- B) Allowing LLMs to invoke external tools/APIs ✓
- C) Calling the LLM API
- D) Error handling

**Answer**: B. Function calling lets LLMs use tools to take actions.

---

**Question 2**: What must you provide for function calling?
- A) Function name only
- B) Function name, description, and parameter schema ✓
- C) Just the function code
- D) Only examples

**Answer**: B. The LLM needs complete function metadata to use it correctly.

---

### Code Challenge 4.1: Weather Tool

**Task**: Create a weather lookup tool with function calling.

```python
def get_weather(city, units="fahrenheit"):
    """
    Get current weather for a city.

    Args:
        city (str): City name
        units (str): "fahrenheit" or "celsius"

    Returns:
        dict: Weather information
    """
    # Mock implementation (use real API in production)
    return {
        "city": city,
        "temperature": 72 if units == "fahrenheit" else 22,
        "condition": "sunny",
        "units": units
    }

# TODO: Create the function definition for OpenAI API
tools = [
    {
        "type": "function",
        "function": {
            # TODO: Complete function definition
        }
    }
]

# TODO: Implement function calling flow
```

**Solution** (see Session 4 materials).

---

## Session 5-8 Quizzes

*(Similar structure for remaining sessions)*

---

## Final Comprehensive Exam (50 Questions)

### Section 1: Fundamentals (10 questions)
1. Token concepts
2. API usage
3. Cost optimization
4. Prompt engineering
5. System design

### Section 2: RAG Systems (10 questions)
6. Document processing
7. Embeddings
8. Vector search
9. Retrieval strategies
10. Evaluation

### Section 3: Agents & Tools (15 questions)
11. Function calling
12. Agent loops
13. Memory systems
14. Multi-agent orchestration
15. Error handling

### Section 4: Production (15 questions)
16. Deployment
17. Monitoring
18. Security
19. Scalability
20. Cost management

---

## Interactive Challenge: Build SupportGenie

**Final Project Assessment**:
- Functionality: 40 points
- Code Quality: 20 points
- Documentation: 10 points
- Performance: 15 points
- Innovation: 15 points

**Total**: 100 points

**Passing**: 70 points

---

## How to Use These Materials

### For Self-Study
1. Complete knowledge checks after each section
2. Take session quiz before moving to next session
3. Attempt code challenges independently
4. Check solutions only after trying

### For Instructors
- Use knowledge checks for in-class participation
- Assign code challenges as homework
- Use session quizzes for grading
- Final exam for certification

### For Practice
- Review wrong answers with explanations
- Implement code challenges from scratch
- Time yourself on quizzes
- Create your own variations

---

**Quiz yourself regularly to reinforce learning!** 🎯
