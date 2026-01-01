"""
Lab 1 - Exercise 2: Token Counting
Solution for understanding and counting tokens

Learning Objectives:
- Use tiktoken to count tokens accurately
- Understand character-to-token ratio
- Visualize token boundaries
- Calculate costs based on token usage
"""

import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """
    Count tokens in text for a specific model

    Args:
        text: String to tokenize
        model: Model name (determines tokenizer)

    Returns:
        Number of tokens
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


def visualize_tokens(text, model="gpt-3.5-turbo"):
    """
    Show how text is broken into tokens

    Args:
        text: String to tokenize
        model: Model name
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    print(f"\n📝 Text: '{text}'")
    print(f"🔢 Tokens ({len(tokens)}):")

    # Decode each token to show boundaries
    for i, token in enumerate(tokens):
        decoded = encoding.decode([token])
        print(f"  [{i}] '{decoded}' (token id: {token})")


def calculate_cost(prompt_tokens, completion_tokens, model="gpt-3.5-turbo"):
    """
    Calculate API cost based on token usage

    Args:
        prompt_tokens: Input tokens
        completion_tokens: Output tokens
        model: Model name for pricing

    Returns:
        Total cost in USD
    """
    # Pricing (as of 2024)
    pricing = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # per 1K tokens
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03}
    }

    if model not in pricing:
        model = "gpt-3.5-turbo"  # Default

    input_cost = (prompt_tokens / 1000) * pricing[model]["input"]
    output_cost = (completion_tokens / 1000) * pricing[model]["output"]

    return input_cost + output_cost


# ============================================================================
# MAIN DEMO
# ============================================================================

print("=" * 70)
print("EXERCISE 2: TOKEN COUNTING")
print("=" * 70)

# Task 2.1: Basic Token Counter
print("\n📊 TASK 2.1: BASIC TOKEN ANALYSIS")
print("=" * 70)

test_texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog",
    "OpenAI GPT-4 is a large language model",
    "Machine learning is transforming artificial intelligence",
]

for text in test_texts:
    token_count = count_tokens(text)
    char_count = len(text)
    ratio = char_count / token_count if token_count > 0 else 0

    print(f"\nText: '{text}'")
    print(f"  Tokens: {token_count}")
    print(f"  Characters: {char_count}")
    print(f"  Ratio: {ratio:.2f} chars/token")

# Task 2.2: Token Visualization
print("\n\n🔍 TASK 2.2: TOKEN BREAKDOWN VISUALIZER")
print("=" * 70)

example_texts = [
    "Hello!",
    "GPT-4",
    "artificial intelligence"
]

for text in example_texts:
    visualize_tokens(text)

# Task 2.3: Cost Calculator
print("\n\n💰 TASK 2.3: COST CALCULATOR")
print("=" * 70)

scenarios = [
    {"name": "Simple question", "prompt": 50, "completion": 100},
    {"name": "Document summary", "prompt": 2000, "completion": 500},
    {"name": "Code generation", "prompt": 500, "completion": 1000},
]

print("\nGPT-3.5-Turbo Cost Estimates:")
for scenario in scenarios:
    cost = calculate_cost(
        scenario["prompt"],
        scenario["completion"],
        "gpt-3.5-turbo"
    )
    print(f"\n{scenario['name']}:")
    print(f"  Prompt: {scenario['prompt']} tokens")
    print(f"  Completion: {scenario['completion']} tokens")
    print(f"  Cost: ${cost:.6f}")

print("\n\n✅ Exercise 2 Complete!")
print("\n💡 Key Takeaways:")
print("  - Tokens ≠ Words (can be parts of words)")
print("  - Average: ~4 characters per token in English")
print("  - Special characters often = separate tokens")
print("  - Always use tiktoken for accurate counting")
print("  - Track tokens to control costs!")

# Additional: Token count for different models
print("\n\n📌 BONUS: SAME TEXT, DIFFERENT MODELS")
print("=" * 70)

sample = "The quick brown fox jumps over the lazy dog"
models = ["gpt-3.5-turbo", "gpt-4"]

for model in models:
    try:
        tokens = count_tokens(sample, model)
        print(f"{model:20} → {tokens} tokens")
    except:
        print(f"{model:20} → Model not available")
