"""
Lab 1 - Exercise 5: Streaming Implementation
Solution for implementing streaming responses

Learning Objectives:
- Understand streaming vs non-streaming
- Implement real-time token display
- Handle stream chunks properly
- Measure streaming performance
"""

from openai import OpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def non_streaming_call(prompt):
    """Standard non-streaming API call"""
    print("\n🔄 NON-STREAMING (wait for complete response):")
    print("-" * 70)

    start_time = time.time()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7
    )

    elapsed = time.time() - start_time

    print(response.choices[0].message.content)
    print(f"\n⏱️ Time taken: {elapsed:.2f}s")
    print(f"📊 Tokens: {response.usage.completion_tokens}")

    return response.choices[0].message.content


def streaming_call(prompt):
    """Streaming API call with real-time display"""
    print("\n⚡ STREAMING (real-time token display):")
    print("-" * 70)

    start_time = time.time()
    full_response = ""

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7,
        stream=True  # Enable streaming
    )

    # Process each chunk as it arrives
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end='', flush=True)  # Print immediately

    elapsed = time.time() - start_time

    print(f"\n\n⏱️ Time taken: {elapsed:.2f}s")
    print(f"📊 Tokens: {len(full_response.split())}")  # Approximate

    return full_response


def streaming_with_callback(prompt, callback=None):
    """Streaming with callback function for each chunk"""
    print("\n🎯 STREAMING WITH CALLBACK:")
    print("-" * 70)

    full_response = ""
    chunk_count = 0

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            chunk_count += 1

            # Call custom callback if provided
            if callback:
                callback(content, chunk_count)
            else:
                print(content, end='', flush=True)

    print(f"\n\n📦 Total chunks received: {chunk_count}")

    return full_response


# Callback examples
def word_counter_callback(content, chunk_num):
    """Example callback: Count words as they stream"""
    print(content, end='', flush=True)
    # Could track stats here


def highlight_callback(content, chunk_num):
    """Example callback: Highlight keywords"""
    keywords = ["AI", "machine learning", "Python"]

    for keyword in keywords:
        if keyword.lower() in content.lower():
            content = content.replace(keyword, f"**{keyword}**")

    print(content, end='', flush=True)


# ============================================================================
# MAIN DEMO
# ============================================================================

print("=" * 70)
print("EXERCISE 5: STREAMING IMPLEMENTATION")
print("=" * 70)

prompt = "Explain how neural networks work in 3 sentences"

# Task 5.1: Compare Streaming vs Non-Streaming
print("\n\n📊 TASK 5.1: STREAMING VS NON-STREAMING COMPARISON")
print("=" * 70)

# Non-streaming
non_streaming_result = non_streaming_call(prompt)

# Add delay to see difference
time.sleep(1)

# Streaming
streaming_result = streaming_call(prompt)

# Task 5.2: Streaming Implementation
print("\n\n⚡ TASK 5.2: BASIC STREAMING IMPLEMENTATION")
print("=" * 70)

def basic_stream_demo():
    """Demonstrate basic streaming"""
    print("\n💬 Question: Tell me a fun fact about space")
    print("Response: ", end='')

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a fun fact about space"}],
        max_tokens=100,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)

    print("\n")

basic_stream_demo()

# Task 5.3: Streaming with Callback
print("\n\n🎯 TASK 5.3: STREAMING WITH CALLBACK")
print("=" * 70)

callback_prompt = "List 3 benefits of AI in healthcare"
streaming_with_callback(callback_prompt, callback=word_counter_callback)

# Task 5.4: Performance Comparison
print("\n\n📈 TASK 5.4: PERFORMANCE ANALYSIS")
print("=" * 70)

print("\nStreaming Benefits:")
print("  ✅ Better user experience (no waiting)")
print("  ✅ Appears faster (immediate feedback)")
print("  ✅ Can process chunks as they arrive")
print("  ✅ Good for real-time applications")

print("\nNon-Streaming Benefits:")
print("  ✅ Simpler to implement")
print("  ✅ Complete response in one piece")
print("  ✅ Easier to handle errors")
print("  ✅ Get exact token count")

print("\n\n✅ Exercise 5 Complete!")
print("\n💡 Key Takeaways:")
print("  - Streaming provides better UX (no waiting)")
print("  - Set stream=True in API call")
print("  - Process chunks with for loop")
print("  - Use flush=True for real-time display")
print("  - Streaming doesn't reduce cost or latency")
print("  - Best for chatbots and interactive apps")

# Practical Implementation Template
print("\n\n📝 STREAMING IMPLEMENTATION TEMPLATE:")
print("=" * 70)
print("""
def stream_response(prompt):
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        stream=True  # Enable streaming
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end='', flush=True)

    return full_response
""")

print("\n🎯 USE STREAMING FOR:")
print("  - Chatbots and conversational AI")
print("  - Long-form content generation")
print("  - Interactive tutorials")
print("  - Any user-facing application")

print("\n🎯 SKIP STREAMING FOR:")
print("  - Batch processing")
print("  - Backend services")
print("  - When you need complete response first")
print("  - Automated systems")
