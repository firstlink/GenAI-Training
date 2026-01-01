# Lab 2: Prompt Engineering

## 🛠️ Hands-On Lab

**Duration**: 60-90 minutes
**Difficulty**: Beginner to Intermediate
**Prerequisites**: Lab 1 completed

---

## What You'll Build

By the end of this lab, you'll have:
- ✅ Mastered prompt construction techniques
- ✅ Created reusable prompt templates
- ✅ Implemented few-shot learning patterns
- ✅ Built edge case handlers
- ✅ **Capstone**: Enhanced SupportGenie v0.2 with advanced prompting

---

## 📋 Setup

### Step 1: Verify Environment

```bash
# Make sure you completed Lab 1 setup
# Your .env should have API keys
```

### Step 2: Create Lab File

```bash
cd /path/to/your/workspace
touch lab2_prompt_engineering.py
```

### Step 3: Imports

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

✅ **Checkpoint**: Run the file - no errors should appear.

---

## Exercise 1: Understanding Prompt Quality (15 min)

**Objective**: Experience the difference between vague and specific prompts.

### Task 1A: Test Vague Prompt

```python
def test_vague_prompt():
    """Test how LLMs respond to vague prompts"""

    vague_prompt = "Tell me about returns"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": vague_prompt}
        ],
        max_tokens=150
    )

    print("=== VAGUE PROMPT ===")
    print(f"Prompt: {vague_prompt}")
    print(f"\nResponse:\n{response.choices[0].message.content}")
    print("\n" + "="*50 + "\n")

test_vague_prompt()
```

**Expected Behavior**: The response will be unfocused - it might discuss financial returns, product returns, programming return statements, etc.

### Task 1B: Test Specific Prompt

```python
def test_specific_prompt():
    """Test how LLMs respond to specific, well-structured prompts"""

    specific_prompt = """
As a customer service agent for TechStore (an electronics retailer),
explain our 30-day product return policy.

Include:
- Eligibility requirements
- Return process steps
- Timeframe for refunds

Keep it under 100 words and use a professional, helpful tone.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": specific_prompt}
        ],
        max_tokens=200
    )

    print("=== SPECIFIC PROMPT ===")
    print(f"Prompt: {specific_prompt}")
    print(f"\nResponse:\n{response.choices[0].message.content}")
    print("\n" + "="*50 + "\n")

test_specific_prompt()
```

**Expected Behavior**: Focused, relevant response specifically about product return policies.

### Task 1C: Compare Results

Run both functions and observe:
- Which response is more useful?
- Which response stays on topic?
- Which response has appropriate length?

✅ **Checkpoint**: You should see dramatically different response quality.

---

## Exercise 2: System Messages (15 min)

**Objective**: Learn how system messages control AI behavior across conversations.

### Task 2A: Build System Messages

```python
def chat_with_system_message(system_msg, user_msg):
    """Helper function to test different system messages"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=150
    )

    return response.choices[0].message.content

# Test Case 1: Vague system message
vague_system = "You are a helpful assistant."

# Test Case 2: Specific system message
specific_system = """
You are SupportGenie, a customer service AI for TechStore.

Guidelines:
- Be professional and empathetic
- Keep responses under 75 words
- Always offer to escalate if needed
- Never make up information about products or policies

Response Format:
1. Acknowledge the customer's concern
2. Provide solution or information
3. Ask if they need additional help
"""

# Test with same user message
user_message = "My order hasn't arrived yet!"

print("=== VAGUE SYSTEM MESSAGE ===")
response1 = chat_with_system_message(vague_system, user_message)
print(f"Response: {response1}\n")

print("=== SPECIFIC SYSTEM MESSAGE ===")
response2 = chat_with_system_message(specific_system, user_message)
print(f"Response: {response2}\n")
```

### Task 2B: Create Your Own System Message

**Your Turn**: Create a system message for a different role.

```python
# Example: Tech support bot
your_system_message = """
# TODO: Create a system message for a technical support bot
# that helps customers troubleshoot laptop issues.
#
# Include:
# - Role and expertise
# - Constraints (ask one question at a time)
# - Tone (patient and clear)
# - Response format
"""

# Test it
user_message = "My laptop is running really slow"
response = chat_with_system_message(your_system_message, user_message)
print(f"Your Bot's Response: {response}")
```

✅ **Checkpoint**: Your system message should produce focused, helpful technical troubleshooting responses.

---

## Exercise 3: Few-Shot Learning (20 min)

**Objective**: Use examples to guide consistent output formatting.

### Task 3A: Zero-Shot (No Examples)

```python
def extract_info_zero_shot(text):
    """Extract customer info without examples"""

    prompt = f"""
Extract the customer name and email from this message.
Return as JSON.

Message: {text}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

    return response.choices[0].message.content

# Test
test_message = "Hi, I'm John Smith. Contact me at john@email.com"
result = extract_info_zero_shot(test_message)
print("Zero-Shot Result:")
print(result)
print()
```

**Note**: Format may be inconsistent across different inputs.

### Task 3B: Few-Shot (With Examples)

```python
def extract_info_few_shot(text):
    """Extract customer info with examples for consistency"""

    prompt = f"""
Extract customer name and email from text. Return as JSON.

Examples:

Text: "My name is Alice Johnson, email alice@test.com"
Output: {{"name": "Alice Johnson", "email": "alice@test.com"}}

Text: "I'm Bob Lee (bob.lee@company.com)"
Output: {{"name": "Bob Lee", "email": "bob.lee@company.com"}}

Text: "Contact Sarah Martinez at s.martinez@mail.com for details"
Output: {{"name": "Sarah Martinez", "email": "s.martinez@mail.com"}}

Now extract from:
Text: "{text}"
Output:
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.3  # Lower temperature for more consistent formatting
    )

    return response.choices[0].message.content

# Test with multiple inputs
test_messages = [
    "Hi, I'm John Smith. Contact me at john@email.com",
    "This is Jennifer Wu, reach me at jwu@company.org",
    "My email is mike.brown@test.net - Mike Brown"
]

print("Few-Shot Results:")
for msg in test_messages:
    result = extract_info_few_shot(msg)
    print(f"Input: {msg}")
    print(f"Output: {result}")
    print()
```

✅ **Checkpoint**: Few-shot results should have consistent JSON format.

### Task 3C: Your Turn - Sentiment Classification

**Challenge**: Create a few-shot prompt for classifying customer message sentiment.

```python
def classify_sentiment(message):
    """
    TODO: Create a few-shot prompt to classify sentiment as:
    - positive
    - negative
    - neutral

    Provide 3-4 examples in your prompt.
    """

    prompt = f"""
# Your few-shot prompt here
# Include examples of positive, negative, and neutral messages
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.3
    )

    return response.choices[0].message.content

# Test cases
test_sentiments = [
    "I love this product! Works perfectly.",
    "Terrible experience. Product broke after 2 days.",
    "The item arrived. It's okay, I guess.",
    "WHERE IS MY ORDER?! I've been waiting 3 weeks!"
]

for msg in test_sentiments:
    sentiment = classify_sentiment(msg)
    print(f"Message: {msg}")
    print(f"Sentiment: {sentiment}\n")
```

✅ **Checkpoint**: Should correctly classify all test cases.

---

## Exercise 4: Chain-of-Thought Prompting (15 min)

**Objective**: Use step-by-step reasoning for complex problems.

### Task 4A: Problem Without CoT

```python
def calculate_total_no_cot(items, discount_percent, shipping):
    """Calculate order total WITHOUT chain-of-thought"""

    prompt = f"""
Calculate the total cost:
- {items} items at $50 each
- {discount_percent}% discount
- ${shipping} shipping

Total:
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )

    return response.choices[0].message.content

result = calculate_total_no_cot(3, 20, 5)
print("Without CoT:")
print(result)
print()
```

### Task 4B: Problem With CoT

```python
def calculate_total_with_cot(items, discount_percent, shipping):
    """Calculate order total WITH chain-of-thought reasoning"""

    prompt = f"""
Calculate the total cost step by step:

Problem:
- {items} items at $50 each
- {discount_percent}% discount code
- ${shipping} shipping

Please solve this step by step:

Step 1: Calculate subtotal (items × price)
Step 2: Apply discount
Step 3: Add shipping
Step 4: Provide final total

Solution:
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    return response.choices[0].message.content

result = calculate_total_with_cot(3, 20, 5)
print("With CoT:")
print(result)
print()
```

**Observe**: The CoT version shows its work, making it easier to verify correctness.

### Task 4C: Your Turn - Troubleshooting with CoT

```python
def troubleshoot_with_cot(issue):
    """
    TODO: Create a CoT prompt for troubleshooting technical issues.

    The prompt should ask the LLM to:
    1. Identify the problem type
    2. List possible causes
    3. Suggest diagnostic questions
    4. Recommend solutions
    """

    prompt = f"""
# Your CoT troubleshooting prompt here
Issue: {issue}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    return response.choices[0].message.content

# Test
issue = "Customer's laptop won't turn on"
solution = troubleshoot_with_cot(issue)
print(solution)
```

✅ **Checkpoint**: Should see step-by-step reasoning for troubleshooting.

---

## Exercise 5: Prompt Templates (20 min)

**Objective**: Build reusable, maintainable prompt templates.

### Task 5A: Create Template Class

```python
class PromptTemplate:
    """Reusable prompt template with variable substitution"""

    def __init__(self, template, variables):
        self.template = template
        self.variables = variables

    def format(self, **kwargs):
        """Format template with provided values"""

        # Validate all required variables are provided
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        return self.template.format(**kwargs)

# Test the class
product_template = PromptTemplate(
    template="""
As a product expert for {company}, answer this question about {product}.

Question: {question}

Guidelines:
- Be accurate and detailed
- Mention key features
- Keep response under {max_words} words
- Use {tone} tone

Answer:
    """,
    variables=["company", "product", "question", "max_words", "tone"]
)

# Use the template
prompt = product_template.format(
    company="TechStore",
    product="iPhone 15 Pro",
    question="What's the battery life?",
    max_words=50,
    tone="professional"
)

print("Generated Prompt:")
print(prompt)
print()
```

### Task 5B: Create Multiple Templates

```python
# Template 1: Customer Service Response
cs_template = PromptTemplate(
    template="""
You are a customer service agent for {company}.

Customer Issue: {issue}
Customer Sentiment: {sentiment}

Response Guidelines:
- Acknowledge their {sentiment} feeling
- Provide {solution_type}
- Keep under {max_words} words
- Use {tone} tone

Your Response:
    """,
    variables=["company", "issue", "sentiment", "solution_type",
               "max_words", "tone"]
)

# Template 2: Email Classification
email_template = PromptTemplate(
    template="""
Classify this customer email into categories: {categories}

Email: {email_text}

Classification Guidelines:
- Primary category: Most relevant category
- Urgency: {urgency_levels}
- Confidence: 0-100%

Return as JSON:
{{
  "primary_category": "",
  "urgency": "",
  "confidence": 0
}}
    """,
    variables=["categories", "email_text", "urgency_levels"]
)

# Test CS Template
cs_prompt = cs_template.format(
    company="TechStore",
    issue="Order delayed by 5 days",
    sentiment="frustrated",
    solution_type="explanation and resolution",
    max_words=75,
    tone="empathetic"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": cs_prompt}],
    max_tokens=150
)

print("Template Response:")
print(response.choices[0].message.content)
```

### Task 5C: Your Turn - Create a Template

```python
# TODO: Create a template for product recommendations
#
# Variables should include:
# - customer_preferences
# - budget_range
# - product_category
# - max_recommendations
# - output_format

recommendation_template = PromptTemplate(
    template="""
# Your template here
    """,
    variables=[]  # List your variables
)

# Test your template
```

✅ **Checkpoint**: Your template should be reusable and validate required inputs.

---

## Exercise 6: Edge Case Handling (15 min)

**Objective**: Build robust prompts that handle unexpected inputs.

### Task 6A: Handle Empty/Unclear Input

```python
def handle_unclear_input():
    """System message that handles unclear customer messages"""

    system_message = """
You are a customer support assistant for TechStore.

EDGE CASE HANDLING:

If the message is empty, unclear, or lacks detail:
- Do NOT make assumptions
- Ask specific clarifying questions
- Be polite and helpful

Example:
Customer: "It doesn't work"
You: "I'm here to help! To assist you better, could you tell me:
1. Which product is having the issue?
2. What specifically isn't working?
3. What happens when you try to use it?"
"""

    # Test with unclear inputs
    unclear_messages = [
        "help",
        "it's broken",
        "???",
        "need assistance"
    ]

    for msg in unclear_messages:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": msg}
            ],
            max_tokens=150
        )

        print(f"Customer: {msg}")
        print(f"Assistant: {response.choices[0].message.content}")
        print("\n" + "="*50 + "\n")

handle_unclear_input()
```

### Task 6B: Handle Out-of-Scope Questions

```python
def handle_out_of_scope():
    """Handle questions outside the bot's domain"""

    system_message = """
You are a customer support assistant for TechStore (electronics retailer).

SCOPE: You help with:
- Product questions
- Order tracking
- Returns and warranty
- Technical support

OUT OF SCOPE: If asked about topics like politics, medical advice,
personal counseling, or other unrelated topics:

Response Template:
"I'm specialized in helping with TechStore products and orders. For
{topic}, I'd recommend consulting {appropriate_resource}. How can I
help you with your TechStore needs today?"
"""

    out_of_scope_questions = [
        "What do you think about the upcoming election?",
        "Can you give me medical advice for my headache?",
        "What's the meaning of life?",
        "Can you help me with my math homework?"
    ]

    for question in out_of_scope_questions:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ],
            max_tokens=100
        )

        print(f"Question: {question}")
        print(f"Response: {response.choices[0].message.content}")
        print()

handle_out_of_scope()
```

### Task 6C: Handle Multiple Questions

```python
def handle_multiple_questions():
    """Handle customer messages with multiple questions"""

    system_message = """
You are a customer support assistant.

MULTIPLE QUESTIONS: If a customer asks multiple questions:

Option 1: If related, address them in order
Option 2: If unrelated, ask which is most urgent
Option 3: Acknowledge all and address most critical first

Always structure your response clearly with numbered points.
"""

    multi_question = """
I have several questions:
1. Where is my order ORD-12345?
2. Can I return an opened item?
3. Do you sell laptops under $500?
4. What's your warranty policy?
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": multi_question}
        ],
        max_tokens=250
    )

    print("Multiple Questions Test:")
    print(response.choices[0].message.content)

handle_multiple_questions()
```

✅ **Checkpoint**: Bot should gracefully handle all edge cases.

---

## Exercise 7: Tone and Style Control (15 min)

**Objective**: Master different communication tones for different situations.

### Task 7A: Compare Tones

```python
def test_different_tones():
    """Test same message with different tones"""

    customer_message = "My order is 3 days late!"

    tones = {
        "professional": """
You are a professional customer service representative.
Use formal language, proper grammar, and business-appropriate tone.
        """,

        "friendly": """
You are a warm, friendly customer service rep.
Use contractions, casual language, and phrases like "Happy to help!"
Be conversational but still professional.
        """,

        "empathetic": """
You are an empathetic customer service rep focused on emotional connection.
Acknowledge frustrations, show genuine concern, use phrases like
"I understand how frustrating..." and "That must be disappointing..."
        """,

        "technical": """
You are a precise, detail-oriented support specialist.
Use clear, specific terminology. Provide exact procedures and systems.
Focus on facts and processes.
        """
    }

    print(f"Customer: {customer_message}\n")

    for tone_name, system_msg in tones.items():
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": customer_message}
            ],
            max_tokens=100
        )

        print(f"=== {tone_name.upper()} TONE ===")
        print(response.choices[0].message.content)
        print()

test_different_tones()
```

### Task 7B: Choose Appropriate Tone

**Scenario-Based Tone Selection**

```python
def choose_tone_for_scenario(scenario, customer_message):
    """Choose appropriate tone based on scenario"""

    tone_mapping = {
        "frustrated_customer": "empathetic",
        "quick_question": "friendly",
        "technical_issue": "technical",
        "business_client": "professional"
    }

    tone_prompts = {
        "empathetic": "Be understanding and compassionate. Acknowledge emotions.",
        "friendly": "Be warm and approachable. Keep it light and helpful.",
        "technical": "Be precise and detailed. Focus on step-by-step solutions.",
        "professional": "Be formal and business-appropriate."
    }

    chosen_tone = tone_mapping.get(scenario, "professional")

    system_message = f"""
You are a customer support assistant.
Tone: {tone_prompts[chosen_tone]}
Keep responses under 75 words.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": customer_message}
        ],
        max_tokens=150
    )

    return response.choices[0].message.content

# Test scenarios
scenarios = [
    ("frustrated_customer", "This is the THIRD time my order was wrong!"),
    ("quick_question", "What time do you close?"),
    ("technical_issue", "Error code 0x8007045D when installing"),
    ("business_client", "Please provide vendor specifications for procurement")
]

for scenario, message in scenarios:
    print(f"Scenario: {scenario}")
    print(f"Message: {message}")
    response = choose_tone_for_scenario(scenario, message)
    print(f"Response: {response}")
    print("\n" + "="*50 + "\n")
```

✅ **Checkpoint**: Each response should match the appropriate tone for the scenario.

---

## 🎯 Capstone Project: SupportGenie v0.2 (30 min)

**Objective**: Build an enhanced chatbot with all prompt engineering techniques.

### Requirements

Your enhanced SupportGenie must:
1. ✅ Use a comprehensive system message
2. ✅ Include prompt templates for common scenarios
3. ✅ Handle edge cases gracefully
4. ✅ Adapt tone based on customer sentiment
5. ✅ Use few-shot examples for consistent formatting
6. ✅ Implement chain-of-thought for complex issues

### Starter Code

```python
class SupportGenieV2:
    """
    SupportGenie Version 0.2
    Enhanced with advanced prompt engineering
    """

    SYSTEM_MESSAGE = """
You are SupportGenie, an expert AI customer support assistant for TechStore.

IDENTITY & EXPERTISE:
- Experienced customer service professional
- Expert in TechStore products, policies, and procedures
- Trained in empathetic communication
- Solution-focused problem solver

YOUR CAPABILITIES:
✓ Answer product questions
✓ Explain policies (returns, shipping, warranty)
✓ Troubleshoot common issues
✓ Track orders
✓ Create support tickets
✓ Escalate to human agents

YOUR CONSTRAINTS:
✗ Never make up information
✗ Never share customer data
✗ Never engage with hostile behavior
✗ Never discuss topics outside TechStore support

RESPONSE GUIDELINES:

1. ACKNOWLEDGMENT (Always start here)
   - Recognize the customer's concern
   - Show empathy for their situation

2. SOLUTION (Core of your response)
   - Provide clear, actionable information
   - Break down complex steps
   - Cite relevant policies when applicable

3. NEXT STEPS (Always end here)
   - Ask if they need additional help
   - Offer to escalate if needed

TONE & STYLE:
- Professional yet warm and approachable
- Patient and understanding
- Clear and concise (under 150 words)
- Use customer's name if provided
- Avoid jargon, explain technical terms

EDGE CASE HANDLING:

If unclear:
"I want to make sure I help you with the right information. Could you
clarify [specific detail]?"

If out of scope:
"I specialize in TechStore products and support. For [topic], I'd
recommend [appropriate resource]. How else can I help with your
TechStore needs?"

If unable to resolve:
"I want to make sure you get the best help possible. Would you like
me to escalate this to a specialized support agent?"

SENTIMENT AWARENESS:
- If customer seems frustrated → Lead with extra empathy
- If urgent language (CAPS, "!!!" ) → Acknowledge urgency
- If confused → Ask clarifying questions patiently
- If satisfied → Confirm resolution and offer future help
    """

    # Prompt templates for common scenarios
    TEMPLATES = {
        "order_tracking": PromptTemplate(
            template="""
Customer is asking about order status.

Order Details:
- Order Number: {order_number}
- Status: {status}
- Expected Delivery: {delivery_date}

Provide a clear update with empathy and next steps.
            """,
            variables=["order_number", "status", "delivery_date"]
        ),

        "return_request": PromptTemplate(
            template="""
Customer wants to return: {product}
Purchase Date: {purchase_date}
Reason: {reason}

Our policy: 30-day returns for unused items in original packaging.

Guide them through the return process if eligible.
            """,
            variables=["product", "purchase_date", "reason"]
        ),

        "technical_support": PromptTemplate(
            template="""
Customer has technical issue: {issue}
Product: {product}

Use chain-of-thought troubleshooting:
1. Identify likely causes
2. Ask one diagnostic question
3. Provide clear next step
            """,
            variables=["issue", "product"]
        )
    }

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = [
            {"role": "system", "content": self.SYSTEM_MESSAGE}
        ]

    def detect_sentiment(self, message):
        """
        Detect customer sentiment using few-shot learning

        TODO: Implement sentiment detection with few-shot examples
        Returns: 'positive', 'negative', 'neutral', or 'urgent'
        """
        prompt = f"""
Classify sentiment:

Message: "I love this product!" → positive
Message: "This is terrible!" → negative
Message: "WHERE IS MY ORDER?!" → urgent
Message: "It's okay I guess" → neutral

Message: "{message}" →
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    def detect_intent(self, message):
        """
        Detect customer intent (order_tracking, return, technical_support, general)

        TODO: Implement intent detection with few-shot examples
        """
        prompt = f"""
Classify intent:

"Where is my order?" → order_tracking
"I want to return this" → return
"It won't turn on" → technical_support
"What are your hours?" → general

"{message}" →
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    def chat(self, message, context=None):
        """
        Enhanced chat with sentiment and intent detection

        Args:
            message: Customer's message
            context: Optional dict with order info, product details, etc.
        """
        # Detect sentiment and intent
        sentiment = self.detect_sentiment(message)
        intent = self.detect_intent(message)

        print(f"[Debug] Detected Sentiment: {sentiment}, Intent: {intent}")

        # Build enhanced message with context
        if context:
            enhanced_message = f"""
Customer Message: {message}
Detected Sentiment: {sentiment}
Detected Intent: {intent}

Additional Context:
{context}
            """
        else:
            enhanced_message = f"""
Customer Message: {message}
Detected Sentiment: {sentiment}
Detected Intent: {intent}
            """

        self.conversation_history.append({
            "role": "user",
            "content": enhanced_message
        })

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.conversation_history,
            temperature=0.7,
            max_tokens=500
        )

        assistant_message = response.choices[0].message.content

        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def reset(self):
        """Start a new conversation"""
        self.conversation_history = [
            {"role": "system", "content": self.SYSTEM_MESSAGE}
        ]


# Test your SupportGenie v0.2
def test_support_genie_v2():
    """Test the enhanced chatbot"""

    api_key = os.getenv("OPENAI_API_KEY")
    bot = SupportGenieV2(api_key)

    # Test Case 1: Frustrated customer with order issue
    print("="*60)
    print("TEST 1: Frustrated Customer")
    print("="*60)
    response = bot.chat(
        "WHERE IS MY ORDER?! I ordered 2 weeks ago!",
        context="Order #ORD-12345, Status: In Transit, Expected: Tomorrow"
    )
    print(f"Bot: {response}\n")

    # Test Case 2: Return request
    print("="*60)
    print("TEST 2: Return Request")
    print("="*60)
    bot.reset()
    response = bot.chat(
        "I'd like to return the laptop I bought last week",
        context="Product: Dell XPS 15, Purchase Date: 7 days ago, Condition: Unopened"
    )
    print(f"Bot: {response}\n")

    # Test Case 3: Technical support
    print("="*60)
    print("TEST 3: Technical Issue")
    print("="*60)
    bot.reset()
    response = bot.chat(
        "My new headphones won't connect to my phone",
        context="Product: Sony WH-1000XM5, Purchased: Yesterday"
    )
    print(f"Bot: {response}\n")

    # Test Case 4: Unclear input (edge case)
    print("="*60)
    print("TEST 4: Unclear Input (Edge Case)")
    print("="*60)
    bot.reset()
    response = bot.chat("help")
    print(f"Bot: {response}\n")

    # Test Case 5: Out of scope (edge case)
    print("="*60)
    print("TEST 5: Out of Scope (Edge Case)")
    print("="*60)
    bot.reset()
    response = bot.chat("What do you think about the economy?")
    print(f"Bot: {response}\n")

# Run the tests
test_support_genie_v2()
```

### Your Tasks

1. **Complete the sentiment detection** with more examples
2. **Complete the intent detection** with more categories
3. **Add at least 2 more test cases**
4. **Customize the system message** for your use case
5. **Add a new template** for a different scenario

✅ **Checkpoint**: Run all tests - bot should handle each scenario appropriately.

---

## 🏆 Extension Challenges

### Challenge 1: Multi-Language Support

Add language detection and response in customer's language:

```python
def detect_language(message):
    """Detect message language and respond accordingly"""
    # TODO: Implement language detection
    pass
```

### Challenge 2: Conversation Summary

After a conversation, generate a summary for the support ticket:

```python
def summarize_conversation(self):
    """Generate summary of conversation for ticket"""
    # TODO: Create a summary of the entire conversation
    pass
```

### Challenge 3: Response Quality Scoring

Score bot responses for quality:

```python
def score_response(self, customer_msg, bot_response):
    """
    Score response on:
    - Empathy (1-5)
    - Accuracy (1-5)
    - Completeness (1-5)
    """
    # TODO: Implement response scoring
    pass
```

---

## 📝 Key Takeaways

After completing this lab, you should understand:

✅ **Specificity matters** - Vague prompts → vague responses
✅ **System messages control behavior** - Set the tone for entire conversations
✅ **Examples improve consistency** - Few-shot learning ensures reliable formatting
✅ **Templates improve maintainability** - Reusable prompts save time
✅ **Edge cases must be handled** - Real users send unexpected inputs
✅ **Tone matches the situation** - Adapt communication style to customer needs
✅ **Chain-of-thought improves reasoning** - Step-by-step logic is more reliable

---

## 🔍 Troubleshooting

**Issue**: Responses are too verbose

**Solution**: Add explicit length constraints:
```python
"Keep your response under 75 words."
```

**Issue**: Bot goes off-topic

**Solution**: Strengthen system message constraints:
```python
"You ONLY handle topics related to TechStore products and orders."
```

**Issue**: Inconsistent formatting

**Solution**: Use few-shot examples with exact format:
```python
"Return in this exact format: {...}"
```

**Issue**: Bot makes up information

**Solution**: Add strong constraints:
```python
"If you don't know, say 'I don't have that information' and offer to escalate."
```

---

## 🎓 What's Next?

You've completed Lab 2! You now have:
- ✅ Deep understanding of prompt engineering
- ✅ Reusable prompt templates
- ✅ Enhanced SupportGenie v0.2

**Next Lab**: Lab 3 - Document Processing & Embeddings
Learn how to give your bot access to a knowledge base!

---

## 📚 Additional Resources

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Learn Prompting](https://learnprompting.org/)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)

---

**Lab 2 Complete!** 🎉
[← Back to Learning Material](learning.md) | [Next: Lab 3 →](../Lab3-Document-Processing/learning.md)
