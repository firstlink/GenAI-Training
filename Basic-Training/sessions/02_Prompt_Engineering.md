# Session 2: Advanced Prompt Engineering

**Duration**: 75 minutes
**Difficulty**: Beginner to Intermediate
**Colab Notebook**: [02_Prompt_Engineering.ipynb](../notebooks/02_Prompt_Engineering.ipynb)

---

## Learning Objectives

By the end of this session, you will:
- 🎯 Master the principles of effective prompt engineering
- 🎯 Use system messages to control AI behavior
- 🎯 Implement few-shot learning with examples
- 🎯 Apply chain-of-thought prompting for reasoning
- 🎯 Create reusable prompt templates
- 🎯 Handle edge cases and ambiguous inputs
- 🎯 Make SupportGenie significantly smarter

---

## Capstone Project: Session 2 Build

**What You'll Build**: Enhanced SupportGenie with intelligent prompting
- Professional response formatting
- Context-aware replies
- Edge case handling
- Consistent tone and style
- Template-based responses

---

## Part 1: What is Prompt Engineering?

### Definition

**Prompt Engineering** is the art and science of crafting inputs to get desired outputs from LLMs.

### Why It Matters

```
Bad Prompt:
"Tell me about returns"

LLM Output: "Returns can refer to many things: financial returns,
product returns, return statements in programming..."

Good Prompt:
"As a customer service agent, explain our 30-day product return
policy. Include eligibility requirements and the return process.
Keep it under 100 words."

LLM Output: "Our 30-day return policy allows you to return unused
products in original packaging within 30 days of purchase. To
initiate a return, contact support@company.com with your order
number. We'll provide a prepaid return label. Refunds are processed
within 5-7 business days after we receive the item..."
```

**The difference?** Specificity, context, and structure.

---

## Part 2: The Anatomy of a Good Prompt

### Core Components

```
┌─────────────────────────────────────────┐
│  1. ROLE/PERSONA                        │
│     "You are a customer service agent"  │
├─────────────────────────────────────────┤
│  2. CONTEXT                             │
│     "For TechStore, an electronics..."  │
├─────────────────────────────────────────┤
│  3. TASK                                │
│     "Explain our return policy..."      │
├─────────────────────────────────────────┤
│  4. CONSTRAINTS                         │
│     "Keep it under 100 words"           │
├─────────────────────────────────────────┤
│  5. FORMAT                              │
│     "Use bullet points"                 │
├─────────────────────────────────────────┤
│  6. EXAMPLES (optional)                 │
│     "Like this: ..."                    │
├─────────────────────────────────────────┤
│  7. TONE                                │
│     "Be professional and empathetic"    │
└─────────────────────────────────────────┘
```

### Example: Complete Prompt

```python
prompt = """
Role: You are an expert technical support agent for TechStore.

Context: TechStore sells electronics and provides warranty support.
You have access to our knowledge base and can create support tickets.

Task: Help the customer troubleshoot their laptop that won't turn on.

Constraints:
- Ask one diagnostic question at a time
- Keep responses under 75 words
- Use simple, non-technical language
- Be patient and empathetic

Format:
1. Acknowledge the issue
2. Ask a diagnostic question
3. Explain why you're asking

Tone: Professional, patient, and reassuring

Customer Message: My laptop won't turn on!
"""
```

---

## Part 3: System Messages - Setting Behavior

### What Are System Messages?

System messages define the AI's **persistent behavior** across the conversation.

### Structure

```python
messages = [
    {
        "role": "system",
        "content": "Your instructions here"  # Sets behavior
    },
    {
        "role": "user",
        "content": "User's message"
    }
]
```

### Example: Customer Service System Message

```python
system_message = """
You are SupportGenie, a customer support AI for TechStore.

IDENTITY:
- Professional customer service representative
- Knowledgeable about products, policies, shipping
- Empathetic and solution-focused

CAPABILITIES:
- Answer questions about products and policies
- Help with order tracking
- Create support tickets
- Escalate to human agents when needed

CONSTRAINTS:
- Keep responses under 100 words
- Never make up information
- Always cite sources when referencing policies
- Admit when you don't know something

TONE:
- Professional but friendly
- Empathetic to customer concerns
- Solution-oriented
- Patient and clear

RESPONSE FORMAT:
1. Acknowledge customer's concern
2. Provide solution or information
3. Ask if they need additional help
"""
```

### ❌ Bad vs ✅ Good System Messages

**❌ Too Vague:**
```python
"You are a helpful assistant."
```

**✅ Specific and Structured:**
```python
"""You are a customer service agent for TechStore.
Guidelines:
- Be professional and empathetic
- Keep responses under 100 words
- Always offer to escalate if needed
- Never make up information"""
```

---

## Part 4: Few-Shot Learning

### What is Few-Shot Learning?

Providing **examples** in the prompt to guide the model's responses.

### Zero-Shot (No Examples)

```python
prompt = "Extract the customer name and email from this text"
text = "Hi, I'm John Smith. Contact me at john@email.com"

# Output may vary in format
```

### Few-Shot (With Examples)

```python
prompt = """
Extract customer name and email from text. Return as JSON.

Examples:

Text: "My name is Alice Johnson, email alice@test.com"
Output: {"name": "Alice Johnson", "email": "alice@test.com"}

Text: "I'm Bob Lee (bob.lee@company.com)"
Output: {"name": "Bob Lee", "email": "bob.lee@company.com"}

Now extract from:
Text: "Hi, I'm John Smith. Contact me at john@email.com"
Output:
"""

# Output will consistently be JSON format
```

### Practical Example: Sentiment Classification

```python
def sentiment_few_shot(text):
    prompt = f"""
Classify the sentiment of customer messages.

Examples:

Message: "I love this product! Works perfectly."
Sentiment: positive

Message: "Terrible experience. Product broke after 2 days."
Sentiment: negative

Message: "The item arrived. It's okay, I guess."
Sentiment: neutral

Message: "WHERE IS MY ORDER?! I've been waiting 3 weeks!"
Sentiment: negative

Now classify:
Message: "{text}"
Sentiment:
"""
    return prompt
```

---

## Part 5: Chain-of-Thought Prompting

### What is Chain-of-Thought (CoT)?

Asking the LLM to **show its reasoning** step-by-step.

### Without CoT

```
Question: A customer ordered 3 items at $50 each. They have a 20%
discount code and $5 shipping. What's the total?

Answer: $125
```
(May be wrong, no reasoning shown)

### With CoT

```
Question: A customer ordered 3 items at $50 each. They have a 20%
discount code and $5 shipping. What's the total?

Let me solve this step by step:

Step 1: Calculate subtotal
3 items × $50 = $150

Step 2: Apply 20% discount
$150 × 0.20 = $30 discount
$150 - $30 = $120

Step 3: Add shipping
$120 + $5 = $125

Answer: $125
```
(Correct, with clear reasoning)

### Implementation

```python
cot_prompt = """
Solve this step by step:

1. Break down the problem
2. Show each calculation
3. Explain your reasoning
4. Provide the final answer

Problem: {problem}

Solution:
"""
```

---

## Part 6: Prompt Templates

### Why Use Templates?

- **Consistency**: Same format every time
- **Reusability**: Write once, use many times
- **Maintainability**: Update in one place
- **Testing**: Easy to A/B test variations

### Template System

```python
class PromptTemplate:
    def __init__(self, template, variables):
        self.template = template
        self.variables = variables

    def format(self, **kwargs):
        # Validate required variables
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        return self.template.format(**kwargs)

# Define template
product_query_template = PromptTemplate(
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

# Use template
prompt = product_query_template.format(
    company="TechStore",
    product="iPhone 15 Pro",
    question="What's the battery life?",
    max_words=50,
    tone="professional"
)
```

---

## Part 7: Handling Edge Cases

### Common Edge Cases

1. **Empty Input**
2. **Ambiguous Questions**
3. **Out-of-Scope Queries**
4. **Hostile/Inappropriate Input**
5. **Multiple Questions**

### Strategy: Defensive Prompting

```python
system_message = """
You are a customer support assistant.

HANDLING EDGE CASES:

If the message is empty or unclear:
- Ask for clarification politely
- Example: "I'd be happy to help! Could you please provide more details?"

If the question is out of scope (politics, personal advice, etc.):
- Politely redirect to your expertise
- Example: "I'm here to help with TechStore products and orders.
  For that topic, I'd recommend consulting a specialist."

If you encounter inappropriate content:
- Do not engage with hostility
- Remain professional
- Example: "I'm here to assist with your TechStore needs. How can
  I help you with your order or product questions?"

If there are multiple questions:
- Address them one by one
- Or ask which is most urgent
"""
```

### Example: Ambiguity Handling

```python
# Ambiguous input: "it doesn't work"

prompt = """
When a customer says something is broken but doesn't specify what:

Bad Response:
"I'm sorry it's not working."

Good Response:
"I'm sorry you're experiencing issues. To help you better, could you
tell me:
1. Which product is having the problem?
2. What specifically isn't working?
3. When did this start?

This will help me provide the right solution for you."
"""
```

---

## Part 8: Tone and Style Control

### Setting the Right Tone

```python
# Professional Tone
"""Be professional, clear, and respectful. Use complete sentences
and proper grammar."""

# Friendly Tone
"""Be warm and conversational. Use contractions (you're, we'll) and
friendly phrases like 'Happy to help!' """

# Empathetic Tone
"""Show understanding and compassion. Acknowledge frustrations. Use
phrases like 'I understand how frustrating that must be.' """

# Technical Tone
"""Be precise and detailed. Use technical terminology when appropriate.
Provide step-by-step instructions."""
```

### Example: Tone Comparison

**Same Message, Different Tones**

```
Customer: "My order is late!"

Professional Tone:
"I apologize for the delay in your order. I will investigate the
status immediately and provide you with an update."

Friendly Tone:
"Oh no, I'm so sorry your order is running late! Let me check on
that for you right away. 😊"

Empathetic Tone:
"I completely understand how frustrating it is when an order doesn't
arrive on time. Let me look into this for you and see what we can
do to make this right."

Technical Tone:
"Order delay detected. Initiating tracking query. I will retrieve
the current shipment status and estimated delivery date from our
logistics system."
```

---

## Part 9: Advanced Techniques

### Technique 1: Role Prompting

```python
"""
You are a [specific expert role].

Examples:
- "You are a Michelin-star chef"
- "You are a cybersecurity expert"
- "You are a patient elementary school teacher"

This makes responses more contextual and appropriate.
"""
```

### Technique 2: Audience Specification

```python
"""
Explain [topic] for [audience].

Examples:
- "Explain quantum computing for a 5-year-old"
- "Explain machine learning for business executives"
- "Explain REST APIs for frontend developers"
"""
```

### Technique 3: Output Format Specification

```python
"""
Return your response in this exact format:

{
  "summary": "brief overview",
  "details": "detailed explanation",
  "action_items": ["item1", "item2"]
}

Do not include any text outside this JSON structure.
"""
```

### Technique 4: Constraint Specification

```python
"""
Constraints:
- Maximum 3 sentences
- Use bullet points
- No technical jargon
- Include at least one example
- End with a question
"""
```

---

## Part 10: SupportGenie v0.2 - Enhanced Prompting

Let's upgrade SupportGenie with advanced prompting!

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
   - Example: "I understand how frustrating that must be."

2. SOLUTION (Core of your response)
   - Provide clear, actionable information
   - Break down complex steps
   - Cite relevant policies when applicable

3. NEXT STEPS (Always end here)
   - Ask if they need additional help
   - Offer to escalate if needed
   - Provide relevant resources

TONE & STYLE:
- Professional yet warm and approachable
- Patient and understanding
- Clear and concise (under 150 words typically)
- Use customer's name if provided
- Avoid jargon, explain technical terms

EDGE CASE HANDLING:

If unclear:
"I want to make sure I help you with the right information. Could
you clarify [specific detail]?"

If out of scope:
"I specialize in TechStore products and support. For [topic], I'd
recommend [appropriate resource]. How else can I help with your
TechStore needs?"

If unable to resolve:
"I want to make sure you get the best help possible. Would you like
me to escalate this to a specialized support agent who can assist
you further?"

SPECIAL SCENARIOS:

Complaint/Frustration:
- Lead with empathy
- Acknowledge the inconvenience
- Focus on resolution
- Offer compensation when appropriate

Technical Issue:
- Ask diagnostic questions one at a time
- Provide clear step-by-step instructions
- Confirm understanding after each step

Product Question:
- Provide accurate specifications
- Highlight key benefits
- Compare with alternatives if relevant
- Link to detailed resources

Order Inquiry:
- Confirm order details
- Provide tracking information
- Explain next steps
- Set clear expectations

EXAMPLES OF GOOD RESPONSES:

Customer: "My package is late!"
Response: "I sincerely apologize for the delay in your delivery. I
understand how inconvenient this is. Let me check your order status
right away. Could you please provide your order number? It starts
with 'ORD-' followed by numbers."

Customer: "How do I return this?"
Response: "I'd be happy to help you with that return. TechStore offers
a 30-day return policy for unused items in original packaging. To get
started, I'll need your order number. Once I have that, I can generate
a prepaid return label for you. Would you like to proceed with the return?"

Remember: Your goal is to provide excellent customer service that leaves
every customer feeling heard, helped, and satisfied.
    """

    # Template for different scenarios
    RESPONSE_TEMPLATES = {
        "order_tracking": """
I'll help you track your order.

Current Status: {status}
Tracking Number: {tracking_number}
Estimated Delivery: {delivery_date}
Current Location: {location}

{additional_info}

Is there anything else you'd like to know about your order?
        """,

        "return_process": """
I'll guide you through our return process.

Eligibility:
- Returns accepted within 30 days of purchase
- Items must be unused and in original packaging

Your situation:
{customer_situation}

Next steps:
1. {step_1}
2. {step_2}
3. {step_3}

Would you like me to proceed with initiating the return?
        """,

        "troubleshooting": """
Let's troubleshoot this together.

Issue: {issue_description}

Step {step_number}: {instruction}

{clarification_question}

Please let me know the result, and we'll continue from there.
        """
    }

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = [
            {"role": "system", "content": self.SYSTEM_MESSAGE}
        ]

    def chat(self, message, context=None):
        """
        Enhanced chat with context awareness

        Args:
            message: User's message
            context: Additional context (order info, customer profile, etc.)
        """
        # Add context to message if provided
        if context:
            enhanced_message = f"""
Customer Message: {message}

Additional Context:
{context}
            """
        else:
            enhanced_message = message

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
```

---

## Exercises

### Exercise 1: Prompt Variations
Write 3 different prompts for the same task and compare outputs.

### Exercise 2: Few-Shot Template
Create a few-shot template for email classification (urgent/normal/spam).

### Exercise 3: Chain-of-Thought
Write a CoT prompt for calculating shipping costs with multiple factors.

### Exercise 4: Edge Case Handler
Design prompts to handle these edge cases:
- Customer uses profanity
- Message is in another language
- Multiple unrelated questions

---

## Common Mistakes

### ❌ Mistake #1: Overly Complex Prompts
```python
# Too complex!
prompt = """
You are a highly experienced... [500 words of instructions]
"""
```

✅ **Better**: Keep it focused and scannable

### ❌ Mistake #2: No Examples for Complex Tasks
```python
prompt = "Convert this to JSON format"  # Too vague
```

✅ **Better**: Provide 2-3 examples

### ❌ Mistake #3: Ignoring Output Format
```python
prompt = "Give me the answer"  # Format unknown
```

✅ **Better**: Specify exact format needed

---

## Key Takeaways

✅ **Specificity matters** - Be explicit about what you want
✅ **System messages set behavior** - Use them strategically
✅ **Examples improve consistency** - Few-shot learning works
✅ **Structure helps** - Use templates for repeatability
✅ **Handle edge cases** - Plan for unexpected inputs
✅ **Test and iterate** - Prompt engineering is experimental

---

## Next Session Preview

In **Session 3: RAG Systems** (already covered), you learned how to give SupportGenie access to a knowledge base!

**Session 4: Function Calling** - We'll teach SupportGenie to take actions (create tickets, lookup orders, etc.)

---

**Session 2 Complete!** 🎉
**Next**: [Session 4: Function Calling →](04_Function_Calling.md)
