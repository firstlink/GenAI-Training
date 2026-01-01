# Lab 2: Prompt Engineering
## 📚 Learning Material

> **Purpose:** Master the art of crafting effective prompts to get exceptional results from LLMs

---

## 📋 Overview

| Property | Value |
|----------|-------|
| **Duration** | 40 minutes (reading) |
| **Difficulty** | Beginner to Intermediate |
| **Prerequisites** | Completed Lab 1 |
| **Next Step** | [Hands-On Lab →](lab.md) |

---

## 📖 Table of Contents

1. [Introduction to Prompt Engineering](#1-introduction-to-prompt-engineering)
2. [Anatomy of a Good Prompt](#2-anatomy-of-a-good-prompt)
3. [System Messages](#3-system-messages)
4. [Few-Shot Learning](#4-few-shot-learning)
5. [Chain-of-Thought Prompting](#5-chain-of-thought-prompting)
6. [Prompt Templates](#6-prompt-templates)
7. [Edge Case Handling](#7-edge-case-handling)
8. [Tone and Style Control](#8-tone-and-style-control)
9. [Advanced Techniques](#9-advanced-techniques)
10. [Review & Key Takeaways](#10-review--key-takeaways)

---

## 1. Introduction to Prompt Engineering

### What is Prompt Engineering?

**Prompt Engineering** is the art and science of crafting inputs (prompts) to get desired outputs from Large Language Models.

Think of it as **programming with natural language** instead of code.

---

### Why It Matters

The difference between a good and bad prompt can be dramatic:

```
┌─────────────────────────────────────────────────┐
│  BAD PROMPT                                     │
├─────────────────────────────────────────────────┤
│  "Tell me about returns"                        │
│                                                  │
│  LLM Output:                                    │
│  "Returns can refer to many things:             │
│  - Financial returns on investment              │
│  - Product returns to a store                   │
│  - Return statements in programming..."         │
│                                                  │
│  ❌ Ambiguous, got wrong answer                 │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  GOOD PROMPT                                    │
├─────────────────────────────────────────────────┤
│  "As a customer service agent, explain our      │
│  30-day product return policy. Include          │
│  eligibility requirements and the return        │
│  process. Keep it under 100 words."             │
│                                                  │
│  LLM Output:                                    │
│  "Our 30-day return policy allows you to       │
│  return unused products in original packaging   │
│  within 30 days of purchase. To initiate a      │
│  return, contact support@company.com with your  │
│  order number. We'll provide a prepaid return   │
│  label. Refunds are processed within 5-7        │
│  business days after we receive the item..."    │
│                                                  │
│  ✅ Specific, clear, got exactly what we needed │
└─────────────────────────────────────────────────┘
```

**The difference?** Specificity, context, and structure.

---

### The Impact on Your Applications

```
Good Prompts → Better Outputs → Happy Users
Bad Prompts → Inconsistent/Wrong Outputs → Frustrated Users
```

**Real-world impact:**
- 🎯 **Accuracy**: Get factual, relevant answers
- 🎯 **Consistency**: Same format every time
- 🎯 **Efficiency**: Fewer retries, less token usage
- 🎯 **User Experience**: Professional, helpful responses

---

### Prompt Engineering is Iterative

```
┌─────────────────────────────────────────────────┐
│  PROMPT ENGINEERING CYCLE                       │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. WRITE → Test initial prompt                 │
│       ↓                                          │
│  2. EVALUATE → Check output quality             │
│       ↓                                          │
│  3. REFINE → Adjust and improve                 │
│       ↓                                          │
│  4. REPEAT → Until satisfactory                 │
│       ↓                                          │
│  5. STANDARDIZE → Save as template              │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## 2. Anatomy of a Good Prompt

### The 7 Essential Components

Every effective prompt should include most or all of these:

```
┌─────────────────────────────────────────────────┐
│  COMPLETE PROMPT STRUCTURE                      │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. ROLE/PERSONA                                │
│     └─ "You are a [specific expert]"            │
│                                                  │
│  2. CONTEXT                                     │
│     └─ Background information needed            │
│                                                  │
│  3. TASK                                        │
│     └─ What exactly to do                       │
│                                                  │
│  4. CONSTRAINTS                                 │
│     └─ Limitations and boundaries               │
│                                                  │
│  5. FORMAT                                      │
│     └─ How to structure the output              │
│                                                  │
│  6. EXAMPLES (optional but powerful)            │
│     └─ Sample inputs and outputs                │
│                                                  │
│  7. TONE                                        │
│     └─ Style and voice to use                   │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

### Example: Complete Prompt Breakdown

```python
prompt = """
┌─────────────────────────────────────────┐
│ 1. ROLE/PERSONA                         │
└─────────────────────────────────────────┘
You are an expert technical support agent for TechStore.

┌─────────────────────────────────────────┐
│ 2. CONTEXT                              │
└─────────────────────────────────────────┘
TechStore sells electronics and provides warranty support.
You have access to our knowledge base and can create support tickets.

┌─────────────────────────────────────────┐
│ 3. TASK                                 │
└─────────────────────────────────────────┘
Help the customer troubleshoot their laptop that won't turn on.

┌─────────────────────────────────────────┐
│ 4. CONSTRAINTS                          │
└─────────────────────────────────────────┘
- Ask one diagnostic question at a time
- Keep responses under 75 words
- Use simple, non-technical language
- Be patient and empathetic

┌─────────────────────────────────────────┐
│ 5. FORMAT                               │
└─────────────────────────────────────────┘
1. Acknowledge the issue
2. Ask a diagnostic question
3. Explain why you're asking

┌─────────────────────────────────────────┐
│ 7. TONE                                 │
└─────────────────────────────────────────┘
Professional, patient, and reassuring

┌─────────────────────────────────────────┐
│ USER INPUT                              │
└─────────────────────────────────────────┘
Customer Message: My laptop won't turn on!
"""
```

**Result:** Clear, structured, helpful response that follows all guidelines.

---

### Progressive Prompting: From Vague to Specific

```
❌ VAGUE:
"Help with my order"

⚠️ BETTER:
"I need help tracking my order"

✅ GOOD:
"I need help tracking my TechStore order #ORD-12345 that was supposed
to arrive yesterday"

✨ EXCELLENT:
"As a customer service agent for TechStore, help me track order
#ORD-12345 that was supposed to arrive on December 28th but hasn't
arrived. Provide the current status, location, and new estimated
delivery date. Keep response under 100 words and be empathetic
about the delay."
```

**Pattern:** Add role + context + specific details + format + tone

---

## 3. System Messages

### What Are System Messages?

**System messages** define the AI's **persistent behavior** across the entire conversation.

```
┌─────────────────────────────────────────────────┐
│  MESSAGE TYPES IN A CONVERSATION                │
├─────────────────────────────────────────────────┤
│                                                  │
│  SYSTEM  → Sets behavior (persistent)           │
│   ↓                                              │
│  USER    → Asks question                        │
│   ↓                                              │
│  ASSISTANT → Responds following system rules    │
│   ↓                                              │
│  USER    → Asks follow-up                       │
│   ↓                                              │
│  ASSISTANT → Still follows system rules         │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Key point:** System message affects ALL responses in the conversation.

---

### System Message Structure

```python
{
    "role": "system",
    "content": """
┌──────────────────────────────────┐
│ WHO YOU ARE (Identity)           │
└──────────────────────────────────┘
You are [role/persona]

┌──────────────────────────────────┐
│ WHAT YOU CAN DO (Capabilities)   │
└──────────────────────────────────┘
✓ Capability 1
✓ Capability 2
✓ Capability 3

┌──────────────────────────────────┐
│ WHAT YOU CAN'T DO (Constraints)  │
└──────────────────────────────────┘
✗ Constraint 1
✗ Constraint 2
✗ Constraint 3

┌──────────────────────────────────┐
│ HOW TO RESPOND (Guidelines)      │
└──────────────────────────────────┘
1. First do this
2. Then do this
3. Finally do this

┌──────────────────────────────────┐
│ HOW TO SOUND (Tone)              │
└──────────────────────────────────┘
Be [tone description]
    """
}
```

---

### ❌ Bad vs ✅ Good System Messages

**❌ TOO VAGUE:**
```python
system_message = "You are a helpful assistant."
```
Problems:
- No specific role
- No guidelines
- No constraints
- Unpredictable behavior

---

**⚠️ SLIGHTLY BETTER:**
```python
system_message = "You are a customer service agent. Be helpful."
```
Problems:
- Still too general
- No specific guidelines
- No format specified

---

**✅ GOOD:**
```python
system_message = """
You are a customer service agent for TechStore.

Guidelines:
- Be professional and empathetic
- Keep responses under 100 words
- Always offer to escalate if needed
- Never make up information

Response format:
1. Acknowledge concern
2. Provide solution
3. Ask if they need more help
"""
```
Better because:
- Specific role
- Clear guidelines
- Defined format
- Boundaries set

---

**✨ EXCELLENT:**
```python
system_message = """
You are SupportGenie, an expert AI customer support assistant for TechStore.

IDENTITY:
- Professional customer service representative
- Knowledgeable about products, policies, shipping
- Empathetic and solution-focused

CAPABILITIES:
✓ Answer questions about products and policies
✓ Help with order tracking
✓ Create support tickets
✓ Escalate to human agents when needed

CONSTRAINTS:
✗ Keep responses under 100 words
✗ Never make up information
✗ Always cite sources when referencing policies
✗ Admit when you don't know something

TONE:
- Professional but friendly
- Empathetic to customer concerns
- Solution-oriented
- Patient and clear

RESPONSE FORMAT:
1. Acknowledge customer's concern
2. Provide solution or information
3. Ask if they need additional help

EDGE CASES:
- If unclear: Ask for clarification politely
- If out of scope: Redirect to appropriate channel
- If hostile: Remain professional, don't engage
"""
```
Excellent because:
- Complete identity
- Clear capabilities and limits
- Specific tone guidelines
- Defined structure
- Edge case handling

---

## 4. Few-Shot Learning

### What is Few-Shot Learning?

**Few-Shot Learning** = Providing **examples** in the prompt to guide the model's responses.

```
┌─────────────────────────────────────────────────┐
│  LEARNING APPROACHES                            │
├─────────────────────────────────────────────────┤
│                                                  │
│  ZERO-SHOT (No examples)                        │
│  ├─ "Classify the sentiment"                    │
│  └─ Model guesses based on training             │
│                                                  │
│  FEW-SHOT (2-5 examples)                        │
│  ├─ "Example 1: [input] → [output]"             │
│  ├─ "Example 2: [input] → [output]"             │
│  ├─ "Example 3: [input] → [output]"             │
│  └─ Model learns pattern from examples          │
│                                                  │
│  MANY-SHOT (10+ examples)                       │
│  └─ Usually overkill, 2-5 is enough             │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

### Zero-Shot vs Few-Shot Comparison

**❌ ZERO-SHOT (Inconsistent):**
```python
prompt = "Extract the customer name and email from this text"
text = "Hi, I'm John Smith. Contact me at john@email.com"

# Output may vary:
# "John Smith, john@email.com"
# "Name: John Smith Email: john@email.com"
# "Customer: John Smith (john@email.com)"
```
Problem: Format is unpredictable

---

**✅ FEW-SHOT (Consistent):**
```python
prompt = """
Extract customer name and email from text. Return as JSON.

Examples:

Input: "My name is Alice Johnson, email alice@test.com"
Output: {"name": "Alice Johnson", "email": "alice@test.com"}

Input: "I'm Bob Lee (bob.lee@company.com)"
Output: {"name": "Bob Lee", "email": "bob.lee@company.com"}

Input: "Contact Sarah Davis at s.davis@email.net"
Output: {"name": "Sarah Davis", "email": "s.davis@email.net"}

Now extract from:
Input: "Hi, I'm John Smith. Contact me at john@email.com"
Output:
"""

# Output will consistently be:
# {"name": "John Smith", "email": "john@email.com"}
```
Benefit: Consistent JSON format every time

---

### Few-Shot Pattern: Show Don't Just Tell

```
┌─────────────────────────────────────────────────┐
│  PATTERN: INPUT → OUTPUT EXAMPLES              │
├─────────────────────────────────────────────────┤
│                                                  │
│  "Classify sentiment of customer messages.      │
│                                                  │
│  Example 1:                                     │
│  Message: 'I love this product!'                │
│  Sentiment: positive                            │
│                                                  │
│  Example 2:                                     │
│  Message: 'Terrible. Product broke.'            │
│  Sentiment: negative                            │
│                                                  │
│  Example 3:                                     │
│  Message: 'It arrived. It's okay.'              │
│  Sentiment: neutral                             │
│                                                  │
│  Now classify:                                  │
│  Message: [new message]                         │
│  Sentiment:"                                    │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Key:** 2-5 diverse examples covering different scenarios

---

### When to Use Few-Shot Learning

**Use Few-Shot When:**
- ✅ You need consistent output format
- ✅ Task involves classification or categorization
- ✅ Extracting structured data
- ✅ Custom formatting requirements
- ✅ Domain-specific patterns

**Example Use Cases:**
- Email classification (urgent/normal/spam)
- Sentiment analysis
- Data extraction (names, dates, emails)
- Format conversion (text to JSON)
- Custom categorization schemes

---

## 5. Chain-of-Thought Prompting

### What is Chain-of-Thought (CoT)?

**Chain-of-Thought** = Asking the LLM to **show its reasoning** step-by-step before giving the final answer.

```
┌─────────────────────────────────────────────────┐
│  WITHOUT CoT                                    │
├─────────────────────────────────────────────────┤
│  Q: Customer ordered 3 items at $50 each.       │
│     20% discount + $5 shipping. Total?          │
│                                                  │
│  A: $125                                        │
│                                                  │
│  ❌ No reasoning shown                          │
│  ❌ Can't verify if correct                     │
│  ❌ Might be wrong                              │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  WITH CoT                                       │
├─────────────────────────────────────────────────┤
│  Q: Customer ordered 3 items at $50 each.       │
│     20% discount + $5 shipping. Total?          │
│                                                  │
│  Let me solve this step by step:               │
│                                                  │
│  Step 1: Calculate subtotal                    │
│    3 items × $50 = $150                         │
│                                                  │
│  Step 2: Apply 20% discount                    │
│    $150 × 0.20 = $30 discount                   │
│    $150 - $30 = $120                            │
│                                                  │
│  Step 3: Add shipping                           │
│    $120 + $5 = $125                             │
│                                                  │
│  Answer: $125                                   │
│                                                  │
│  ✅ Reasoning shown                             │
│  ✅ Can verify each step                        │
│  ✅ More reliable                               │
└─────────────────────────────────────────────────┘
```

---

### How to Implement CoT

**Basic CoT Prompt:**
```python
prompt = """
Solve this step by step:

1. Break down the problem
2. Show each calculation
3. Explain your reasoning
4. Provide the final answer

Problem: {problem}

Solution:
"""
```

**Advanced CoT Prompt:**
```python
prompt = """
Think through this carefully and show your work.

For each step:
- Explain what you're doing
- Show the calculation or reasoning
- State any assumptions

Problem: {problem}

Step-by-step solution:
"""
```

---

### When to Use Chain-of-Thought

**Use CoT When:**
- ✅ Complex calculations
- ✅ Multi-step reasoning
- ✅ Logic puzzles
- ✅ Debugging/troubleshooting
- ✅ Need to verify correctness
- ✅ Explaining to users

**Example Scenarios:**
- Pricing calculations with discounts
- Eligibility determinations
- Troubleshooting technical issues
- Policy application
- Math word problems

**Don't Use CoT When:**
- ❌ Simple, single-step tasks
- ❌ Need very short responses
- ❌ Speed is critical
- ❌ Just need final answer

---

## 6. Prompt Templates

### Why Use Templates?

**Benefits:**
- **Consistency** - Same structure every time
- **Reusability** - Write once, use many times
- **Maintainability** - Update in one place
- **Testing** - Easy to A/B test variations
- **Scaling** - Works across team

---

### Template System Architecture

```
┌─────────────────────────────────────────────────┐
│  TEMPLATE SYSTEM                                │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. DEFINE TEMPLATE                             │
│     ├─ Placeholder variables                    │
│     ├─ Fixed structure                          │
│     └─ Format specifications                    │
│                                                  │
│  2. VALIDATE INPUTS                             │
│     ├─ Check required variables                 │
│     ├─ Validate data types                      │
│     └─ Ensure constraints met                   │
│                                                  │
│  3. FORMAT PROMPT                               │
│     ├─ Replace placeholders                     │
│     ├─ Apply formatting                         │
│     └─ Return complete prompt                   │
│                                                  │
│  4. USE REPEATEDLY                              │
│     └─ Same template, different inputs          │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

### Template Example

```python
# Template definition
customer_support_template = """
You are a {role} for {company}.

Context: {context}

Customer Question: {question}

Respond following these guidelines:
- Keep response under {max_words} words
- Use {tone} tone
- {additional_instructions}

Response:
"""

# Usage 1: Order inquiry
prompt1 = customer_support_template.format(
    role="order specialist",
    company="TechStore",
    context="Customer is asking about order status",
    question="Where is my order?",
    max_words=75,
    tone="professional and empathetic",
    additional_instructions="Offer to escalate if needed"
)

# Usage 2: Product question
prompt2 = customer_support_template.format(
    role="product expert",
    company="TechStore",
    context="Customer is comparing products",
    question="What's the difference between iPhone 15 and 15 Pro?",
    max_words=100,
    tone="informative and helpful",
    additional_instructions="Focus on key differences, not all specs"
)
```

**Same template, different contexts!**

---

### Template Categories

**1. Customer Support Templates**
- Order inquiries
- Product questions
- Return/refund requests
- Technical support

**2. Content Generation Templates**
- Blog post outlines
- Social media posts
- Email drafts
- Product descriptions

**3. Data Processing Templates**
- Extraction (names, dates, etc.)
- Classification
- Summarization
- Translation

**4. Analysis Templates**
- Sentiment analysis
- Intent detection
- Entity recognition
- Topic classification

---

## 7. Edge Case Handling

### Common Edge Cases

```
┌─────────────────────────────────────────────────┐
│  EDGE CASES YOU MUST HANDLE                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. EMPTY INPUT                                 │
│     └─ User sends blank message                 │
│                                                  │
│  2. AMBIGUOUS QUESTIONS                         │
│     └─ "It doesn't work" (what doesn't?)        │
│                                                  │
│  3. OUT-OF-SCOPE QUERIES                        │
│     └─ Politics, medical advice, etc.           │
│                                                  │
│  4. HOSTILE/INAPPROPRIATE INPUT                 │
│     └─ Profanity, harassment                    │
│                                                  │
│  5. MULTIPLE QUESTIONS                          │
│     └─ 5 unrelated questions at once            │
│                                                  │
│  6. WRONG LANGUAGE                              │
│     └─ User writes in non-English               │
│                                                  │
│  7. GIBBERISH/RANDOM TEXT                       │
│     └─ "asdkfjasldkfj"                          │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

### Strategy: Defensive Prompting

**Include edge case instructions in system message:**

```python
system_message = """
You are a customer support assistant.

EDGE CASE HANDLING:

If the message is empty or unclear:
→ "I'd be happy to help! Could you please provide more details
   about what you need assistance with?"

If the question is out of scope (politics, medical, etc.):
→ "I'm here to help with TechStore products and orders. For [topic],
   I'd recommend consulting a specialist. How else can I assist you
   with your TechStore needs?"

If you encounter inappropriate content:
→ "I'm here to assist with your TechStore questions. How can I help
   you with your order or product inquiries?"

If there are multiple unrelated questions:
→ "I see you have several questions. Let's address them one at a time.
   Which would you like to start with?"

If you don't have enough information:
→ "To help you better, I need a bit more information. Could you tell
   me [specific detail needed]?"
"""
```

---

### Handling Ambiguity Example

```
❌ BAD RESPONSE to "it doesn't work":
"I'm sorry it's not working."
(Doesn't help at all)

✅ GOOD RESPONSE to "it doesn't work":
"I'm sorry you're experiencing issues. To help you effectively,
could you tell me:

1. Which product is having the problem?
2. What specifically isn't working?
3. When did this start?
4. Have you tried any troubleshooting steps?

This will help me provide the right solution for you."
```

**Pattern:** Politely request specific information needed to help.

---

## 8. Tone and Style Control

### The Power of Tone

Same message, dramatically different impact based on tone:

```
Customer: "My order is late!"

┌──────────────────────────────────────────┐
│ PROFESSIONAL TONE                        │
└──────────────────────────────────────────┘
"I apologize for the delay in your order.
I will investigate the status immediately
and provide you with an update."

┌──────────────────────────────────────────┐
│ FRIENDLY TONE                            │
└──────────────────────────────────────────┘
"Oh no, I'm so sorry your order is running
late! Let me check on that for you right
away. 😊"

┌──────────────────────────────────────────┐
│ EMPATHETIC TONE                          │
└──────────────────────────────────────────┘
"I completely understand how frustrating it
is when an order doesn't arrive on time.
Let me look into this for you and see what
we can do to make this right."

┌──────────────────────────────────────────┐
│ TECHNICAL TONE                           │
└──────────────────────────────────────────┘
"Order delay detected. Initiating tracking
query. I will retrieve the current shipment
status and estimated delivery date from our
logistics system."
```

---

### Tone Specifications in Prompts

```python
# Professional Tone
tone_instruction = """
Be professional, clear, and respectful. Use complete sentences
and proper grammar. Avoid emojis and casual language.
"""

# Friendly Tone
tone_instruction = """
Be warm and conversational. Use contractions (you're, we'll) and
friendly phrases like 'Happy to help!' Feel free to use appropriate
emojis sparingly.
"""

# Empathetic Tone
tone_instruction = """
Show understanding and compassion. Acknowledge frustrations. Use
phrases like 'I understand how frustrating that must be' and
'Let me help make this right for you.'
"""

# Technical Tone
tone_instruction = """
Be precise and detailed. Use technical terminology when appropriate.
Provide step-by-step instructions. Focus on accuracy over friendliness.
"""
```

---

### Choosing the Right Tone

```
CUSTOMER TYPE         RECOMMENDED TONE
─────────────────────────────────────────────
Frustrated/Angry  →  Empathetic + Professional
Confused          →  Patient + Clear + Friendly
Tech-savvy        →  Technical + Efficient
Casual inquiry    →  Friendly + Professional
Business customer →  Professional + Formal
First-time user   →  Patient + Encouraging
VIP customer      →  Professional + Personalized
```

---

## 9. Advanced Techniques

### Technique 1: Role Prompting

**Concept:** Make the AI adopt a specific expert role.

```python
# Generic (weak)
"Explain machine learning"

# Role-based (strong)
"You are a Stanford AI professor. Explain machine learning to
undergraduate students who have basic Python knowledge."

# Ultra-specific (strongest)
"You are Andrew Ng teaching CS229. Explain supervised learning
to students who understand calculus and linear algebra but have
never coded ML before."
```

**Effect:** More contextual, appropriate, and expert responses.

---

### Technique 2: Audience Specification

**Concept:** Define who the answer is for.

```
SAME TOPIC, DIFFERENT AUDIENCES:

"Explain quantum computing for:"
├─ "a 5-year-old" → Use analogies, very simple
├─ "a high school student" → More detail, some science
├─ "business executives" → Focus on applications, ROI
├─ "physics PhD students" → Technical depth, equations
└─ "frontend developers" → Relate to web concepts
```

**Pattern:** "[Topic] for [specific audience with characteristics]"

---

### Technique 3: Output Format Specification

**Concept:** Define exact structure of output.

```python
# Vague
"Give me product pros and cons"

# Specific format
"""
Return exactly in this JSON format:
{
  "product_name": "string",
  "pros": ["pro1", "pro2", "pro3"],
  "cons": ["con1", "con2"],
  "overall_rating": number (1-5),
  "recommendation": "string"
}

Do not include any text outside this JSON structure.
"""
```

**Formats you can specify:**
- JSON, XML, YAML
- Markdown tables
- Bullet lists
- Numbered steps
- Code blocks
- Email format
- HTML

---

### Technique 4: Constraint Layering

**Concept:** Stack multiple constraints for precision.

```python
prompt = """
Write a product description with these constraints:

LENGTH: Maximum 3 sentences
STRUCTURE: Feature → Benefit → Call-to-action
STYLE: Use active voice only
TONE: Professional but exciting
REQUIREMENTS:
- Mention the price
- Include one statistic
- End with a question
- No technical jargon
- Use exactly one emoji

Product: {product_details}
"""
```

**Result:** Highly controlled, consistent output.

---

## 10. Review & Key Takeaways

### 🎯 What You've Learned

✅ **Prompt Engineering Fundamentals**
- It's programming with natural language
- Small changes = big impact on outputs
- Iterative refinement is key

✅ **The 7-Part Prompt Structure**
- Role, Context, Task, Constraints, Format, Examples, Tone
- More components = more control
- Not all needed for every prompt

✅ **System Messages**
- Define persistent behavior
- Include identity, capabilities, constraints, tone
- Edge case handling crucial

✅ **Few-Shot Learning**
- 2-5 examples teach patterns
- Ensures consistent output format
- Show don't just tell

✅ **Chain-of-Thought**
- Shows reasoning step-by-step
- Improves accuracy for complex tasks
- Makes outputs verifiable

✅ **Templates**
- Reusable, consistent, maintainable
- Separate structure from content
- Essential for scaling

✅ **Edge Case Handling**
- Plan for unexpected inputs
- Graceful degradation
- Professional responses always

✅ **Tone Control**
- Same content, different impact
- Match tone to audience and context
- Be deliberate about style

---

### 🎓 Knowledge Check

<details>
<summary>Question 1: What are the 7 components of a good prompt?</summary>

1. Role/Persona
2. Context
3. Task
4. Constraints
5. Format
6. Examples (optional)
7. Tone

</details>

<details>
<summary>Question 2: When should you use few-shot learning?</summary>

Use few-shot learning when you need:
- Consistent output format
- Custom classification or categorization
- Structured data extraction
- Domain-specific patterns
- Examples: sentiment analysis, data extraction, format conversion

</details>

<details>
<summary>Question 3: What's the difference between system messages and user messages?</summary>

**System messages:**
- Set persistent behavior across entire conversation
- Define who the AI is, what it can/can't do
- Stay in effect for all responses

**User messages:**
- Individual queries or inputs
- Change throughout conversation
- Responded to individually

</details>

<details>
<summary>Question 4: When should you use Chain-of-Thought prompting?</summary>

Use CoT when:
- Complex calculations or multi-step reasoning
- Need to verify correctness
- Explaining process to users
- Debugging or troubleshooting

Don't use when:
- Simple, single-step tasks
- Need very short responses
- Speed is critical

</details>

---

### 💡 Common Mistakes to Avoid

**❌ Mistake 1: Overly Complex Prompts**
```
Don't write 1000-word system messages. Keep it focused and scannable.
```

**❌ Mistake 2: No Examples for Complex Tasks**
```
Vague: "Convert to JSON"
Better: Show 2-3 input→output examples
```

**❌ Mistake 3: Ignoring Output Format**
```
Vague: "Give me the answer"
Better: "Return as JSON: {answer: string, confidence: number}"
```

**❌ Mistake 4: Forgetting Edge Cases**
```
What if input is empty? Ambiguous? Out of scope?
Handle in system message!
```

**❌ Mistake 5: Inconsistent Tone**
```
Pick a tone and stick with it throughout conversation
```

---

### 🚀 Ready for Hands-On Practice?

Now that you understand prompt engineering theory, it's time to **apply these techniques**!

👉 **[Continue to Hands-On Lab →](lab.md)**

In the lab, you'll:
- ✅ Build progressively better prompts
- ✅ Create system messages for different scenarios
- ✅ Implement few-shot learning
- ✅ Build prompt templates
- ✅ Handle edge cases
- ✅ **Enhance SupportGenie v0.2** with advanced prompting

---

### 📚 Additional Reading (Optional)

**Deep Dives:**
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Chain-of-Thought Prompting Paper](https://arxiv.org/abs/2201.11903)

**Tools:**
- [PromptPerfect](https://promptperfect.jina.ai/) - Optimize prompts
- [Prompt.ai](https://prompt.ai/) - Template library

---

**Next:** [Hands-On Lab →](lab.md)
