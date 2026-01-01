"""
Lab 2 - Exercise 5: Prompt Templates
Solution for creating reusable prompt templates

Learning Objectives:
- Build reusable prompt templates
- Use variable substitution
- Create template library
- Implement template versioning
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class PromptTemplate:
    """Reusable prompt template with variable substitution"""

    def __init__(self, template, description=""):
        self.template = template
        self.description = description

    def format(self, **kwargs):
        """Format template with provided variables"""
        return self.template.format(**kwargs)

    def get_variables(self):
        """Extract variable names from template"""
        import re
        return re.findall(r'\{(\w+)\}', self.template)


# Template Library
TEMPLATES = {
    "customer_support": PromptTemplate(
        template="""You are a customer support agent for {company_name}.

Customer Issue: {customer_message}

Respond with:
1. Empathetic acknowledgment
2. Clear solution or next steps
3. Offer additional help

Tone: {tone}
Max length: {max_words} words""",
        description="Professional customer support response"
    ),

    "code_review": PromptTemplate(
        template="""Review this {language} code for:
- Code quality
- Best practices
- Potential bugs
- Performance issues

Code:
```{language}
{code}
```

Provide {num_suggestions} specific suggestions for improvement.""",
        description="Code review with specific criteria"
    ),

    "email_writer": PromptTemplate(
        template="""Write a {tone} email for {purpose}.

Context: {context}
Recipient: {recipient}
Key points to include:
{key_points}

Format: Professional business email""",
        description="Professional email composition"
    ),

    "meeting_summary": PromptTemplate(
        template="""Summarize this meeting transcript in {format} format.

Meeting: {meeting_topic}
Participants: {participants}

Transcript:
{transcript}

Include:
- Key decisions
- Action items
- Next steps""",
        description="Meeting summary generator"
    ),

    "social_media": PromptTemplate(
        template="""Create a {platform} post about {topic}.

Target audience: {audience}
Goal: {goal}
Tone: {tone}

Requirements:
- Engaging hook
- Clear message
- Call to action
- {max_chars} characters max""",
        description="Social media content creator"
    )
}


def demonstrate_basic_template():
    """Demonstrate basic template usage"""
    print("=" * 70)
    print("TASK 5A: BASIC TEMPLATE USAGE")
    print("=" * 70)

    template = TEMPLATES["customer_support"]

    print(f"\n📝 Template: {template.description}")
    print(f"Variables needed: {template.get_variables()}")

    # Fill template
    prompt = template.format(
        company_name="TechCorp",
        customer_message="My order hasn't arrived and it's been 5 days",
        tone="empathetic and professional",
        max_words="100"
    )

    print(f"\n✅ Formatted prompt:\n{prompt}")

    # Get response
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    print(f"\n💬 Response:\n{response.choices[0].message.content}")


def demonstrate_template_reuse():
    """Show template reuse with different values"""
    print("\n\n" + "=" * 70)
    print("TASK 5B: TEMPLATE REUSE")
    print("=" * 70)

    template = TEMPLATES["code_review"]

    code_samples = [
        {
            "language": "Python",
            "code": "def calc(x,y):\n    return x+y",
            "num_suggestions": "3"
        },
        {
            "language": "JavaScript",
            "code": "function getData() {\n  return fetch('/api/data')\n}",
            "num_suggestions": "2"
        }
    ]

    for i, sample in enumerate(code_samples, 1):
        print(f"\n{'='*70}")
        print(f"Sample {i}: {sample['language']}")
        print(f"{'='*70}")

        prompt = template.format(**sample)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )

        print(f"\n💬 Review:\n{response.choices[0].message.content}")


def create_template_library():
    """Build a complete template library"""
    print("\n\n" + "=" * 70)
    print("TASK 5C: TEMPLATE LIBRARY")
    print("=" * 70)

    print("\n📚 Available Templates:\n")
    for name, template in TEMPLATES.items():
        print(f"  • {name}")
        print(f"    Description: {template.description}")
        print(f"    Variables: {', '.join(template.get_variables())}")
        print()


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all demonstrations"""
    print("=" * 70)
    print("EXERCISE 5: PROMPT TEMPLATES")
    print("=" * 70)

    demonstrate_basic_template()
    demonstrate_template_reuse()
    create_template_library()

    print("\n\n" + "=" * 70)
    print("✅ EXERCISE 5 COMPLETE!")
    print("=" * 70)

    print("\n💡 KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Templates enable prompt reusability")
    print("2. Use {variables} for substitution")
    print("3. Build a library of common templates")
    print("4. Test templates with different values")
    print("5. Version templates as you refine them")
    print("6. Document template purpose and variables")


if __name__ == "__main__":
    main()
