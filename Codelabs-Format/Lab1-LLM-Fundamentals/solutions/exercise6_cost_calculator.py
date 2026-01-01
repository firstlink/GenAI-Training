"""
Lab 1 - Exercise 6: Cost Calculator
Solution for calculating and tracking LLM API costs

Learning Objectives:
- Calculate costs for different models
- Track cumulative spending
- Estimate project costs
- Implement budget monitoring
"""

from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class CostCalculator:
    """Track and calculate LLM API costs"""

    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }

    def __init__(self):
        self.history = []
        self.total_cost = 0.0

    def calculate_cost(self, model, prompt_tokens, completion_tokens):
        """
        Calculate cost for a single API call

        Args:
            model: Model name
            prompt_tokens: Input tokens
            completion_tokens: Output tokens

        Returns:
            Cost in USD
        """
        if model not in self.PRICING:
            print(f"⚠️ Unknown model: {model}, using gpt-3.5-turbo pricing")
            model = "gpt-3.5-turbo"

        input_cost = (prompt_tokens / 1000) * self.PRICING[model]["input"]
        output_cost = (completion_tokens / 1000) * self.PRICING[model]["output"]

        return input_cost + output_cost

    def log_call(self, model, prompt_tokens, completion_tokens, description=""):
        """
        Log an API call and track cost

        Args:
            model: Model used
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            description: Optional description of the call
        """
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)

        call_info = {
            "timestamp": datetime.now(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "description": description
        }

        self.history.append(call_info)
        self.total_cost += cost

        return cost

    def get_summary(self):
        """Get cost summary"""
        if not self.history:
            return "No API calls logged yet"

        total_prompt_tokens = sum(call["prompt_tokens"] for call in self.history)
        total_completion_tokens = sum(call["completion_tokens"] for call in self.history)

        summary = f"""
📊 COST SUMMARY
{'=' * 70}
Total API Calls: {len(self.history)}
Total Input Tokens: {total_prompt_tokens:,}
Total Output Tokens: {total_completion_tokens:,}
Total Tokens: {total_prompt_tokens + total_completion_tokens:,}
Total Cost: ${self.total_cost:.4f}
Average Cost per Call: ${self.total_cost / len(self.history):.6f}
"""
        return summary

    def estimate_project_cost(self, calls_per_day, avg_prompt_tokens, avg_completion_tokens, model, days=30):
        """
        Estimate project cost

        Args:
            calls_per_day: Expected API calls per day
            avg_prompt_tokens: Average prompt length
            avg_completion_tokens: Average completion length
            model: Model to use
            days: Number of days

        Returns:
            Estimated cost
        """
        cost_per_call = self.calculate_cost(model, avg_prompt_tokens, avg_completion_tokens)
        daily_cost = cost_per_call * calls_per_day
        total_cost = daily_cost * days

        return {
            "cost_per_call": cost_per_call,
            "daily_cost": daily_cost,
            "monthly_cost": total_cost,
            "yearly_cost": total_cost * 12
        }


# ============================================================================
# MAIN DEMO
# ============================================================================

print("=" * 70)
print("EXERCISE 6: COST CALCULATOR")
print("=" * 70)

# Initialize cost calculator
calculator = CostCalculator()

# Task 6.1: Calculate Individual Call Costs
print("\n\n💰 TASK 6.1: INDIVIDUAL CALL COST CALCULATION")
print("=" * 70)

models_to_test = ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"]
prompt_tokens = 100
completion_tokens = 200

print(f"\nScenario: {prompt_tokens} input tokens, {completion_tokens} output tokens\n")

for model in models_to_test:
    cost = calculator.calculate_cost(model, prompt_tokens, completion_tokens)
    print(f"{model:20} → ${cost:.6f}")

# Task 6.2: Track Actual API Calls
print("\n\n📝 TASK 6.2: TRACKING ACTUAL API CALLS")
print("=" * 70)

test_prompts = [
    "What is Python?",
    "Explain machine learning in one sentence",
    "List 3 benefits of AI"
]

for prompt in test_prompts:
    print(f"\n💬 Prompt: '{prompt}'")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7
    )

    # Log the call
    cost = calculator.log_call(
        model="gpt-3.5-turbo",
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        description=prompt
    )

    print(f"   Response: {response.choices[0].message.content[:80]}...")
    print(f"   Tokens: {response.usage.prompt_tokens} + {response.usage.completion_tokens} = {response.usage.total_tokens}")
    print(f"   Cost: ${cost:.6f}")

# Show summary
print(calculator.get_summary())

# Task 6.3: Project Cost Estimation
print("\n\n📊 TASK 6.3: PROJECT COST ESTIMATION")
print("=" * 70)

scenarios = [
    {
        "name": "Small Chatbot",
        "calls_per_day": 100,
        "avg_prompt": 50,
        "avg_completion": 150,
        "model": "gpt-3.5-turbo"
    },
    {
        "name": "Medium SaaS App",
        "calls_per_day": 1000,
        "avg_prompt": 200,
        "avg_completion": 300,
        "model": "gpt-4o-mini"
    },
    {
        "name": "Enterprise Solution",
        "calls_per_day": 10000,
        "avg_prompt": 500,
        "avg_completion": 500,
        "model": "gpt-4-turbo"
    }
]

for scenario in scenarios:
    estimate = calculator.estimate_project_cost(
        calls_per_day=scenario["calls_per_day"],
        avg_prompt_tokens=scenario["avg_prompt"],
        avg_completion_tokens=scenario["avg_completion"],
        model=scenario["model"],
        days=30
    )

    print(f"\n📌 {scenario['name']}")
    print(f"   Model: {scenario['model']}")
    print(f"   API Calls: {scenario['calls_per_day']}/day")
    print(f"   Avg Tokens: {scenario['avg_prompt']} → {scenario['avg_completion']}")
    print(f"   Cost per call: ${estimate['cost_per_call']:.6f}")
    print(f"   Daily cost: ${estimate['daily_cost']:.2f}")
    print(f"   Monthly cost: ${estimate['monthly_cost']:.2f}")
    print(f"   Yearly cost: ${estimate['yearly_cost']:.2f}")

# Task 6.4: Budget Monitor
print("\n\n🚨 TASK 6.4: BUDGET MONITORING")
print("=" * 70)

class BudgetMonitor:
    """Monitor spending against budget"""

    def __init__(self, daily_budget):
        self.daily_budget = daily_budget
        self.today_spending = 0.0
        self.alerts = []

    def add_cost(self, cost):
        """Add cost and check budget"""
        self.today_spending += cost

        percentage = (self.today_spending / self.daily_budget) * 100

        if percentage >= 90:
            self.alerts.append(f"⚠️ CRITICAL: {percentage:.1f}% of daily budget used!")
        elif percentage >= 75:
            self.alerts.append(f"⚠️ WARNING: {percentage:.1f}% of daily budget used")

        return percentage

    def get_status(self):
        """Get budget status"""
        percentage = (self.today_spending / self.daily_budget) * 100
        remaining = self.daily_budget - self.today_spending

        return f"""
Daily Budget: ${self.daily_budget:.2f}
Spent Today: ${self.today_spending:.2f}
Remaining: ${remaining:.2f}
Usage: {percentage:.1f}%
"""

# Example budget monitoring
budget_monitor = BudgetMonitor(daily_budget=10.00)

print(f"\nSetting daily budget: $10.00")
print("\nSimulating API calls...")

simulated_costs = [0.05, 0.10, 0.15, 2.50, 3.00, 5.00]

for i, cost in enumerate(simulated_costs, 1):
    percentage = budget_monitor.add_cost(cost)
    print(f"\nCall {i}: ${cost:.2f} → {percentage:.1f}% of budget")

    # Check for alerts
    if budget_monitor.alerts:
        for alert in budget_monitor.alerts:
            print(f"   {alert}")
        budget_monitor.alerts = []

print(budget_monitor.get_status())

print("\n\n✅ Exercise 6 Complete!")
print("\n💡 Key Takeaways:")
print("  - Always track token usage for cost monitoring")
print("  - GPT-3.5-turbo is 20x cheaper than GPT-4")
print("  - Input tokens cost less than output tokens")
print("  - Set max_tokens to control costs")
print("  - Monitor spending against budget")
print("  - Estimate costs before launching production")

print("\n\n🎯 COST OPTIMIZATION TIPS:")
print("=" * 70)
print("1. Use gpt-3.5-turbo for simple tasks")
print("2. Use gpt-4o-mini for balanced cost/performance")
print("3. Set appropriate max_tokens")
print("4. Cache responses when possible")
print("5. Use temperature=0 for deterministic tasks (save on retries)")
print("6. Batch similar requests")
print("7. Monitor and set budget alerts")
print("8. Choose the right model for each task")
