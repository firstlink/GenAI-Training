# Session 7: Evaluation & Testing

**Duration**: 75 minutes  
**Difficulty**: Intermediate  
**Colab Notebook**: [07_Evaluation.ipynb](../notebooks/07_Evaluation.ipynb)

## Learning Objectives
- 🎯 Create test datasets
- 🎯 Define quality metrics
- 🎯 Build evaluation frameworks
- 🎯 Implement automated testing
- 🎯 A/B test different approaches
- 🎯 Set up monitoring dashboards

## Part 1: Why Evaluate?

**Problems without evaluation**:
- Don't know if system is improving
- Can't compare approaches
- Don't catch regressions
- No data for decisions

**Benefits of evaluation**:
- Measure quality objectively
- Track improvements over time
- Compare models/prompts
- Catch bugs early

## Part 2: Types of Evaluation

### 1. Accuracy
Does the system give correct answers?

### 2. Relevance
Is the response on-topic?

### 3. Consistency
Are similar queries answered similarly?

### 4. Safety
Does it avoid harmful content?

### 5. Performance
Response time, cost, reliability

## Part 3: Creating Test Datasets

```python
test_cases = [
    {
        "id": 1,
        "category": "order_status",
        "query": "Where is my order ORD-12345?",
        "expected_function": "get_order_status",
        "expected_args": {"order_id": "ORD-12345"},
        "expected_contains": ["shipped", "tracking"],
        "difficulty": "easy"
    },
    {
        "id": 2,
        "category": "refund",
        "query": "I want a refund for broken product",
        "expected_function": "create_support_ticket",
        "expected_contains": ["ticket", "refund"],
        "difficulty": "medium"
    },
    # Add 50-100 test cases
]
```

## Part 4: Automated Evaluation

```python
class RAGEvaluator:
    def __init__(self, rag_system):
        self.system = rag_system
    
    def evaluate_retrieval(self, test_cases):
        """Evaluate retrieval quality"""
        results = []
        
        for test in test_cases:
            retrieved = self.system.retrieve(test['query'])
            
            # Check if relevant docs retrieved
            relevant = any(
                test['expected_doc'] in doc 
                for doc in retrieved
            )
            
            results.append({
                "query": test['query'],
                "relevant_retrieved": relevant,
                "retrieval_score": self.calculate_score(retrieved, test)
            })
        
        # Calculate metrics
        accuracy = sum(r['relevant_retrieved'] for r in results) / len(results)
        return {"accuracy": accuracy, "details": results}
    
    def evaluate_generation(self, test_cases):
        """Evaluate response quality"""
        results = []
        
        for test in test_cases:
            response = self.system.generate(test['query'])
            
            # Check response quality
            score = {
                "contains_expected": self.check_contains(response, test['expected_contains']),
                "factually_correct": self.check_facts(response, test),
                "tone_appropriate": self.check_tone(response)
            }
            
            results.append(score)
        
        return self.aggregate_scores(results)
```

## Part 5: LLM-as-Judge

Use LLMs to evaluate other LLMs:

```python
def llm_evaluate(query, response, expected):
    eval_prompt = f"""Evaluate this customer support response.

Query: {query}
Response: {response}
Expected elements: {expected}

Rate from 1-5 on:
1. Accuracy
2. Helpfulness
3. Tone
4. Completeness

Return JSON:
{{
  "accuracy": 1-5,
  "helpfulness": 1-5,
  "tone": 1-5,
  "completeness": 1-5,
  "reasoning": "brief explanation"
}}"""
    
    result = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": eval_prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(result.choices[0].message.content)
```

## Part 6: Metrics Dashboard

```python
class MetricsDashboard:
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_latency": 0,
            "total_cost": 0,
            "error_count": 0
        }
    
    def track_query(self, query, response, latency, cost, error=None):
        self.metrics["total_queries"] += 1
        
        if not error:
            self.metrics["successful_queries"] += 1
        else:
            self.metrics["error_count"] += 1
        
        # Update averages
        n = self.metrics["total_queries"]
        self.metrics["average_latency"] = (
            (self.metrics["average_latency"] * (n-1) + latency) / n
        )
        self.metrics["total_cost"] += cost
    
    def get_summary(self):
        return {
            "success_rate": self.metrics["successful_queries"] / self.metrics["total_queries"],
            "avg_latency": self.metrics["average_latency"],
            "total_cost": self.metrics["total_cost"],
            "error_rate": self.metrics["error_count"] / self.metrics["total_queries"]
        }
```

## Part 7: A/B Testing

```python
class ABTest:
    def __init__(self, version_a, version_b):
        self.version_a = version_a
        self.version_b = version_b
        self.results = {"a": [], "b": []}
    
    def run_test(self, test_cases):
        for test in test_cases:
            # Randomly assign
            version = random.choice(["a", "b"])
            
            if version == "a":
                response = self.version_a.process(test['query'])
            else:
                response = self.version_b.process(test['query'])
            
            score = self.evaluate(response, test)
            self.results[version].append(score)
    
    def analyze(self):
        avg_a = sum(self.results["a"]) / len(self.results["a"])
        avg_b = sum(self.results["b"]) / len(self.results["b"])
        
        return {
            "version_a_score": avg_a,
            "version_b_score": avg_b,
            "winner": "A" if avg_a > avg_b else "B"
        }
```

## Exercises
1. Create 50-test golden dataset
2. Implement automated evaluation
3. Build metrics dashboard
4. Run A/B test on prompts

## Key Takeaways
✅ Evaluation enables improvement  
✅ Create comprehensive test datasets  
✅ Use multiple metrics  
✅ Automate evaluation  
✅ A/B test changes

**Session 7 Complete!** 🎉  
**Next**: [Session 8: Production →](08_Production.md)
