#!/usr/bin/env python3
"""
Verify all fixes are present in notebooks by searching for key content
"""

import sys

def check_file_contains(filepath, search_strings, should_not_contain=None):
    """Check if file contains all search strings and doesn't contain excluded strings"""
    with open(filepath, 'r') as f:
        content = f.read()

    results = []
    for search in search_strings:
        if search in content:
            results.append((search, True))
        else:
            results.append((search, False))

    if should_not_contain:
        for exclude in should_not_contain:
            if exclude in content:
                results.append((f"NOT {exclude}", False))
            else:
                results.append((f"NOT {exclude}", True))

    return results


def main():
    """Verify all fixes"""
    print("="*70)
    print("VERIFYING ALL FIXES")
    print("="*70)

    base = "/Users/firstlinkconsultingllc/udemy/Training/GenAI-Training/Codelabs-Format"

    tests = [
        {
            "name": "Lab 1 - Client initialization",
            "file": f"{base}/Lab1-LLM-Fundamentals/Lab1-LLM-Fundamentals.ipynb",
            "should_contain": [
                "client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))",
                "_calculate_call_cost",
                "stream_options"
            ]
        },
        {
            "name": "Lab 2 - Sentiment/Intent detection",
            "file": f"{base}/Lab2-Prompt-Engineering/Lab2-Prompt-Engineering.ipynb",
            "should_contain": [
                "def detect_sentiment(self, message):",
                "def detect_intent(self, message):",
                "order_tracking"
            ]
        },
        {
            "name": "Lab 3 - ChromaDB fix",
            "file": f"{base}/Lab3-Document-Processing/Lab3-Document-Processing.ipynb",
            "should_contain": [
                "chromadb.PersistentClient"
            ],
            "should_not_contain": [
                "from chromadb.config import Settings"
            ]
        },
        {
            "name": "Lab 4 - Dependencies",
            "file": f"{base}/Lab4-Semantic-Search/Lab4-Semantic-Search.ipynb",
            "should_contain": [
                "langchain",
                "langchain-community"
            ]
        },
        {
            "name": "Lab 5 - Error handling",
            "file": f"{base}/Lab5-RAG-Pipeline/Lab5-RAG-Pipeline.ipynb",
            "should_contain": [
                "sentence-transformers",
                "chromadb",
                "raise RuntimeError"
            ],
            "should_not_contain": [
                "# exit()"
            ]
        },
        {
            "name": "Lab 6 - Type hints",
            "file": f"{base}/Lab6-AI-Agents/Lab6-AI-Agents.ipynb",
            "should_contain": [
                "def execute_tool_with_retry"
            ],
            "should_not_contain": [
                "-> tuple[bool, Any]"
            ]
        },
        {
            "name": "Lab 7 - Security (eval removal)",
            "file": f"{base}/Lab7-Agent-Memory/Lab7-Agent-Memory.ipynb",
            "should_contain": [
                "safe_eval_math",
                "ast.parse"
            ],
            "should_not_contain": [
                "result = eval(expression)"
            ]
        },
        {
            "name": "Lab 8 - Security (eval removal)",
            "file": f"{base}/Lab8-Advanced-Agents/Lab8-Advanced-Agents.ipynb",
            "should_contain": [
                "safe_eval_math",
                "ast.parse"
            ],
            "should_not_contain": [
                "result = eval(expression)"
            ]
        },
    ]

    all_passed = True

    for test in tests:
        print(f"\n[{test['name']}]")
        should_contain = test.get('should_contain', [])
        should_not_contain = test.get('should_not_contain', [])

        results = check_file_contains(test['file'], should_contain, should_not_contain)

        test_passed = all(passed for _, passed in results)

        for item, passed in results:
            status = "✓" if passed else "✗"
            # Truncate long strings for display
            display_item = item if len(item) < 60 else item[:57] + "..."
            print(f"  {status} {display_item}")

        if not test_passed:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL FIXES VERIFIED SUCCESSFULLY")
    else:
        print("✗ SOME FIXES NOT FOUND")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
