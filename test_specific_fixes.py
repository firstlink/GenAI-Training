#!/usr/bin/env python3
"""
Test specific fixes made to each lab
"""

import json
import sys


def read_notebook(path):
    """Read and return notebook"""
    with open(path, 'r') as f:
        return json.load(f)


def get_cell_source(notebook, cell_id):
    """Get source code from a specific cell"""
    for cell in notebook.get('cells', []):
        if cell.get('id') == cell_id:
            source = cell.get('source', [])
            if isinstance(source, list):
                return ''.join(source)
            return source
    return None


def test_lab1_fixes():
    """Test Lab1 specific fixes"""
    print("\n[LAB 1] Testing client initialization and token tracking fixes")

    notebook = read_notebook("/Users/firstlinkconsultingllc/udemy/Training/GenAI-Training/Codelabs-Format/Lab1-LLM-Fundamentals/Lab1-LLM-Fundamentals.ipynb")

    # Test 1: Cell 29 should have client initialization
    cell_29 = get_cell_source(notebook, 'cell-29')
    if cell_29 and 'client = OpenAI' in cell_29:
        print("  ✓ Cell 29: Client properly initialized")
    else:
        print("  ✗ Cell 29: Missing client initialization")
        return False

    # Test 2: Cell 41 should have token tracking
    cell_41 = get_cell_source(notebook, 'cell-41')
    if cell_41 and '_calculate_call_cost' in cell_41 and 'stream_options' in cell_41:
        print("  ✓ Cell 41: Token tracking implemented")
    else:
        print("  ✗ Cell 41: Missing token tracking")
        return False

    return True


def test_lab2_fixes():
    """Test Lab2 specific fixes"""
    print("\n[LAB 2] Testing sentiment/intent detection implementation")

    notebook = read_notebook("/Users/firstlinkconsultingllc/udemy/Training/GenAI-Training/Codelabs-Format/Lab2-Prompt-Engineering/Lab2-Prompt-Engineering.ipynb")

    cell_47 = get_cell_source(notebook, 'cell-47')
    if not cell_47:
        print("  ✗ Cell 47: Not found")
        return False

    # Check for full implementations
    if 'def detect_sentiment' in cell_47 and 'few-shot' not in cell_47.lower() and 'examples:' in cell_47.lower():
        print("  ✓ Cell 47: detect_sentiment() fully implemented with examples")
    else:
        print("  ✗ Cell 47: detect_sentiment() incomplete")
        return False

    if 'def detect_intent' in cell_47 and 'order_tracking' in cell_47:
        print("  ✓ Cell 47: detect_intent() fully implemented with examples")
    else:
        print("  ✗ Cell 47: detect_intent() incomplete")
        return False

    return True


def test_lab3_fixes():
    """Test Lab3 specific fixes"""
    print("\n[LAB 3] Testing ChromaDB import fix")

    notebook = read_notebook("/Users/firstlinkconsultingllc/udemy/Training/GenAI-Training/Codelabs-Format/Lab3-Document-Processing/Lab3-Document-Processing.ipynb")

    cell_36 = get_cell_source(notebook, 'cell-36')
    if not cell_36:
        print("  ✗ Cell 36: Not found")
        return False

    # Check that invalid import is removed
    if 'from chromadb.config import Settings' in cell_36:
        print("  ✗ Cell 36: Still has invalid Settings import")
        return False
    else:
        print("  ✓ Cell 36: Invalid Settings import removed")

    # Check that it has correct chromadb usage
    if 'chromadb.PersistentClient' in cell_36:
        print("  ✓ Cell 36: Uses correct PersistentClient")
    else:
        print("  ✗ Cell 36: Missing PersistentClient")
        return False

    return True


def test_lab4_fixes():
    """Test Lab4 specific fixes"""
    print("\n[LAB 4] Testing dependency additions")

    notebook = read_notebook("/Users/firstlinkconsultingllc/udemy/Training/GenAI-Training/Codelabs-Format/Lab4-Semantic-Search/Lab4-Semantic-Search.ipynb")

    cell_4 = get_cell_source(notebook, 'cell-4')
    if not cell_4:
        print("  ✗ Cell 4: Not found")
        return False

    required = ['langchain', 'langchain-community']
    for dep in required:
        if dep in cell_4:
            print(f"  ✓ Cell 4: {dep} dependency added")
        else:
            print(f"  ✗ Cell 4: Missing {dep} dependency")
            return False

    return True


def test_lab5_fixes():
    """Test Lab5 specific fixes"""
    print("\n[LAB 5] Testing error handling improvements")

    notebook = read_notebook("/Users/firstlinkconsultingllc/udemy/Training/GenAI-Training/Codelabs-Format/Lab5-RAG-Pipeline/Lab5-RAG-Pipeline.ipynb")

    # Test 1: Dependencies
    cell_2 = get_cell_source(notebook, 'cell-2')
    if cell_2 and 'sentence-transformers' in cell_2 and 'chromadb' in cell_2:
        print("  ✓ Cell 2: Dependencies added")
    else:
        print("  ✗ Cell 2: Missing dependencies")
        return False

    # Test 2: Error handling
    cell_10 = get_cell_source(notebook, 'cell-10')
    if cell_10 and 'raise RuntimeError' in cell_10:
        print("  ✓ Cell 10: Proper error handling (raises exception)")
    else:
        print("  ✗ Cell 10: Still has silent error handling")
        return False

    if cell_10 and '# exit()' not in cell_10:
        print("  ✓ Cell 10: No commented exit() statement")
    else:
        print("  ✗ Cell 10: Still has commented exit()")
        return False

    return True


def test_lab6_fixes():
    """Test Lab6 specific fixes"""
    print("\n[LAB 6] Testing type hint compatibility fix")

    notebook = read_notebook("/Users/firstlinkconsultingllc/udemy/Training/GenAI-Training/Codelabs-Format/Lab6-AI-Agents/Lab6-AI-Agents.ipynb")

    cell_29 = get_cell_source(notebook, 'cell-29')
    if not cell_29:
        print("  ✗ Cell 29: Not found")
        return False

    # Check that Python 3.9+ syntax is removed
    if '-> tuple[bool, Any]' in cell_29:
        print("  ✗ Cell 29: Still has Python 3.9+ type hint syntax")
        return False
    else:
        print("  ✓ Cell 29: Python 3.9+ syntax removed")

    # Check that return type is documented
    if 'Tuple[bool, Any]' in cell_29 or 'Returns:' in cell_29:
        print("  ✓ Cell 29: Return type documented in docstring")
    else:
        print("  ⚠ Cell 29: Return type not clearly documented")

    return True


def test_lab7_fixes():
    """Test Lab7 specific fixes"""
    print("\n[LAB 7] Testing eval() security fix")

    notebook = read_notebook("/Users/firstlinkconsultingllc/udemy/Training/GenAI-Training/Codelabs-Format/Lab7-Agent-Memory/Lab7-Agent-Memory.ipynb")

    cell_17 = get_cell_source(notebook, 'cell-17')
    if not cell_17:
        print("  ✗ Cell 17: Not found")
        return False

    # Check that eval() is NOT used
    if 'result = eval(expression)' in cell_17:
        print("  ✗ Cell 17: SECURITY RISK - Still uses eval()")
        return False
    else:
        print("  ✓ Cell 17: eval() removed (security fix)")

    # Check that safe alternative is used
    if 'safe_eval_math' in cell_17 or 'ast.parse' in cell_17:
        print("  ✓ Cell 17: Safe math evaluator implemented")
    else:
        print("  ✗ Cell 17: Missing safe evaluation method")
        return False

    return True


def test_lab8_fixes():
    """Test Lab8 specific fixes"""
    print("\n[LAB 8] Testing eval() security fix")

    notebook = read_notebook("/Users/firstlinkconsultingllc/udemy/Training/GenAI-Training/Codelabs-Format/Lab8-Advanced-Agents/Lab8-Advanced-Agents.ipynb")

    cell_16 = get_cell_source(notebook, 'cell-16')
    if not cell_16:
        print("  ✗ Cell 16: Not found")
        return False

    # Check that eval() is NOT used
    if 'result = eval(expression)' in cell_16:
        print("  ✗ Cell 16: SECURITY RISK - Still uses eval()")
        return False
    else:
        print("  ✓ Cell 16: eval() removed (security fix)")

    # Check that safe alternative is used
    if 'safe_eval_math' in cell_16 or 'ast.parse' in cell_16:
        print("  ✓ Cell 16: Safe math evaluator implemented")
    else:
        print("  ✗ Cell 16: Missing safe evaluation method")
        return False

    return True


def main():
    """Run all specific fix tests"""
    print("="*70)
    print("TESTING SPECIFIC FIXES IN ALL LABS")
    print("="*70)

    tests = [
        ("Lab 1", test_lab1_fixes),
        ("Lab 2", test_lab2_fixes),
        ("Lab 3", test_lab3_fixes),
        ("Lab 4", test_lab4_fixes),
        ("Lab 5", test_lab5_fixes),
        ("Lab 6", test_lab6_fixes),
        ("Lab 7", test_lab7_fixes),
        ("Lab 8", test_lab8_fixes),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("FIX VERIFICATION SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print("="*70)
    print(f"TOTAL: {passed_count}/{total_count} labs verified")
    print("="*70)

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
