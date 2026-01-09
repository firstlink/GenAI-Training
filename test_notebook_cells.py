#!/usr/bin/env python3
"""
Test actual notebook cells for syntax errors
"""

import json
import sys

def test_notebook_cells(notebook_path, lab_name):
    """Test all code cells in a notebook for syntax errors"""
    print(f"\n{'='*70}")
    print(f"Testing {lab_name}")
    print('='*70)

    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"✗ Failed to load notebook: {e}")
        return False

    cells = notebook.get('cells', [])
    code_cells = [c for c in cells if c.get('cell_type') == 'code']

    print(f"Found {len(code_cells)} code cells")

    errors = []
    passed = 0

    for i, cell in enumerate(code_cells):
        source = cell.get('source', [])
        if isinstance(source, list):
            code = ''.join(source)
        else:
            code = source

        # Skip empty cells and cells with just comments/magic commands
        code_stripped = code.strip()
        if not code_stripped:
            continue

        # Skip cells that start with magic commands or contain them
        if code_stripped.startswith('!') or code_stripped.startswith('%'):
            continue

        # Skip cells that have magic commands in them (like !pip install)
        if '!pip' in code or '!python' in code or '%' in code[:10]:
            continue

        # Try to compile the code
        try:
            compile(code, f'<cell-{i}>', 'exec')
            passed += 1
        except SyntaxError as e:
            errors.append({
                'cell': i,
                'error': str(e),
                'line': e.lineno,
                'code_preview': code[:100]
            })

    # Report results
    if errors:
        print(f"\n✗ Found {len(errors)} syntax errors:")
        for err in errors[:5]:  # Show first 5 errors
            print(f"\n  Cell {err['cell']}:")
            print(f"    Line {err['line']}: {err['error']}")
            print(f"    Code: {err['code_preview']}...")
    else:
        print(f"✓ All {passed} code cells have valid syntax")

    return len(errors) == 0


def main():
    """Test all notebooks"""
    base_path = "/Users/firstlinkconsultingllc/udemy/Training/GenAI-Training/Codelabs-Format"

    labs = [
        ("Lab1-LLM-Fundamentals/Lab1-LLM-Fundamentals.ipynb", "Lab 1 - LLM Fundamentals"),
        ("Lab2-Prompt-Engineering/Lab2-Prompt-Engineering.ipynb", "Lab 2 - Prompt Engineering"),
        ("Lab3-Document-Processing/Lab3-Document-Processing.ipynb", "Lab 3 - Document Processing"),
        ("Lab4-Semantic-Search/Lab4-Semantic-Search.ipynb", "Lab 4 - Semantic Search"),
        ("Lab5-RAG-Pipeline/Lab5-RAG-Pipeline.ipynb", "Lab 5 - RAG Pipeline"),
        ("Lab6-AI-Agents/Lab6-AI-Agents.ipynb", "Lab 6 - AI Agents"),
        ("Lab7-Agent-Memory/Lab7-Agent-Memory.ipynb", "Lab 7 - Agent Memory"),
        ("Lab8-Advanced-Agents/Lab8-Advanced-Agents.ipynb", "Lab 8 - Advanced Agents"),
    ]

    results = []
    for notebook_file, lab_name in labs:
        notebook_path = f"{base_path}/{notebook_file}"
        passed = test_notebook_cells(notebook_path, lab_name)
        results.append((lab_name, passed))

    # Summary
    print(f"\n{'='*70}")
    print("NOTEBOOK SYNTAX TEST SUMMARY")
    print('='*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print('='*70)
    print(f"TOTAL: {passed_count}/{total_count} notebooks passed")
    print('='*70)

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
