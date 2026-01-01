# Templates for Creating Remaining Labs

This guide provides templates for creating all remaining components efficiently.

---

## 📚 Template 1: learning.md Structure

```markdown
# Lab X: [Title]
## 📚 Learning Material

> **Purpose:** Understand the theory and concepts before you code

---

## 📋 Overview

| Property | Value |
|----------|-------|
| **Duration** | XX minutes (reading) |
| **Difficulty** | [Beginner/Intermediate/Advanced] |
| **Prerequisites** | [List prerequisites] |
| **Next Step** | [Hands-On Lab →](lab.md) |

---

## 📖 Table of Contents

1. [Introduction](#1-introduction)
2. [Core Concept 1](#2-core-concept-1)
3. [Core Concept 2](#3-core-concept-2)
4. [Core Concept 3](#4-core-concept-3)
5. [Best Practices](#5-best-practices)
6. [Common Pitfalls](#6-common-pitfalls)
7. [Review & Key Takeaways](#7-review--key-takeaways)

---

## 1. Introduction

### What You'll Learn

[Bullet list of key concepts]

### Why This Matters

[Real-world applications and importance]

---

## 2. Core Concept 1

### Definition

[Clear explanation]

### How It Works

\`\`\`
[Visual diagram or flowchart]
\`\`\`

### Key Points

- ✅ Point 1
- ✅ Point 2
- ✅ Point 3

---

## 3. Core Concept 2

[Repeat pattern from Concept 1]

---

## 4. Core Concept 3

[Repeat pattern from Concept 1]

---

## 5. Best Practices

### Do's
- ✅ Practice 1
- ✅ Practice 2

### Don'ts
- ❌ Anti-pattern 1
- ❌ Anti-pattern 2

---

## 6. Common Pitfalls

### Pitfall 1: [Name]
**Problem:** [Description]
**Solution:** [How to avoid]

---

## 7. Review & Key Takeaways

### 🎯 What You've Learned

✅ **Concept 1:** [Summary]
✅ **Concept 2:** [Summary]
✅ **Concept 3:** [Summary]

### 🎓 Knowledge Check

<details>
<summary>Question 1: [Question text]</summary>
[Answer]
</details>

### 🚀 Ready for Hands-On Practice?

👉 **[Continue to Hands-On Lab →](lab.md)**

---

**Next:** [Hands-On Lab →](lab.md)
```

---

## 🛠️ Template 2: lab.md Structure

```markdown
# Lab X: [Title]
## 🛠️ Hands-On Lab

> **Purpose:** Apply what you learned through practical coding exercises

---

## 📋 Lab Overview

| Property | Value |
|----------|-------|
| **Duration** | XX-XX minutes (coding) |
| **Difficulty** | [Beginner/Intermediate/Advanced] |
| **Prerequisites** | Completed [learning.md](learning.md) |
| **What You'll Build** | [Main project name] |

---

## 📖 Table of Contents

1. [Setup Your Environment](#1-setup-your-environment)
2. [Exercise 1: [Name]](#2-exercise-1-name)
3. [Exercise 2: [Name]](#3-exercise-2-name)
4. [Exercise 3: [Name]](#4-exercise-3-name)
5. [Exercise 4: [Name]](#5-exercise-4-name)
6. [Exercise 5: [Name]](#6-exercise-5-name)
7. [Capstone: [Project Name]](#7-capstone-project-name)
8. [Challenges & Extensions](#8-challenges--extensions)

---

## 1. Setup Your Environment

### 🛠️ Step 1.1: Install Packages

\`\`\`bash
pip install [package1] [package2] [package3]
\`\`\`

### ✅ Step 1.2: Verify Installation

\`\`\`python
# test_setup.py
import package1
print("✅ Setup complete!")
\`\`\`

---

## 2. Exercise 1: [Name]

**Duration:** X minutes
**Objective:** [Clear goal]

### 🎯 Task 1.1: [Subtask Name]

Create \`exercise1.py\`:

\`\`\`python
# Complete code example here
\`\`\`

**Run it:**
\`\`\`bash
python exercise1.py
\`\`\`

**Expected output:**
\`\`\`
[Show expected output]
\`\`\`

### ✅ Checkpoint

**Verify:** [What student should check]

---

## 3-6. [Repeat Exercise Pattern]

---

## 7. Capstone: [Project Name]

**Duration:** XX minutes
**Objective:** [Build complete project]

### 🎯 Requirements

Build **[Project Name]** with:
- ✅ Feature 1
- ✅ Feature 2
- ✅ Feature 3

### 🎯 Implementation

Create \`capstone_[name].py\`:

\`\`\`python
# Complete capstone project code
\`\`\`

### 🎯 Test Scenarios

Try these test cases:
1. [Test case 1]
2. [Test case 2]

### ✅ Success Criteria

Your project should:
- ✅ Criterion 1
- ✅ Criterion 2

---

## 8. Challenges & Extensions

### 🏆 Challenge 1: [Name]
[Description and hints]

### 🏆 Challenge 2: [Name]
[Description and hints]

---

## 🎉 Congratulations!

You've completed Lab X!

### 🚀 Next Steps

👉 **[Lab X+1: [Next Lab Name] →](../LabX+1-Name/learning.md)**

---

**End of Lab X** ✅
```

---

## 💻 Template 3: Solution File

```python
#!/usr/bin/env python3
"""
Lab X - Exercise Y: [Exercise Name]
Complete Solution with Explanations

Author: [Your name]
Date: [Date]
"""

# ============================================================================
# IMPORTS
# ============================================================================
import module1
import module2
from dotenv import load_dotenv
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
load_dotenv()
API_KEY = os.getenv('API_KEY_NAME')

# ============================================================================
# SOLUTION
# ============================================================================

def main_function(parameter1, parameter2):
    """
    Clear docstring explaining what this does.

    Args:
        parameter1 (type): Description
        parameter2 (type): Description

    Returns:
        type: Description

    Example:
        >>> main_function("input", 42)
        "expected output"
    """
    # Step 1: [Explanation of this step]
    result = do_something(parameter1)

    # Step 2: [Explanation of this step]
    processed = process_result(result, parameter2)

    # Step 3: [Explanation of this step]
    return processed


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def helper_function():
    """Helper function docstring"""
    pass


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test the solution
    print("Testing solution...")

    # Test case 1
    result1 = main_function("test1", 10)
    print(f"Test 1 result: {result1}")

    # Test case 2
    result2 = main_function("test2", 20)
    print(f"Test 2 result: {result2}")

    print("\n✅ All tests passed!")


# ============================================================================
# NOTES AND EXPLANATIONS
# ============================================================================
"""
KEY CONCEPTS:
- Concept 1: Explanation
- Concept 2: Explanation

BEST PRACTICES DEMONSTRATED:
- Practice 1
- Practice 2

COMMON MISTAKES TO AVOID:
- Mistake 1
- Mistake 2

VARIATIONS:
You could also solve this by:
1. Alternative approach 1
2. Alternative approach 2
"""
```

---

## 📝 Template 4: Quiz File (Theory)

```markdown
# Lab X: Theory Quiz

**Duration:** 10 minutes
**Passing Score:** 80% (8/10)

---

## Multiple Choice Questions

### Question 1
What is [concept]?

A) Wrong answer
B) Wrong answer
C) Correct answer ✅
D) Wrong answer

**Explanation:** [Why C is correct]

---

### Question 2
When should you use [technique]?

A) Wrong answer
B) Correct answer ✅
C) Wrong answer
D) Wrong answer

**Explanation:** [Why B is correct]

---

## True/False Questions

### Question 3
Statement about concept. (True/False)

**Answer:** True ✅

**Explanation:** [Explanation]

---

## Fill in the Blank

### Question 4
Complete the sentence: "The purpose of [concept] is to ________"

**Answer:** [correct completion]

**Explanation:** [Explanation]

---

## Short Answer

### Question 5
Explain in 2-3 sentences: Why is [concept] important?

**Sample Answer:**
[Example good answer that covers key points]

**Grading Rubric:**
- ✅ Mentions point 1 (1 point)
- ✅ Mentions point 2 (1 point)
- ✅ Provides example (1 point)

---

## Answer Key

1. C
2. B
3. True
4. [Answer]
5. [See rubric]

**Scoring:**
- 9-10: Excellent
- 8: Pass
- 7 or below: Review material
```

---

## 🧪 Template 5: Quiz File (Practical)

```markdown
# Lab X: Practical Quiz

**Duration:** 20 minutes
**Format:** Coding challenges

---

## Challenge 1: [Name]

**Difficulty:** Easy
**Points:** 2

Write a function that:
- Takes [input]
- Returns [output]
- Handles [edge case]

\`\`\`python
def solution(input_param):
    # Your code here
    pass

# Test cases
assert solution("test") == "expected"
assert solution("edge_case") == "expected"
\`\`\`

**Solution:**
<details>
<summary>Click to reveal</summary>

\`\`\`python
def solution(input_param):
    # Implementation
    return result
\`\`\`

**Explanation:** [Why this works]
</details>

---

## Challenge 2: Debug This Code

**Difficulty:** Medium
**Points:** 3

Find and fix the bug:

\`\`\`python
def buggy_function(x):
    result = x + "10"  # Bug here
    return result
\`\`\`

**Solution:**
<details>
<summary>Click to reveal</summary>

**Bug:** Type mismatch - can't add string to number

**Fixed code:**
\`\`\`python
def fixed_function(x):
    result = x + 10  # Convert "10" to int
    return result
\`\`\`
</details>

---

## Challenge 3: Predict the Output

**Difficulty:** Medium
**Points:** 2

What does this code print?

\`\`\`python
x = [1, 2, 3]
y = x
y.append(4)
print(x)
\`\`\`

A) [1, 2, 3]
B) [1, 2, 3, 4] ✅
C) [4]
D) Error

**Explanation:** Lists are mutable; y references same list as x

---

## Scoring

- 10 points total
- 8+ points: Pass
- 6-7 points: Review and retry
- <6 points: Revisit lab material
```

---

## 📊 Template 6: Comprehensive Assessment

```markdown
# Lab X: Comprehensive Assessment

**Duration:** 60 minutes
**Type:** Project-based
**Passing:** Complete project meeting all criteria

---

## Project Brief

Build a [project type] that demonstrates mastery of:
- Concept 1
- Concept 2
- Concept 3

---

## Requirements

### Functional Requirements
1. [ ] Feature 1 implemented
2. [ ] Feature 2 implemented
3. [ ] Feature 3 implemented
4. [ ] Handles errors gracefully
5. [ ] User-friendly interface

### Technical Requirements
1. [ ] Code is well-organized
2. [ ] Functions have docstrings
3. [ ] Variables named clearly
4. [ ] Follows best practices from lab
5. [ ] No hardcoded values

### Quality Requirements
1. [ ] Code runs without errors
2. [ ] Passes all test cases
3. [ ] Efficient implementation
4. [ ] Comments explain complex logic

---

## Test Cases

Your project must pass these tests:

### Test 1: Basic Functionality
**Input:** [test input]
**Expected Output:** [expected output]

### Test 2: Edge Case
**Input:** [edge case]
**Expected Output:** [expected output]

### Test 3: Error Handling
**Input:** [invalid input]
**Expected Behavior:** [graceful error]

---

## Grading Rubric

| Category | Points | Criteria |
|----------|--------|----------|
| **Functionality** | 40 | All features work correctly |
| **Code Quality** | 30 | Clean, organized, documented |
| **Best Practices** | 20 | Follows lab guidelines |
| **Testing** | 10 | Passes all test cases |
| **Total** | 100 | |

**Grading Scale:**
- 90-100: Excellent (A)
- 80-89: Good (B)
- 70-79: Satisfactory (C)
- 60-69: Needs Improvement (D)
- <60: Incomplete (F)

---

## Submission

Submit:
1. Source code file(s)
2. README with usage instructions
3. Test results screenshot
4. (Optional) Video demo

---

## Sample Solution

<details>
<summary>Click after attempting (instructors only)</summary>

[Complete reference implementation]

**Key Points:**
- Point 1
- Point 2
- Point 3
</details>
```

---

## 🎨 Template 7: Visual Diagrams (Mermaid)

```markdown
## Flowchart Example

\`\`\`mermaid
flowchart TD
    A[Start] --> B{Decision?}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
    C --> E[End]
    D --> E
\`\`\`

## Sequence Diagram

\`\`\`mermaid
sequenceDiagram
    participant User
    participant API
    participant LLM

    User->>API: Send request
    API->>LLM: Forward to model
    LLM-->>API: Return response
    API-->>User: Display result
\`\`\`

## State Diagram

\`\`\`mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Processing: User input
    Processing --> Streaming: Start stream
    Streaming --> Complete: Finish
    Complete --> Idle: Reset
    Complete --> [*]
\`\`\`

## Architecture Diagram

\`\`\`mermaid
graph LR
    A[User] --> B[Frontend]
    B --> C[API Layer]
    C --> D[LLM Provider]
    C --> E[Vector DB]
    D --> F[Response]
    E --> F
    F --> B
\`\`\`
```

---

## 🌐 Template 8: HTML Conversion Script

```python
#!/usr/bin/env python3
"""
Convert Markdown Labs to HTML

Usage:
    python convert_to_html.py
"""

import markdown
from pathlib import Path
import re

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="../styles.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
</head>
<body>
    <nav class="breadcrumb">
        <a href="../index.html">Home</a> &gt;
        <a href="index.html">{lab_name}</a> &gt;
        <span>{page_type}</span>
    </nav>

    <div class="container">
        <aside class="sidebar">
            <h3>Navigation</h3>
            {nav_links}
        </aside>

        <main class="content">
            {content}
        </main>

        <aside class="toc">
            <h3>On This Page</h3>
            {toc}
        </aside>
    </div>

    <footer>
        <div class="pagination">
            {prev_link}
            {next_link}
        </div>
    </footer>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
</body>
</html>
"""

def convert_markdown_to_html(md_file, output_file):
    """Convert markdown file to HTML"""

    # Read markdown
    with open(md_file, 'r') as f:
        md_content = f.read()

    # Convert to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['fenced_code', 'tables', 'toc']
    )

    # Extract title
    title_match = re.search(r'# (.+)', md_content)
    title = title_match.group(1) if title_match else "Lab"

    # Generate final HTML
    final_html = HTML_TEMPLATE.format(
        title=title,
        lab_name="Lab Name",
        page_type="Learning",
        nav_links="<!-- Navigation -->",
        content=html_content,
        toc="<!-- TOC -->",
        prev_link="",
        next_link=""
    )

    # Write HTML
    with open(output_file, 'w') as f:
        f.write(final_html)

    print(f"✅ Converted {md_file} → {output_file}")


if __name__ == "__main__":
    # Convert all labs
    labs_dir = Path(".")

    for lab_dir in labs_dir.glob("Lab*"):
        if lab_dir.is_dir():
            learning_md = lab_dir / "learning.md"
            lab_md = lab_dir / "lab.md"

            if learning_md.exists():
                output = f"html-output/{lab_dir.name}_learning.html"
                convert_markdown_to_html(learning_md, output)

            if lab_md.exists():
                output = f"html-output/{lab_dir.name}_lab.html"
                convert_markdown_to_html(lab_md, output)
```

---

## 📋 Usage Instructions

### Creating a New Lab:

1. **Copy templates** for learning.md and lab.md
2. **Fill in content** following Lab 1 as example
3. **Add diagrams** using Mermaid or ASCII
4. **Create exercises** with clear objectives
5. **Build capstone** project
6. **Test thoroughly** - run all code

### Creating Solutions:

1. **Complete each exercise** yourself
2. **Add detailed comments** explaining logic
3. **Test with edge cases**
4. **Document alternatives**
5. **Save in** `/solutions/labX/` directory

### Creating Quizzes:

1. **Extract key concepts** from learning.md
2. **Create 10-15 questions** mixing types
3. **Include explanations** for all answers
4. **Test difficulty** level
5. **Save in** `/assessments/labX/` directory

### Converting to HTML:

1. **Run conversion script**
2. **Customize styling** in styles.css
3. **Test navigation** links
4. **Verify code highlighting**
5. **Check mobile responsiveness**

---

**Use these templates to efficiently create all remaining content!**
