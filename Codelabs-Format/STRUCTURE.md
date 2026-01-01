# Codelabs Structure Guide

## 📁 Directory Organization

```
Codelabs-Format/
├── README.md                    # Main landing page
├── STRUCTURE.md                 # This file
│
├── Lab1-LLM-Fundamentals/
│   ├── learning.md             # 📚 Theory & Concepts (30 min)
│   └── lab.md                  # 🛠️ Hands-On Exercises (60-90 min)
│
├── Lab2-Prompt-Engineering/
│   ├── learning.md             # 📚 Theory & Concepts
│   └── lab.md                  # 🛠️ Hands-On Exercises
│
├── Lab3-Document-Processing/
│   ├── learning.md             # 📚 Theory & Concepts
│   └── lab.md                  # 🛠️ Hands-On Exercises
│
├── Lab4-Semantic-Search/
│   ├── learning.md             # 📚 Theory & Concepts
│   └── lab.md                  # 🛠️ Hands-On Exercises
│
├── Lab5-RAG-Pipeline/
│   ├── learning.md             # 📚 Theory & Concepts
│   └── lab.md                  # 🛠️ Hands-On Exercises
│
├── Lab6-AI-Agents/
│   ├── learning.md             # 📚 Theory & Concepts
│   └── lab.md                  # 🛠️ Hands-On Exercises
│
├── Lab7-Agent-Memory/
│   ├── learning.md             # 📚 Theory & Concepts
│   └── lab.md                  # 🛠️ Hands-On Exercises
│
└── Lab8-Advanced-Agents/
    ├── learning.md             # 📚 Theory & Concepts
    └── lab.md                  # 🛠️ Hands-On Exercises
```

---

## 🎯 Two-File System

Each lab is split into **two complementary files**:

### 📚 `learning.md` - Theory & Concepts

**Purpose:** Understand before you code

**Contains:**
- ✅ Conceptual explanations
- ✅ Visual diagrams
- ✅ How things work
- ✅ When and why to use techniques
- ✅ Best practices
- ✅ Comparisons and trade-offs
- ✅ Quick knowledge checks

**Duration:** 20-40 minutes
**Format:** Reading with interactive quizzes
**Goal:** Build understanding

---

### 🛠️ `lab.md` - Hands-On Exercises

**Purpose:** Apply what you learned

**Contains:**
- ✅ Environment setup
- ✅ Step-by-step coding exercises
- ✅ Complete code examples
- ✅ Checkpoints and verification
- ✅ Challenges and extensions
- ✅ Troubleshooting guides
- ✅ Capstone project

**Duration:** 60-120 minutes
**Format:** Hands-on coding
**Goal:** Build skills and projects

---

## 🔄 Recommended Workflow

```
┌─────────────────────────────────────────────────┐
│  STEP 1: READ LEARNING.MD                      │
│  ─────────────────────────────────────────      │
│  • Read concepts                                │
│  • Study diagrams                               │
│  • Answer knowledge checks                      │
│  • Take notes                                   │
│  Duration: 30 min                               │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  STEP 2: SETUP ENVIRONMENT                      │
│  ─────────────────────────────────────────      │
│  • Install packages                             │
│  • Configure API keys                           │
│  • Verify setup                                 │
│  Duration: 10 min                               │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  STEP 3: COMPLETE LAB.MD EXERCISES             │
│  ─────────────────────────────────────────      │
│  • Work through exercises 1-7                   │
│  • Run all code examples                        │
│  • Verify with checkpoints                      │
│  • Debug and troubleshoot                       │
│  Duration: 45-60 min                            │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  STEP 4: BUILD CAPSTONE PROJECT                │
│  ─────────────────────────────────────────      │
│  • Read requirements                            │
│  • Code the solution                            │
│  • Test thoroughly                              │
│  • Verify success criteria                      │
│  Duration: 30-45 min                            │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  STEP 5: OPTIONAL CHALLENGES                    │
│  ─────────────────────────────────────────      │
│  • Try extension challenges                     │
│  • Experiment and modify                        │
│  • Build additional features                    │
│  Duration: Variable                             │
└─────────────────────────────────────────────────┘
```

---

## 📊 Learning vs Lab Comparison

| Aspect | learning.md | lab.md |
|--------|-------------|--------|
| **Goal** | Understand concepts | Build with code |
| **Format** | Reading + diagrams | Coding exercises |
| **Duration** | 20-40 min | 60-120 min |
| **Interactive** | Knowledge checks | Hands-on coding |
| **Output** | Mental model | Working code |
| **Required** | Yes (do first!) | Yes (do after learning) |
| **Repeatable** | Review as needed | Practice multiple times |

---

## 🎓 For Different Learning Styles

### Visual Learners
- Focus on diagrams in `learning.md`
- Draw your own flowcharts
- Visualize the code flow in `lab.md`

### Hands-On Learners
- Skim `learning.md` quickly
- Jump into `lab.md` exercises
- Refer back to `learning.md` when stuck

### Reading Learners
- Read `learning.md` thoroughly
- Take detailed notes
- Reference notes while doing `lab.md`

### Social Learners
- Form study groups
- Discuss `learning.md` concepts together
- Code `lab.md` exercises with partners

---

## 💡 Pro Tips

### For Instructors

**Workshop Format:**
1. **Pre-work:** Students read `learning.md` before class
2. **Class time:** Work through `lab.md` together
3. **Homework:** Complete capstone project
4. **Review:** Discuss challenges next session

**Flipped Classroom:**
1. **Assign:** `learning.md` as homework
2. **Quiz:** Quick check at start of class
3. **Lab:** Complete `lab.md` in class with support
4. **Discussion:** Share capstone solutions

---

### For Self-Learners

**First Time Through:**
- Don't skip `learning.md` - it saves time later
- Code along with examples in `lab.md`
- Take breaks between exercises
- Complete the capstone project

**Review/Practice:**
- Skim `learning.md` for quick reference
- Jump directly to specific exercises in `lab.md`
- Try the challenge problems
- Modify code to experiment

---

## 🔍 Finding What You Need

### Quick Reference
```
Need theory? → Open learning.md
Need code? → Open lab.md
Need both? → Start with learning.md, then lab.md
```

### Specific Topics

**Concepts & Theory:**
- How things work → `learning.md`
- Why use this technique → `learning.md`
- When to apply → `learning.md`
- Comparisons → `learning.md`

**Practical Code:**
- Setup instructions → `lab.md`
- Code examples → `lab.md`
- Exercises → `lab.md`
- Troubleshooting → `lab.md`
- Complete projects → `lab.md`

---

## ✅ Completion Checklist

For each lab, you've completed it when:

**learning.md:**
- [ ] Read all sections
- [ ] Understood key concepts
- [ ] Passed knowledge checks
- [ ] Can explain in your own words

**lab.md:**
- [ ] Setup environment successfully
- [ ] Completed all exercises
- [ ] Passed all checkpoints
- [ ] Built capstone project
- [ ] Verified it works

---

## 🚀 Getting Started

**New to the course?**
1. Start with [Lab 1 Learning Material](Lab1-LLM-Fundamentals/learning.md)
2. Then do [Lab 1 Hands-On Lab](Lab1-LLM-Fundamentals/lab.md)
3. Check off your progress in the main README

**Looking for specific topics?**
- Use the table of contents in README.md
- Each `learning.md` has its own TOC
- Each `lab.md` has exercise list

---

## 📝 Notes for Content Creators

When creating new labs, follow this structure:

### learning.md Template:
```markdown
# Lab X: [Title]
## 📚 Learning Material

## Overview
[Lab details table]

## Table of Contents
[Numbered sections]

## 1. Introduction
[Why this matters]

## 2-7. Core Concepts
[Theory, diagrams, explanations]

## 8. Review & Key Takeaways
[Summary, knowledge checks]

[Link to lab.md]
```

### lab.md Template:
```markdown
# Lab X: [Title]
## 🛠️ Hands-On Lab

## Overview
[Lab details table]

## Table of Contents
[Exercise list]

## 1. Setup
[Environment setup]

## 2-8. Exercises
[Step-by-step coding]

## 9. Capstone Project
[Complete project]

## 10. Challenges
[Optional extensions]

[Link to next lab]
```

---

## 🎯 Success Metrics

You're using this structure effectively if:

✅ You complete `learning.md` before `lab.md`
✅ You understand concepts before coding
✅ You can explain what you built
✅ You complete capstone projects
✅ Code runs without errors
✅ You feel confident moving to next lab

---

**Ready to start learning?**

👉 [Begin with Lab 1 Learning Material →](Lab1-LLM-Fundamentals/learning.md)
