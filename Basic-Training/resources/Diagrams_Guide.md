# Visual Diagrams for Course

This document contains all the visual diagrams used throughout the course. These are provided in Mermaid format (renders in GitHub, many markdown viewers, and can be converted to images).

---

## RAG System Architecture

### Complete RAG Pipeline

```mermaid
flowchart TD
    A[User Query] --> B[Embed Query]
    B --> C[Vector Search]
    C --> D{Relevant<br/>Documents<br/>Found?}
    D -->|Yes| E[Retrieve Top-K<br/>Documents]
    D -->|No| F[Fallback Response]
    E --> G[Format Context]
    G --> H[Augment Prompt]
    H --> I[LLM Generation]
    I --> J[Return Answer<br/>+ Sources]
    F --> J

    style A fill:#e1f5ff
    style J fill:#c8e6c9
    style D fill:#fff9c4
```

### Document Processing Flow

```mermaid
flowchart LR
    A[Raw Documents] --> B[Load Documents]
    B --> C[Clean Text]
    C --> D[Chunk Text]
    D --> E[Generate<br/>Embeddings]
    E --> F[Store in<br/>Vector DB]

    D -.->|metadata| F

    style A fill:#ffebee
    style F fill:#e8f5e9
```

### Vector Search Visualization

```
Query: "How do I return a product?"
   ↓ [Embedding Model]
Vector: [0.23, -0.45, 0.67, 0.12, ...]
   ↓
┌─────────────────────────────────────┐
│     Vector Database (ChromaDB)      │
│                                     │
│  Doc1: [0.25, -0.43, 0.69, ...]  ←─ Distance: 0.05 ✓
│  Doc2: [0.89, 0.12, -0.34, ...]  ←─ Distance: 0.87
│  Doc3: [0.22, -0.47, 0.65, ...]  ←─ Distance: 0.07 ✓
│  Doc4: [-0.56, 0.78, 0.34, ...] ←─ Distance: 0.92
│                                     │
└─────────────────────────────────────┘
   ↓
Return: Doc1, Doc3 (most similar)
```

---

## Agent System Architecture

### Simple Agent Loop (ReAct Pattern)

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant T as Tools
    participant L as LLM

    U->>A: Query
    loop Until Task Complete
        A->>L: Think (Reasoning)
        L->>A: Thought + Action Plan
        A->>T: Execute Tool
        T->>A: Observation
        A->>L: Process Result
        L->>A: Decision (Continue/Done)
    end
    A->>U: Final Answer
```

### Agent Execution Flow

```mermaid
flowchart TD
    A[Receive Query] --> B{Understand<br/>Task}
    B --> C[Plan Actions]
    C --> D[Select Tool]
    D --> E[Execute Tool]
    E --> F{Success?}
    F -->|No| G[Handle Error]
    G --> D
    F -->|Yes| H[Process Result]
    H --> I{Task<br/>Complete?}
    I -->|No| C
    I -->|Yes| J[Generate Response]
    J --> K[Return to User]

    style A fill:#e3f2fd
    style K fill:#c8e6c9
    style F fill:#fff9c4
    style I fill:#fff9c4
```

### Tool Calling Process

```mermaid
sequenceDiagram
    participant LLM
    participant System
    participant Tool

    LLM->>System: Function Call Request<br/>{name: "weather", args: {"city": "NYC"}}
    System->>Tool: Execute Function
    Tool->>System: Return Result<br/>{"temp": 72, "condition": "sunny"}
    System->>LLM: Tool Response
    LLM->>System: Final Answer<br/>"The weather in NYC is 72°F and sunny"
```

---

## Multi-Agent Systems

### Hierarchical Multi-Agent Architecture

```mermaid
graph TD
    A[User Query] --> B[Orchestrator Agent]
    B --> C{Route to<br/>Specialist}
    C -->|General| D[Support Agent]
    C -->|Technical| E[Technical Agent]
    C -->|Sales| F[Sales Agent]
    C -->|Complex| G[Escalation Agent]

    D --> H[Knowledge Base]
    E --> I[Technical Docs]
    F --> J[Product Catalog]

    D --> K[Response]
    E --> K
    F --> K
    G --> K
    K --> L[Return to User]

    style B fill:#ffecb3
    style D fill:#c5e1a5
    style E fill:#c5e1a5
    style F fill:#c5e1a5
    style G fill:#ef9a9a
```

### Router Agent Decision Tree

```mermaid
flowchart TD
    A[Incoming Query] --> B{Classify Intent}
    B -->|"How to..."| C[Support Agent]
    B -->|"Error code..."| D[Technical Agent]
    B -->|"Price..."| E[Sales Agent]
    B -->|"Complaint..."| F[Escalation Agent]

    C --> G{Can<br/>Resolve?}
    D --> G
    E --> G

    G -->|Yes| H[Respond]
    G -->|No| F
    F --> H

    style B fill:#fff9c4
    style G fill:#fff9c4
    style F fill:#ffcdd2
```

### Sequential Agent Workflow

```mermaid
sequenceDiagram
    participant U as User
    participant A1 as Research Agent
    participant A2 as Critic Agent
    participant A3 as Writer Agent

    U->>A1: Task: "Write article about AI"
    A1->>A1: Research topic
    A1->>A2: Draft research
    A2->>A2: Critique & feedback
    A2->>A3: Research + Critique
    A3->>A3: Write article
    A3->>U: Final article
```

---

## LLM Concepts

### Token Visualization

```
Text: "Hello, how are you?"
   ↓ [Tokenization]
Tokens: ["Hello", ",", " how", " are", " you", "?"]
Token IDs: [15496, 11, 703, 389, 345, 30]
Count: 6 tokens

Cost calculation:
Input: 6 tokens
Output: ~12 tokens (estimated)
Total: 18 tokens

Price (GPT-3.5-turbo):
Input:  6 tokens × $0.0000005 = $0.000003
Output: 12 tokens × $0.0000015 = $0.000018
Total: $0.000021 per query
```

### Temperature Effect

```
Temperature = 0.0 (Deterministic)
Query: "Name a color"
Responses: "Blue", "Blue", "Blue", "Blue"

Temperature = 0.7 (Balanced)
Query: "Name a color"
Responses: "Blue", "Red", "Blue", "Green"

Temperature = 1.5 (Creative)
Query: "Name a color"
Responses: "Cerulean", "Magenta", "Ochre", "Vermillion"
```

### Context Window

```
┌─────────────────────────────────────┐
│     LLM Context Window (4K tokens)  │
│                                     │
│  System Prompt:     200 tokens      │
│  Conversation:    2,500 tokens      │
│  RAG Context:     1,000 tokens      │
│  ──────────────────────────         │
│  Used:            3,700 tokens      │
│  Available:         300 tokens      │
│  (for response)                     │
└─────────────────────────────────────┘

Warning: Context full! Response truncated or conversation summarized.
```

---

## Embeddings & Vector Search

### Embedding Space Visualization

```
2D projection of 384-dimensional space:

                 documents
                     ↓
    "refund" ●  ● "return policy"
              \ |
               \|
                ● "exchange"
                |


            ● "shipping"
           /
          /
    ● "delivery"        ● "payment"


Semantic similarity = Euclidean distance (or cosine similarity)
```

### Similarity Scoring

```
Query: "How to return items?"
Embedding: [0.23, -0.45, 0.67, ...]

Documents:
┌─────────────────────────────────────────┬───────┐
│ Document                                │ Score │
├─────────────────────────────────────────┼───────┤
│ "Return Policy: 30-day return..."      │ 0.95  │ ✓ High
│ "Shipping Information: We offer..."    │ 0.45  │
│ "Account Management: Create..."        │ 0.23  │
│ "Product Specifications: This..."      │ 0.12  │ ✗ Low
└─────────────────────────────────────────┴───────┘

Return top 2 documents (score > 0.7)
```

---

## Production Architecture

### Full System Architecture

```mermaid
graph TB
    subgraph "Frontend"
        A[Gradio UI]
        B[React App]
    end

    subgraph "Backend"
        C[FastAPI Server]
        D[Load Balancer]
    end

    subgraph "AI Layer"
        E[LLM APIs]
        F[RAG System]
        G[Agent Orchestrator]
    end

    subgraph "Data Layer"
        H[(PostgreSQL)]
        I[(ChromaDB)]
        J[(Redis Cache)]
    end

    subgraph "Monitoring"
        K[Logging]
        L[Metrics]
        M[Alerts]
    end

    A --> D
    B --> D
    D --> C
    C --> E
    C --> F
    C --> G
    F --> I
    G --> E
    C --> H
    C --> J
    C --> K
    K --> L
    L --> M

    style E fill:#e1bee7
    style F fill:#c5cae9
    style G fill:#c5cae9
```

### Deployment Pipeline

```mermaid
flowchart LR
    A[Code Push] --> B[GitHub Actions]
    B --> C[Run Tests]
    C --> D{Tests Pass?}
    D -->|No| E[Notify Developer]
    D -->|Yes| F[Build Docker Image]
    F --> G[Push to Registry]
    G --> H[Deploy to Staging]
    H --> I[Run Integration Tests]
    I --> J{Tests Pass?}
    J -->|No| E
    J -->|Yes| K[Deploy to Production]
    K --> L[Monitor]

    style A fill:#e3f2fd
    style K fill:#c8e6c9
    style E fill:#ffcdd2
```

### Caching Strategy

```mermaid
sequenceDiagram
    participant U as User
    participant API as API Server
    participant C as Redis Cache
    participant LLM as LLM API

    U->>API: Query
    API->>C: Check Cache
    alt Cache Hit
        C->>API: Cached Response
        API->>U: Return Response (Fast!)
    else Cache Miss
        API->>LLM: API Call
        LLM->>API: Response
        API->>C: Store in Cache
        API->>U: Return Response
    end
```

---

## Evaluation Framework

### Testing Pipeline

```mermaid
flowchart TD
    A[Test Suite] --> B[Unit Tests]
    A --> C[Integration Tests]
    A --> D[E2E Tests]

    B --> E{All Pass?}
    C --> E
    D --> E

    E -->|No| F[Identify Failures]
    E -->|Yes| G[Quality Metrics]

    F --> H[Fix Issues]
    H --> A

    G --> I[Accuracy Score]
    G --> J[Latency Score]
    G --> K[Cost Score]

    I --> L{Meets<br/>Threshold?}
    J --> L
    K --> L

    L -->|No| F
    L -->|Yes| M[Approved for Deployment]

    style M fill:#c8e6c9
    style F fill:#ffcdd2
```

### Quality Metrics Dashboard

```
┌─────────────────────────────────────────────────────┐
│           SupportGenie Quality Dashboard            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Response Accuracy:    ████████████░░  92%  ✓      │
│  Source Attribution:   ██████████████  95%  ✓      │
│  Response Time (avg):  1.2s                ✓      │
│  Cost per Query:       $0.08               ✓      │
│  User Satisfaction:    ████████████░░  4.6/5  ✓   │
│                                                     │
│  Common Issues (last 24h):                         │
│  1. Order tracking queries: 127 (resolved: 98%)    │
│  2. Return policy questions: 89 (resolved: 100%)   │
│  3. Account access: 45 (escalated: 8%)             │
│                                                     │
│  Performance Trends: ↗ Improving                    │
└─────────────────────────────────────────────────────┘
```

---

## Error Handling Flow

### Retry Logic with Exponential Backoff

```mermaid
flowchart TD
    A[API Call] --> B{Success?}
    B -->|Yes| C[Return Response]
    B -->|No| D{Retryable<br/>Error?}
    D -->|No| E[Return Error]
    D -->|Yes| F{Attempts<br/>< Max?}
    F -->|No| E
    F -->|Yes| G[Wait<br/>2^attempt seconds]
    G --> A

    style C fill:#c8e6c9
    style E fill:#ffcdd2
```

### Error Recovery Strategy

```
┌─────────────────────────────────────┐
│        Error Occurred               │
└───────────┬─────────────────────────┘
            │
            ↓
    ┌───────────────┐
    │  Classify     │
    │  Error Type   │
    └───────┬───────┘
            │
      ┌─────┴─────┬─────────┬──────────┐
      ↓           ↓         ↓          ↓
┌──────────┐ ┌────────┐ ┌──────┐ ┌─────────┐
│ Rate     │ │Network │ │  API │ │  Model  │
│ Limit    │ │ Error  │ │ Error│ │  Error  │
└────┬─────┘ └───┬────┘ └───┬──┘ └────┬────┘
     │           │           │         │
     ↓           ↓           ↓         ↓
  Wait 60s    Retry 3x   Log &    Fallback
   Retry               Escalate    Model
     │           │           │         │
     └───────────┴───────────┴─────────┘
                      │
                      ↓
              ┌────────────────┐
              │  Return to     │
              │  User with     │
              │  Graceful Msg  │
              └────────────────┘
```

---

## Security Architecture

### Input Validation Flow

```mermaid
flowchart TD
    A[User Input] --> B[Length Check]
    B --> C{< Max Length?}
    C -->|No| D[Reject: Too Long]
    C -->|Yes| E[Sanitize Input]
    E --> F[Check for Injection]
    F --> G{Malicious<br/>Pattern?}
    G -->|Yes| H[Reject: Security Risk]
    G -->|No| I[Content Moderation]
    I --> J{Appropriate<br/>Content?}
    J -->|No| K[Reject: Inappropriate]
    J -->|Yes| L[Process Request]

    style L fill:#c8e6c9
    style D fill:#ffcdd2
    style H fill:#ffcdd2
    style K fill:#ffcdd2
```

---

## How to Use These Diagrams

### In Markdown Files
Mermaid diagrams render automatically in:
- GitHub README files
- GitLab
- Many markdown editors (VS Code with extension, Typora, etc.)

### Convert to Images
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Convert to PNG
mmdc -i diagram.mmd -o diagram.png

# Convert to SVG
mmdc -i diagram.mmd -o diagram.svg
```

### In Jupyter Notebooks
```python
from IPython.display import Image, display

# Display image
display(Image(filename='diagram.png'))
```

### Online Editors
- [Mermaid Live Editor](https://mermaid.live/)
- [Draw.io](https://draw.io)
- [Excalidraw](https://excalidraw.com/)

---

## Creating Your Own Diagrams

### Template: Basic Flowchart
```mermaid
flowchart TD
    A[Start] --> B[Process]
    B --> C{Decision}
    C -->|Yes| D[Action 1]
    C -->|No| E[Action 2]
    D --> F[End]
    E --> F
```

### Template: Sequence Diagram
```mermaid
sequenceDiagram
    participant A as Actor A
    participant B as Actor B
    A->>B: Message
    B->>A: Response
```

---

**All diagrams are provided in the course materials and can be copied/modified as needed!**
