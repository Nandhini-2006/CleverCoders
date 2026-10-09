# Agentic Document Answering System

An agentic document question-answering system that provides **grounded, evidence-based answers from PDF documents** using controlled document tools, information-gain-based action selection, and a strict execution budget.

The system is designed to reduce hallucinations, handle conflicting information, defend against prompt injection, and stop searching when sufficient evidence has been collected.

## Overview

Instead of blindly searching a fixed number of chunks, the system dynamically decides **which document action should be performed next** based on the current information available.

The agent operates under a deterministic document-tool budget of **6 calls**. Every document-tool call is evaluated by the decision process and passed through the budget controller.

## Key Features

- Agentic document reasoning
- Information-gain-based action selection
- Strict 6-call document-tool budget
- PDF-grounded question answering
- Keyword-based document search
- Heading and page discovery
- Trigram matching for typo-tolerant search
- Levenshtein distance for similar-term matching
- Aho–Corasick multi-keyword search
- Relevance scoring
- Evidence-based answer generation
- Contradiction detection
- Temporal supersession handling
- Insufficient-information detection
- Prompt-injection protection
- Answer validation
- Page-level evidence and citations
- Early stopping when sufficient evidence is available

## Algorithms Used

### Shannon Entropy

Used to measure the **uncertainty of the current search state**. It helps the agent understand whether it has enough confidence about where the required information exists.

### Information Gain

Used to determine **which action is most useful to perform next**. The agent prefers the action that is expected to provide the most useful information and reduce uncertainty.

### Keyword Matching

Used to identify pages containing important terms from the user's query.

### Trigram Matching

Words are divided into three-character sequences to improve matching when the query contains small spelling variations or partial matches.

### Levenshtein Distance

Measures the number of edits required to transform one word into another. It helps the system handle spelling mistakes and find similar terms.

### Aho–Corasick Algorithm

Used for efficient matching of multiple keywords simultaneously within document text.

### Relevance Scoring

Candidate pages are assigned relevance based on how closely their content matches the user's query. More relevant pages receive higher priority during evidence collection.

### Contradiction Detection

Identifies situations where different pages provide conflicting information. Instead of guessing, the system can report an unresolved contradiction when there is not enough evidence to determine which statement is correct.

### Temporal Supersession

Checks whether newer information explicitly replaces older information. When a later statement supersedes an earlier one, the newer evidence is preferred.

### Evidence Validation

Checks whether the generated answer is properly supported by the evidence collected from the document.

## Mathematical Approach

The system follows a relationship between **query relevance, probability, uncertainty, information gain, and action selection**.

**Query → Relevance Scoring → Probability Distribution → Shannon Entropy → Information Gain → Action Selection → Evidence Collection → Validation**

- **Relevance scoring** determines how strongly each page relates to the user's query.
- These relevance values are used to estimate a **probability distribution** over candidate pages.
- **Shannon Entropy** measures the uncertainty in that distribution.
- **Information Gain** determines which available action can reduce that uncertainty most effectively.
- The agent selects the most informative action while respecting the **6-call tool budget**.
- The result of the selected action updates the current evidence and search state.
- The process continues until sufficient evidence is collected or the available budget is exhausted.
- **Evidence validation** then checks whether the final answer is supported by the collected evidence.

## Document Tools

The agent interacts with documents through controlled tools.

### `list_documents()`

Returns available document information such as document ID and total page count.

### `list_headings(doc_id)`

Returns the document structure, including headings, levels, and page numbers.

### `search_keyword(doc_id, keyword)`

Searches the document for a keyword and returns matching pages and relevant snippets.

### `get_page(doc_id, page_number)`

Retrieves the complete content of a specific page.

## Security

Document content is treated as **untrusted data**.

Instructions found inside a PDF cannot modify the agent's system instructions or decision-making process.

The system follows these security principles:

- Document content cannot modify system instructions
- The LLM does not directly access the filesystem
- Document access occurs through controlled tools
- Retrieved content is treated as untrusted evidence
- API keys are protected from exposure in logs
- Tool execution is controlled by the budget controller

## Contradiction Handling

The system does not automatically select one statement when the document contains conflicting information.

For example, if two pages provide different values and there is no evidence showing which statement is correct or newer, the system identifies the situation as an **unresolved contradiction** rather than guessing.

## Temporal Supersession

When the document contains older and newer information, the system checks whether the newer information explicitly replaces the older information.

If the newer statement clearly supersedes the previous statement, the newer evidence is preferred.

## Insufficient Information

If the requested information cannot be found in the document, the system does not fabricate an answer.

Instead, it reports that there is **insufficient information in the provided document**.

## Agent Decision Process

The agent continuously evaluates the current evidence and determines the most useful next action.

The decision process considers:

- Current evidence
- Page relevance
- Search uncertainty
- Expected information gain
- Remaining tool budget
- Evidence sufficiency

The agent stops when sufficient evidence has been collected, avoiding unnecessary document operations.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Streamlit | User interface |
| NVIDIA Nemotron | Agent reasoning and language generation |
| PDF Processing | Document extraction |
| Shannon Entropy | Uncertainty measurement |
| Information Gain | Action selection |
| Trigram Matching | Typo-tolerant search |
| Levenshtein Distance | Similar-term matching |
| Aho–Corasick | Multi-keyword search |
| Pytest | Automated testing |
| JSONL | Structured logging |

## Design Principles

### Evidence Over Guessing

The system prioritizes information supported by the document instead of generating unsupported information.

### Adaptive Actions

The next action depends on the current state of the investigation.

### Fixed Budget

The agent cannot perform unlimited document operations and must operate within the defined tool budget.

### Explicit Uncertainty

When evidence is insufficient or contradictory, the system reports the uncertainty instead of guessing.

### Secure Document Access

The language model interacts with documents through controlled tools rather than unrestricted filesystem access.

### Early Stopping

The agent stops when enough evidence has been collected, reducing unnecessary tool calls.

## Future Enhancements

- Improved typo-tolerant search
- Enhanced relevance scoring
- Improved evidence ranking
- Multi-document support
- Additional document formats
- Agent observability
- Evaluation dashboard
- Local LLM support

## License

This project is intended for educational, research, and development purposes.
