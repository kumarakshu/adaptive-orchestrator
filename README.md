# Adaptive LLM Orchestrator

An intelligent, multi-tier orchestration engine for Large Language Models (LLMs) built with FastAPI, Pydantic, and HTTPX.

## Overview

The **Adaptive LLM Orchestrator** is designed to dynamically evaluate incoming prompt queries, route them to the most cost-effective model tier capable of solving the task, validate response quality, and automatically escalate to higher-capability models when validation criteria or confidence thresholds are not satisfied.

### Architecture Highlights
- **Prompt Analyzer**: Analyzes query complexity, domain, token budget, and intent.
- **Intelligent Router**: Directs traffic to optimal model tiers (e.g., fast/lightweight, balanced, reasoning/heavy).
- **Model Registry**: Centralized metadata and capability definitions for registered LLMs.
- **Response Validator**: Assesses response completeness, format conformity, and quality signals.
- **Escalation Engine**: Manages fallback strategies, retries, and dynamic model upgrades.
- **Modular Boundaries**: Decoupled executor and cache/context layers allowing independent integration with provider connectors and Redis context management.

---

## Project Structure

```text
app/
    __init__.py
    main.py
    api/
        __init__.py
        routes.py
        schemas.py
    core/
        __init__.py
        config.py
    orchestrator/
        __init__.py
        analyzer.py
        router.py
        executor.py
        escalation.py
        validator.py
        engine.py
    models/
        __init__.py
        registry.py
tests/
    test_analyzer.py
    test_router.py
    test_escalation.py
    test_orchestrator.py
```

---

## Getting Started

### 1. Prerequisites
- Python 3.11+
- Virtual environment tool (`venv`)

### 2. Setup Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 5. Run the Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Run Tests
```bash
pytest
```

