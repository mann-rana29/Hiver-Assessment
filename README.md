# Hiver AI Support Agent (AppleSupport)

This repository contains the code for an AI customer support agent designed to handle Twitter interactions for AppleSupport. It uses LangChain and the Gemini API to classify intent, decide on escalation, and draft responses based on historical context.

## Setup Instructions (Reproducible in < 15 minutes)

### 1. Environment Setup
Create a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API Keys
Rename `.env.example` to `.env` and fill in your keys:
- `GEMINI_API_KEY`: Your Google Gemini API Key.
- `GEMINI_MODEL`: `gemini-1.5-flash` (or another model of your choice).
- `KAGGLE_USERNAME` & `KAGGLE_KEY`: Your Kaggle API credentials (used for downloading the dataset).

### 3. Data Preparation
Run the data preparation script. This will download the Kaggle dataset, filter for AppleSupport, and create a sample to be used as a golden set.
```bash
python -m src.data_prep
```
*Note: A sample `golden_set_template.csv` will be generated in the `data/` directory. For the pipeline to work end-to-end, you must manually label (fill in 'intent', 'auto_handle', 'escalate_reason') a subset of rows and save it as `data/golden_set.csv`.*

### 4. Run Evaluation
Run the evaluation harness to test the agent against the golden set. This will generate a `metrics.json` file in the `data/` directory.
```bash
python -m src.evaluate
```

## Architecture
- `src/agent.py`: Contains the `AISupportAgent` class which uses a LangChain pipeline for Intent Classification (with Structured Output parsing) and a RAG approach using ChromaDB for Response Drafting.
- `src/evaluate.py`: Runs the test set and implements an LLM-as-a-judge system (using LangChain) to grade the generated responses against human benchmarks.
