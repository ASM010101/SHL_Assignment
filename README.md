# Conversational SHL Assessment Recommender

A production-grade, stateless conversational recommender agent built on FastAPI and Google Gemini that guides recruiters and hiring managers from vague requirements (e.g. "I am hiring a Java developer") to a targeted shortlist of SHL assessments.

## Features

- **Stateless API**: No conversation state stored on server; state is reconstructed on every request from message history.
- **Deterministic Conversation Planner**: Orchestrates transitions between Greeting, Clarifying, Recommending, Refining, and Comparing using strict rules to honor turn caps.
- **Hybrid Retrieval Pipeline**: Combines FAISS-based vector search (using local `all-MiniLM-L6-v2` embeddings) with TF-IDF keyword search using Reciprocal Rank Fusion (RRF) and metadata boosting.
- **Strict Output Validation**: Validates all generated assessment URLs and names against the catalog before sending, ensuring zero hallucinations.
- **Layered Guardrails**: Input validation, prompt injection blocking, off-topic filtering, and legal question refusals.
- **Precomputed Enrichment**: Generates additional metadata (technologies, domains, traits) for the 377 Individual Test Solutions catalog items during the Docker build stage.

## Repository Structure

```
├── app/
│   ├── api/
│   │   ├── routes.py                  # FastAPI route endpoints (/health, /chat)
│   │   └── schemas.py                 # Pydantic request/response models
│   ├── agents/
│   │   ├── clarifier.py              # Prioritized clarification policy
│   │   ├── comparator.py             # Grounded catalog-only comparisons
│   │   ├── conversation_analyzer.py   # Extracts structured HiringProfile from history
│   │   ├── conversation_planner.py    # Rule-based routing state machine
│   │   ├── guardrails.py             # Security & scope filters (injection/off-topic)
│   │   ├── intent_detector.py         # Greeting/Search/Comparison classification
│   │   ├── orchestrator.py            # Coordinates execution pipeline
│   │   └── recommender.py            # Retrieval and LLM reranking coordinator
│   ├── retrieval/
│   │   ├── embeddings.py             # Sentence-transformers local embedding model
│   │   ├── vector_store.py           # FAISS index vector store
│   │   ├── keyword_search.py         # TF-IDF keyword search engine
│   │   ├── hybrid_retriever.py       # RRF merged search pipeline
│   │   └── ranker.py                 # Metadata boosting and filtering
│   ├── catalog/
│   │   ├── loader.py                 # Loads JSON store with lookup dicts
│   │   └── enrichment.py             # Rule-based metadata tag expansion
│   ├── prompts/
│   │   └── templates.py              # Modular prompt instructions
│   ├── utils/
│   │   ├── helpers.py                # Formatting and abbreviations helpers
│   │   ├── logger.py                 # Request context logging
│   │   ├── llm_client.py             # Dual SDK + REST HTTP Gemini API wrapper
│   │   └── validators.py             # Pre-response schema/link checker
│   ├── config.py                      # Configurations and settings
│   └── main.py                        # Application lifecycle and initialization
├── data/
│   ├── shl_product_catalog.json      # Raw SHL catalog data
│   └── enriched_catalog.json         # Precomputed enriched catalog
├── tests/
│   ├── test_api.py                   # API routes integration tests
│   ├── test_catalog.py               # Catalog integrity & helpers tests
│   ├── test_edge_cases.py            # Turn caps & validator tests
│   ├── test_guardrails.py            # Injection & off-topic tests
│   └── test_schema.py                # Schema field structure tests
├── evaluation/
│   └── evaluate.py                   # Replays public traces & scores Mean Recall@10
├── scripts/
│   └── enrich_catalog.py             # Build-time offline catalog enrichment script
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── requirements.txt
├── .env.example
├── approach_document.md              # Maximum 2-page system architecture doc
└── README.md
```

## Setup & Running

### Prerequisites

- Python 3.10+
- Google Gemini API Key

### Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repo-url>
   cd Assign
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
   Add your `GEMINI_API_KEY` to the `.env` file.

### Running Local Server

Start the local development server:
```bash
python app/main.py
```
The API documentation will be available at `http://localhost:8000/docs` and the health endpoint at `http://localhost:8000/health`.

### Running Tests

Run the full pytest suite:
```bash
python -m pytest tests/
```

### Running Evaluation

Replay the C1-C10 conversation traces and compute Mean Recall@10:
```bash
python evaluation/evaluate.py
```

## API Specifications

### `GET /health`
Returns service status. Returns HTTP 200 `{"status": "ok"}` when the catalog is parsed and search indices are built.

### `POST /chat`
Accepts full stateless history and returns recommendations:
*   **Request Schema**:
    ```json
    {
      "messages": [
        {"role": "user", "content": "I am hiring a Java developer"},
        {"role": "assistant", "content": "Sure, what is the seniority level?"},
        {"role": "user", "content": "Senior, 5+ years of experience"}
      ]
    }
    ```
*   **Response Schema**:
    ```json
    {
      "reply": "Here are my recommended assessments for a Senior Java developer.",
      "recommendations": [
        {
          "name": "Core Java (Advanced Level) (New)",
          "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
          "test_type": "K"
        }
      ],
      "end_of_conversation": false
    }
    ```

## Deployment

### Docker
Build and run the container locally:
```bash
docker build -t recommender .
docker run -p 8000:8000 --env-file .env recommender
```

### Render Deployment
This project is configured with a `render.yaml` blueprint. Link your repository to Render, add your `GEMINI_API_KEY` under Environment Variables, and Deploy. The service will build via the Dockerfile, precompute the enrichment layers, and start automatically.
