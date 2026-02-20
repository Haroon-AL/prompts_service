# Prompts Service

A FastAPI-based service that utilizes Retrieval-Augmented Generation (RAG) to dynamically construct high-quality, structured extraction prompts for various e-commerce product attributes.

It indexes a provided CSV of historical prompt configurations into a local ChromaDB vector store, extracting the most semantically related examples to guide an Azure OpenAI LLM in generating the perfect extraction template for any requested attribute. But the expected use will require the {attributes,prompts} as csv or db connection.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- Azure OpenAI credentials

## Environment Variables

Create a `.env` file in the root directory (or export these in your shell):

```env
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"
AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

## Setup & Running

1. **Install Dependencies:**
   The project defines dependencies in `pyproject.toml`. You can sync them via `uv`:

   ```bash
   uv sync
   ```

2. **Start the API:**
   Run the application using `uvicorn`:

   ```bash
   uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
   ```

   *Note: On the first run, the `src.data_loader` will parse `data/public_llm_prompt_configuration_export.csv` and initialize the local ChromaDB semantic index (`data/chroma_db`).*

## API Endpoints

### `POST /generate-prompt`

Generates extraction prompts for a given attribute. If an exact attribute match is found in the database, it performs an instant bypass to return the matched structure, appending formatting rules on the fly. Otherwise, it retrieves the top 3 similar semantic attributes and leverages the LLM for Few-Shot prompting.

**Request Payload (`application/json`):**

```json
{
  "attribute_name": "Screen Resolution",
  "description": "Contextual description of the attribute to guide the model (optional)",
  "has_fixed_values": true, // weather the attribute has fixed values
  "has_failed": false // weather the prompt has failed
}
```

- `has_fixed_values`: Appends a directive forcing the output to select from `{allowed_values}`.
- `has_failed`: If `true`, the standard exact-match bypass is disabled. The model is forced to analyze the failing prompt and generate an improved iteration.

**Response:**
Returns a JSON array of length 2, representing the prompt logic targeting both image context (`user_input='all_images'`) and text-only context (`user_input='None'`).

```json
[
  {
    "prompt": "What is the screen_resolution... Select only from the following allowed values...",
    "system_role": "You are a product specification expert...",
    "user_input": "all_images"
  },
  {
    "prompt": "What is the screen_resolution... Select only from the following allowed values...",
    "system_role": "You are a product specification expert...",
    "user_input": "None"
  }
]
```

## Project Structure

- `src/api.py`: FastAPI server layer and routing.
- `src/generator.py`: LLM orchestration wrapping `openai.AsyncAzureOpenAI`.
- `src/vector_store.py`: ChromaDB storage wrapper for semantic embeddings (`all-MiniLM-L6-v2`).
- `src/data_loader.py`: CSV ingestion and pre-processing.
- `src/models.py`: Pydantic object models enforcing strict schemas.
- `src/logger.py`: Configures `structlog` for uniform JSON console logging.
