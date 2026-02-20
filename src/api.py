from fastapi import FastAPI, HTTPException
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from typing import List, Optional
from src.models import PromptGenerationRequest, GeneratedPrompt, SimilarAttribute
from src.vector_store import VectorStore
from src.data_loader import load_and_index_data
from src.generator import PromptGenerator
from src.logger import setup_logging

logger = setup_logging()

load_dotenv()

vector_store = None
generator = None


def append_formatting_rules(
    prompt: str, attribute_name: str, has_fixed_values: Optional[bool]
) -> str:
    lines = [prompt]
    if has_fixed_values:
        lines.append("Select only from the following allowed values.")
        lines.append('"allowed_values": {allowed_values}')

    lines.append(
        '- return the output in {language} language in JSON format {{ "'
        + attribute_name
        + '" : <your_classification> }}'
    )

    lines.append("- Strictly return the JSON object only.")
    lines.append(
        "- Do not include markdown formatting, code blocks, escaped characters or explanations."
    )

    return "\n".join(lines)


def initialize_data():
    """Loads CSV data into ChromaDB on startup if DB is empty."""
    global vector_store
    if vector_store.count() == 0:
        logger.info("Vector DB is empty. Loading data from CSVs...")
        base_dir = os.path.dirname(os.path.dirname(__file__))

        files = os.listdir(base_dir)
        attr_csv = next(
            (f for f in files if f.startswith("public_attributes_definition_export")),
            None,
        )
        mapper_csv = next(
            (f for f in files if f.startswith("public_llm_mapper_export")), None
        )
        prompt_csv = next(
            (
                f
                for f in files
                if f.startswith("public_llm_prompt_configuration_export")
            ),
            None,
        )

        if attr_csv and mapper_csv and prompt_csv:
            load_and_index_data(
                attributes_csv=os.path.join(base_dir, attr_csv),
                mapper_csv=os.path.join(base_dir, mapper_csv),
                prompt_csv=os.path.join(base_dir, prompt_csv),
                vector_store=vector_store,
            )
        else:
            logger.warning(
                "CSV files not found in root directory. Skipping initial data load."
            )
    else:
        logger.info(f"Vector DB already contains {vector_store.count()} items.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, generator
    logger.info("Initializing application...")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    db_dir = os.path.join(base_dir, "data", "chroma_db")

    vector_store = VectorStore(persist_directory=db_dir)
    generator = PromptGenerator()

    initialize_data()

    yield

    logger.info("Shutting down application...")


app = FastAPI(title="Attribute Prompt Generator API", lifespan=lifespan)


@app.post(
    "/generate-prompt",
    response_model=List[GeneratedPrompt],
    summary="Generate Attribute Extraction Prompts",
    description="Generates specific extraction prompts for an attribute using an intelligent LLM alongside semantic similarities stored in a local ChromaDB.",
    responses={
        500: {
            "description": "Internal server error occurred when generating the prompt or fetching from the vector store."
        }
    },
)
async def generate_prompt(request: PromptGenerationRequest):
    logger.info(f"Received generation request for attribute: {request.attribute_name}")

    try:
        similar_docs, distances = vector_store.search(
            query=request.attribute_name, top_k=3
        )

        existing_failed_prompt = None

        if similar_docs and distances:
            first_doc = similar_docs[0]
            if first_doc["attribute_name"].lower() == request.attribute_name.lower():
                if request.has_failed:
                    logger.info(
                        f"Exact match found in DB for {request.attribute_name}, but 'has_failed' is true. "
                        "Passing to LLM for improvement."
                    )
                    existing_failed_prompt = first_doc["prompt"]
                else:
                    logger.info(
                        f"Exact match found in DB for {request.attribute_name}. Bypassing LLM generation."
                    )
                    final_prompt = append_formatting_rules(
                        prompt=first_doc["prompt"],
                        attribute_name=request.attribute_name,
                        has_fixed_values=request.has_fixed_values,
                    )
                    return [
                        GeneratedPrompt(
                            prompt=final_prompt,
                            system_role=first_doc["system_role"],
                            user_input="all_images",
                        ),
                        GeneratedPrompt(
                            prompt=final_prompt,
                            system_role=first_doc["system_role"],
                            user_input="None",
                        ),
                    ]

        similar_attributes = []
        for doc, dist in zip(similar_docs, distances):
            similar_attributes.append(
                SimilarAttribute(
                    attribute_name=doc["attribute_name"],
                    prompt=doc["prompt"],
                    system_role=doc["system_role"],
                    distance=dist,
                )
            )

        logger.info(f"Found {len(similar_attributes)} similar attributes.")

        generated_data = await generator.generate_prompt(
            attribute_name=request.attribute_name,
            description=request.description or "",
            examples=similar_attributes,
            existing_failed_prompt=existing_failed_prompt,
        )

        final_gen_prompt = append_formatting_rules(
            prompt=generated_data["prompt"],
            attribute_name=request.attribute_name,
            has_fixed_values=request.has_fixed_values,
        )

        return [
            GeneratedPrompt(
                prompt=final_gen_prompt,
                system_role=generated_data["system_role"],
                user_input="all_images",
            ),
            GeneratedPrompt(
                prompt=final_gen_prompt,
                system_role=generated_data["system_role"],
                user_input="None",
            ),
        ]

    except Exception as e:
        logger.error(f"Failed to generate prompt: {e}")
        # Raise HTTP 500 when Azure OpenAI fails or ChromaDB operations cause unexpected errors
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok", "db_count": vector_store.count() if vector_store else 0}
