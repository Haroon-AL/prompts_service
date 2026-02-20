import os
import json
from typing import List, Optional
import adalflow
from adalflow.components.model_client import AzureAIClient
from src.models import SimilarAttribute
from src.logger import setup_logging
from pathlib import Path

logger = setup_logging()

def load_prompt_template() -> str:
    template_path = Path(__file__).parent / "prompts" / "attribute_extraction.jinja"
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


class PromptGenerator(adalflow.Component):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        os.environ["AZURE_OPENAI_API_KEY"] = self.api_key

        self.client = AzureAIClient(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        )

        self.llm = adalflow.Generator(
            model_client=self.client,
            model_kwargs={
                "model": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                "temperature": 0.2,
                "max_tokens": 1500,
                "response_format": {"type": "json_object"},
            },
        )
        self.prompt_template = adalflow.Prompt(template=load_prompt_template())

    async def generate_prompt(
        self,
        attribute_name: str,
        description: str,
        examples: List[SimilarAttribute],
        existing_failed_prompt: Optional[str] = None,
    ) -> dict:
        """
        Generates a prompt for a new attribute using few-shot examples via AdalFlow pipeline.
        Returns a dictionary with 'prompt' and 'system_role'.
        """
        prompt_kwargs = {
            "attribute_name": attribute_name,
            "description": description,
            "examples": examples,
            "existing_failed_prompt": existing_failed_prompt,
        }
        prompt_text = self.prompt_template(**prompt_kwargs)

        try:
            response = await self.llm.acall(prompt_kwargs={"input_str": prompt_text})
            content = response.data

            return json.loads(content)

        except Exception as e:
            logger.error(f"Error calling AdalFlow/AzureOpenAI: {e}")
            raise
