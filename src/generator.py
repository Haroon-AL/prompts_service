import os
from typing import List, Optional
from openai import AsyncAzureOpenAI
from src.models import SimilarAttribute
from src.logger import setup_logging

logger = setup_logging()


class PromptGenerator:
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        if not self.api_key or not self.endpoint:
            logger.warning(
                "AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT environment variable not set. LLM generation may fail if not provided in environment."
            )
        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )

    async def generate_prompt(
        self,
        attribute_name: str,
        description: str,
        examples: List[SimilarAttribute],
        existing_failed_prompt: Optional[str] = None,
    ) -> dict:
        """
        Generates a prompt for a new attribute using few-shot examples.
        Returns a dictionary with 'prompt' and 'system_role'.
        """
        system_prompt = (
            "You are an expert prompt engineer for an e-commerce platform. "
            "Your task is to write high-quality extraction prompts for product attributes.\n"
            "You will be given the name of the attribute, an optional description, and some examples of how prompts were written for similar attributes.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            '1. Output exactly a JSON object with two string keys: "prompt" and "system_role". Do not wrap in markdown blocks like ```json.\n'
            "2. Note: We will manually append the final return schema to your prompt. You focus merely on writing the conceptual extraction logic, providing steps and criteria for identifying the attribute.\n"
        )

        if existing_failed_prompt:
            system_prompt += (
                "3. The user has indicated that the previous prompt for this attribute FAILED to extract correctly. "
                "You must heavily analyze the provided examples and write an IMPROVED, more robust version of the "
                "following failed prompt. Do NOT just copy it.\n"
                f"--- FAILED PROMPT ---\n{existing_failed_prompt}\n----------------------\n"
            )

        user_prompt = f"Attribute Name: {attribute_name}\n"
        if description:
            user_prompt += f"Description: {description}\n"

        user_prompt += "\nExamples of prompts for similar attributes:\n"
        for i, ex in enumerate(examples, 1):
            user_prompt += f"\n--- Example {i} ---\nAttribute: {ex.attribute_name}\nSystem Role: {ex.system_role}\nPrompt Setup:\n{ex.prompt}\n"

        user_prompt += "\nNow, based on the patterns in the examples, generate the extraction prompt and system role for the new attribute in JSON format."

        try:
            response = await self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            import json

            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error generating prompt with LLM: {e}")
            raise
