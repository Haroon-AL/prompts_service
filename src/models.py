from pydantic import BaseModel, Field
from typing import Optional


class PromptGenerationRequest(BaseModel):
    """Payload to request the generation of an extraction prompt."""

    attribute_name: str = Field(
        ...,
        description="The name of the attribute to generate a prompt for (e.g. 'heart_notes', 'Screen Resolution')",
        examples=["heart_notes"],
    )
    description: Optional[str] = Field(
        None,
        description="Optional human-readable description providing context about the attribute",
        examples=[
            "Context: Heart Notes refer to the scent of a perfume that emerges just prior to when the top notes dissipate."
        ],
    )
    has_fixed_values: Optional[bool] = Field(
        False,
        description="A flag indicating if the attribute contains a predefined set of allowed values, used to restrict the LLM's classification surface.",
        examples=[True],
    )
    has_failed: Optional[bool] = Field(
        default=False,
        description="A flag indicating if the existing prompt for this attribute has failed previously, prompting the system to generate an improved version instead of using the exact match.",
        examples=[True],
    )


class SimilarAttribute(BaseModel):
    """Information regarding a similar attribute pulled from the underlying ChromaDB vector store."""

    attribute_name: str = Field(description="The similar attribute's original name")
    prompt: str = Field(description="The prompt logic stored for the similar attribute")
    system_role: str = Field(
        description="The system role/persona assigned to the similar attribute"
    )
    distance: float = Field(
        description="The cosine distance/similarity score (0.0 is exact match)"
    )


class GeneratedPrompt(BaseModel):
    """The structured response representing the ready-to-use LLM parameters based on the request."""

    prompt: str = Field(
        description="The generated prompt logic with appended strict JSON JSON schemas and formatting rules."
    )
    system_role: str = Field(
        description="The generated system persona for the extraction model behavior."
    )
    user_input: Optional[str] = Field(
        default=None,
        description="Optional value controlling how the targeted system handles input variations (e.g. 'all_images').",
        examples=["all_images"],
    )
