import pytest
from httpx import AsyncClient, ASGITransport
import os

# Set dummy Azure API key and endpoint for tests if not set
if "AZURE_OPENAI_API_KEY" not in os.environ:
    os.environ["AZURE_OPENAI_API_KEY"] = "test-dummy-key"
if "AZURE_OPENAI_ENDPOINT" not in os.environ:
    os.environ["AZURE_OPENAI_ENDPOINT"] = (
        "https://test-dummy-endpoint.openai.azure.com/"
    )

from src.api import app


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "db_count" in data


@pytest.mark.asyncio
async def test_generate_prompt_missing_field():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Missing attribute_name
        response = await ac.post("/generate-prompt", json={"description": "test"})

    assert (
        response.status_code == 422
    )  # Unprocessable Entity due to Pydantic validation
