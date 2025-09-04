import json
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from langfuse.api.resources.score_configs.types.create_score_config_request import (
    CreateScoreConfigRequest,
)
from langfuse.openai import openai
from pydantic import BaseModel
from typing import Literal

"""
Langfuse Scoring Tutorial - Human Annotation & Boolean Classification

Prerequisites:
- Set up your .env file with LanLangfusegFuse credentials:
  LANGFUSE_SECRET_KEY=your-secret-key
  LANGFUSE_PUBLIC_KEY=your-public-key  
  LANGFUSE_HOST=your-host-url
  
This example is for Langfuse SDK v2 (not v3).
"""

load_dotenv()
langfuse = Langfuse()


class CustomerInquiry(BaseModel):
    category: Literal["complaint", "feature_request", "billing", "other"]
    response: str


def load_event(filename: str) -> dict:
    """Load test event from JSON file"""
    with open(f"events/{filename}", "r") as f:
        return json.load(f)


@observe()
def process_customer_message(message: str) -> CustomerInquiry:
    """Process customer message with LangFuse tracing"""
    inquiry = openai.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a customer service AI that analyzes customer inquiries.",
            },
            {"role": "user", "content": message},
        ],
        response_format=CustomerInquiry,
        temperature=0,
    )

    result = inquiry.choices[0].message.parsed

    # Add metadata to current observation
    langfuse_context.update_current_observation(
        metadata={
            "category": result.category,
            "response_length": len(result.response),
            "customer_message_length": len(message),
        }
    )

    return result


# --------------------------------------------------------------
# Create a score config
# --------------------------------------------------------------


def create_score_config_example():
    score_config_data = CreateScoreConfigRequest(
        name="Pass",
        data_type="BOOLEAN",
        description="Binary quality assessment: 1 for good quality, 0 for poor quality",
    )
    config = langfuse.api.score_configs.create(request=score_config_data)

    return config


def run_evaluation_pipeline():
    test_files = ["billing_test.json", "feature_request_test.json", "failing_test.json"]

    for i, filename in enumerate(test_files, 1):
        event = load_event(filename)
        message = event["message"]
        result = process_customer_message(message)

        print(result.response)


if __name__ == "__main__":
    # create_score_config_example()
    run_evaluation_pipeline()


"""
Run the evaluation pipeline

1. Create a score config (only once)
2. Run the evaluation pipeline
3. Add scores to LangFuse (using the UI)
4. View scores in the LangFuse dashboard
"""
