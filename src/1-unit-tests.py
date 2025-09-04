import os
import json
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_event(filename: str) -> dict:
    with open(f"events/{filename}", "r") as f:
        return json.load(f)


# --------------------------------------------------------------
# Define the example workflow
# --------------------------------------------------------------


class CustomerInquiry(BaseModel):
    category: Literal["complaint", "feature_request", "billing", "other"]
    response: str


def process_customer_message(message: str) -> CustomerInquiry:
    inquiry = client.responses.parse(
        model="gpt-4o-mini",
        text_format=CustomerInquiry,
        input=[
            {
                "role": "system",
                "content": "You are a customer service AI that analyzes customer inquiries.",
            },
            {"role": "user", "content": message},
        ],
        temperature=0,
    )
    return inquiry.output_parsed


# --------------------------------------------------------------
# Individual test functions
# --------------------------------------------------------------


def test_billing_categorization():
    event = load_event("billing_test.json")
    result = process_customer_message(event["message"])
    assert result.category == "billing"
    assert len(result.response) > 10


def test_feature_request_categorization():
    event = load_event("feature_request_test.json")
    result = process_customer_message(event["message"])
    assert result.category == "feature_request"
    assert len(result.response) > 10


def test_support_categorization():
    event = load_event("failing_test.json")
    result = process_customer_message(event["message"])
    assert result.category == "complaint"  # This should fail - actual is "other"
    assert len(result.response) > 5


# --------------------------------------------------------------
# Run all tests
# --------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_billing_categorization,
        test_feature_request_categorization,
        test_support_categorization,
    ]
    passed = 0

    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__}")

    print(f"\nResults: {passed}/{len(tests)} tests passed")
