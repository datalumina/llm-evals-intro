from dotenv import load_dotenv
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import openai
from pydantic import BaseModel, Field

"""
First, go to https://langfuse.com and create a free account to get your Langfuse credentials.

Make sure you have set the environment variables in the .env file:
LANGFUSE_SECRET_KEY=your-langfuse-api-secret-here
LANGFUSE_PUBLIC_KEY=your-langfuse-api-key-here
LANGFUSE_HOST=your-langfuse-host-here

This example is for Langfuse SDK v2 (not v3).
"""

load_dotenv()


# --------------------------------------------------------------
# Example 1: Minimal LangFuse Setup
# --------------------------------------------------------------


@observe()
def simple_story_generator(topic: str) -> str:
    response = openai.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": f"Write a short story about: {topic}"},
        ],
    )

    story = response.output_text

    return story


def run_simple_example():
    story = simple_story_generator("a robot learning to cook")
    print(f"Generated story: {story[:100]}...")
    print("Check your LangFuse dashboard to see the tracked call!")


# --------------------------------------------------------------
# Example 2: Simplified Business Use Case
# --------------------------------------------------------------


class CustomerQuery(BaseModel):
    """Simple customer query classification."""

    category: str = Field(description="Query category: billing, technical, or general")
    urgency: str = Field(description="Urgency: low, medium, high")
    summary: str = Field(description="Brief summary of the issue")


@observe()
def analyze_customer_query(query: str) -> CustomerQuery:
    response = openai.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": """Analyze customer queries and classify them.
                
                Categories:
                - billing: payment, subscription, refund issues
                - technical: bugs, errors, setup problems  
                - general: questions, feature requests
                
                Urgency levels:
                - low: general inquiries
                - medium: affecting service usage
                - high: business-critical issues""",
            },
            {"role": "user", "content": f"Customer query: {query}"},
        ],
        response_format=CustomerQuery,
        temperature=0.1,
    )

    analysis = response.choices[0].message.parsed

    # Add metadata to LangFuse
    langfuse_context.update_current_observation(
        name="customer_query_analysis",
        metadata={
            "category": analysis.category,
            "urgency": analysis.urgency,
            "query_length": len(query),
        },
        input={"query": query},
        output=analysis.model_dump(),
    )

    return analysis


@observe()
def generate_response(query: str, analysis: CustomerQuery) -> str:
    response = openai.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": f"""You are a helpful customer service representative.
                
                Customer context:
                - Issue category: {analysis.category}
                - Urgency level: {analysis.urgency}
                - Issue summary: {analysis.summary}
                
                Generate a professional, helpful response.""",
            },
            {
                "role": "user",
                "content": f"Original query: {query}\n\nGenerate a response:",
            },
        ],
    )

    service_response = response.output_text

    # Add metadata to LangFuse
    langfuse_context.update_current_observation(
        name="response_generation",
        metadata={
            "category": analysis.category,
            "urgency": analysis.urgency,
            "response_length": len(service_response),
        },
        input={"query": query, "analysis": analysis.model_dump()},
        output={"response": service_response},
    )

    return service_response


@observe()
def customer_support_pipeline(query: str):
    # Step 1: Analyze the query
    analysis = analyze_customer_query(query)

    # Step 2: Generate response
    response = generate_response(query, analysis)

    # Update main observation with overall metrics
    langfuse_context.update_current_observation(
        name="customer_support_pipeline",
        metadata={
            "total_steps": 2,
            "final_category": analysis.category,
            "final_urgency": analysis.urgency,
        },
        input={"customer_query": query},
        output={"analysis": analysis.model_dump(), "response": response},
        tags=["customer_support", "pipeline", "business_example"],
    )

    return analysis, response


def run_business_example():
    # Sample customer query
    query = """Hi, I'm having issues with my Premium subscription. 
    I was charged twice this month and need a refund. 
    This is urgent as it's affecting my budget."""

    analysis, response = customer_support_pipeline(query)

    print("Analysis:")
    print(f"- Category: {analysis.category}")
    print(f"- Urgency: {analysis.urgency}")
    print(f"- Summary: {analysis.summary}")

    print("\nGenerated Response:\n")
    print(f"{response[:150]}...")

    print("\nCheck your LangFuse dashboard for detailed tracking!")


# --------------------------------------------------------------
# Main execution
# --------------------------------------------------------------

if __name__ == "__main__":
    # Run minimal example first
    run_simple_example()
    # Run business example
    run_business_example()
