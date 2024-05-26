import os

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Access the OPENAI_API_KEY environment variable

api_key = os.environ.get('OPENAI_API_KEY')

# Check if the API key is set
"""
if api_key is None:
    print("API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit()
"""

# Create the OpenAI client
# client = OpenAI(api_key=api_key)
client = OpenAI(api_key=api_key)


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(model: str, messages, tools=None, tool_choice=None):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
