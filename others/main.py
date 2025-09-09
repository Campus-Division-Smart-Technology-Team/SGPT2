from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
response = client.responses.create(
    model="gpt-4o-mini",
    input="Write a short bedtime story about a unicorn."
)

print(response.output_text)
