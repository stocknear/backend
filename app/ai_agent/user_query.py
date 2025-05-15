import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
vector_store_id = OpenAI(api_key=os.getenv("VECTOR_STORE_ID"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o-mini"

# Retrieve your existing vector store
vector_store = client.vector_stores.retrieve(vector_store_id=vector_store_id)

# Make the chat-completion call with the correct parameter name
response = client.chat.completions.create(
    model=CHAT_MODEL,
    messages=[
        {
            "role": "system",
            "content": (
                "You are an expert financial analyst. "
                "When asked about historical stock prices, you should query the vector store "
                "for the relevant embedding and return the exact price. "
                "Do not include any source citations, references, or metadata markers like [1],  , or similar in your response. "
                "Your answer should be clean and natural, as if speaking to a user."
            ),
        },
        {
            "role": "user",
            "content": "What was the price of GME on January 27, 2021?",
        },
    ],
    tools=[{"type": "file_search"}],
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},  # ← fixed
)

print(response.choices[0].message.content)
