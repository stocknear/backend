import os
from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vector_store_id = os.getenv("VECTOR_STORE_ID")

CHAT_MODEL = "gpt-4o-mini"


assistant = client.beta.assistants.create(
    name="Financial Analyst Assistant",
    model=CHAT_MODEL,
    instructions=(
    "You are an expert financial analyst."
    "When asked about historical stock prices, you should query the vector store "
    "for the relevant embedding and return the exact price. "
    "Do not include any source citations, references, or metadata markers like [1],  , or similar in your response."
    "Your answer should be clean and natural, as if speaking to a user."
    ),
    tools=[{"type": "file_search"}],
    tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
)



thread = client.beta.threads.create()


client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="what are the short data of the peers of gme",)


run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

while True:
    run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    if run_status.status in ["completed", "failed"]:
        break
    time.sleep(1)

# 10. Retrieve and print the assistant's answer
messages = client.beta.threads.messages.list(thread_id=thread.id)
for msg in messages.data[::-1]:
    print(msg.role + ":", msg.content[0].text.value)
