from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    provider="hf-inference"   # 🔥 IMPORTANT FIX
)

response = client.chat_completion(
    messages=[
        {"role": "user", "content": "Hello"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)