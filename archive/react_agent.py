#USING LANGCHAIN AGENTS WITH GOOGLE GEMINI-2.5-FLASH MODEL
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()

@tool
def calculator(expression: str):
    """Calculate a mathematical expression."""
    return str(eval(expression))

@tool
def uppercase(text: str):
    """Convert text to uppercase."""
    return text.upper()

@tool
def word_count(text: str):
    """Count words in text."""
    return str(len(text.split()))

#for chat history
#chat_history = []
memory = MemorySaver()

tools = [calculator, uppercase, word_count]

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

agent = create_agent(
    model=chat_model,
    tools=tools,
    checkpointer=memory
)

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    

    response = agent.invoke(
    {
        "messages": [
            {
                "role": "system",
                "content": """
You are a helpful AI assistant.
Use tools only when necessary.
Answer normally for general questions.
Use memory to remember previous conversations.
"""
            },
            {
                "role": "user",
                "content": query
            }
        ]
    },
    config={
        "configurable": {
            "thread_id": "user1"
        }
    }
)

    answer = response["messages"][-1].content

    print("Agent:", answer)

    





