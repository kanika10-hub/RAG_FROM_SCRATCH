#USING LANGCHAIN AGENTS WITH GOOGLE GEMINI-2.5-FLASH MODEL
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
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
chat_history = []

tools = [calculator, uppercase, word_count]

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

agent = create_agent(
    model=chat_model,
    tools=tools
)

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    chat_history.append(("user", query))

    response = agent.invoke(
        {
            "messages": chat_history
        }
    )

    answer = response["messages"][-1].content

    print("Agent:", answer)

    chat_history.append(("assistant", answer))