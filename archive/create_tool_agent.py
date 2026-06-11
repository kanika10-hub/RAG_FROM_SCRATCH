from langchain_core.tools import tool
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

# -------------------
# GEMINI SETUP
# -------------------

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# -------------------
# TOOLS
# -------------------

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

tools = {
    "calculator": calculator,
    "uppercase": uppercase,
    "word_count": word_count
}

# -------------------
# USER QUESTION
# -------------------

question = input("Ask something: ")

tool_prompt = f"""
You are a helpful assistant .
answer normall using your own knowledge and reasoning.
only use following tools when they are actually needed.

Available tools:

1. calculator
   Use for mathematical calculations.

2. uppercase
   Use for converting text to uppercase.

3. word_count
   Use for counting words.

User Question:
{question}

Return ONLY in this format:

TOOL: tool_name
INPUT: tool_input
"""

# -------------------
# TOOL SELECTION
# -------------------

response = llm.generate_content(tool_prompt)

decision = response.text.strip()

print("\nTOOL DECISION:")
print(decision)

# -------------------
# PARSE DECISION
# -------------------

lines = decision.split("\n")

tool_name = lines[0].replace("TOOL:", "").strip()
tool_input = lines[1].replace("INPUT:", "").strip()

# -------------------
# EXECUTE TOOL
# -------------------

result = tools[tool_name].invoke(tool_input)

print("\nTOOL RESULT:")
print(result)

# -------------------
# FINAL ANSWER
# -------------------

final_prompt = f"""
User Question:
{question}

Tool Used:
{tool_name}

Tool Result:
{result}

Answer the user's question using the tool result.
"""

final_response = llm.generate_content(
    final_prompt
)

print("\n" + "=" * 50)
print("FINAL ANSWER")
print("=" * 50)

print(final_response.text)