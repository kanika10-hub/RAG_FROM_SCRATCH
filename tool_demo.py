from langchain_core.tools import tool

@tool
def calculator(expression: str):
    """Calculate a mathematical expression."""
    return eval(expression)

@tool
def uppercase(text: str):
    """Convert text to uppercase."""
    return text.upper()

print(calculator.invoke("25 * 14"))

print(uppercase.invoke("hello world"))

#WITH @ , LANGCHAIN KNOWS THIS IS A TOOL, SO IT CAN BE CALLED INSIDE A CHAIN OR CALLED DIRECTLY. WITHOUT @, IT'S JUST A NORMAL FUNCTION.
#LANGCHAIN CONVERTS IT INTO SOMETHING AN AGENT CAN UNDERSTAND .

'''simulating an LLM decision'''

question = input("Ask something: ")

if "*" in question or "+" in question:
    result = calculator.invoke(question)
else:
    result = uppercase.invoke(question)

print(result)

##using "invoke" to call the tool.


