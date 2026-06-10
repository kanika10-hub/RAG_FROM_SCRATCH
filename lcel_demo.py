from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """
    Question:
    {question}
    """
)

formatted = prompt.invoke(
    {
        "question": "What is Python?"
    }
)

print(formatted)
print(formatted.messages[0].content)