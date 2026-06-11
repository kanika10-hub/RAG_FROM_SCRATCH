from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    """
    Question:
    {question}
    """
)

parser = StrOutputParser()

chain = prompt | parser

result = chain.invoke(
    {
        "question": "What is Python?"
    }
)

print(result)


#we are doing PROMPT->PARSER->OUTPUT 
#but the parser expects: PROMPT->LLM->PARSER->OUTPUT
#so we need to add a dummy LLM in between
#but my current LLM is HUGGINGFACE INFERENCE API which does not support streaming output, so we cannot use it in the chain.
#WHICH MEANS IT is a Hugging Face client, not a LangChain Runnable.
#LCEL's | operator only works with LangChain-compatible components.
#ITS ALWAYS RETRIEVER|PROMPT|LLM|PARSER 