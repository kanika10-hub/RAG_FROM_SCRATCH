from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#for langchain rag chain 
load_dotenv()

hug_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = InferenceClient(
    token=hug_token,
    model="meta-llama/Llama-3.1-8B-Instruct"
)
loader = PyPDFLoader("document.pdf")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
#print("Total Chunks:", len(chunks))
#for i, chunk in enumerate(chunks):
   # print(f"\nChunk {i+1}")
  #  print(chunk.page_content)



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)






query = input("Ask a question: ")

results = retriever.invoke(query)

context = "\n\n".join(
    [doc.page_content for doc in results]
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer using the provided context.

    Context:
    {context}

    Question:
    {question}

    If the context contains relevant information, use it to answer.
    Only say "I don't know" if the context contains no information related to the question.
    
    Answer:
    """
)

formatted = prompt.invoke(
    {
        "context": context,
        "question": query
    }
)

print(formatted)

#for doc in results:
  #  print("-" * 50)
  #  print(doc.page_content)

#for doc in results:
   # print(doc.metadata)

#print("\nPROMPT SENT TO LLM")
#print("=" * 50)
#print(formatted.messages[0].content)

response = llm.chat_completion(
    messages=[
        {
            "role": "user",
            "content": formatted.messages[0].content
        }
    ],
    max_tokens=300
)
print("\n" + "=" * 50)
print("ANSWER")
print("=" * 50)

print(response.choices[0].message.content)