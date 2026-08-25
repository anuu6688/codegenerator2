import os
from fastapi import FastAPI
from langserve import add_routes

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
INDEX_DIR = "/content/faiss_index"

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

# Load the FAISS index built earlier in the notebook
vectorstore = FAISS.load_local(
    INDEX_DIR,
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

RAG_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below. \
If the answer is not contained in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:"""
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

app = FastAPI(
    title="Gemini RAG API",
    version="1.0"
)

# Plain chat, no retrieval
add_routes(
    app,
    llm,
    path="/crew"
)

# Retrieval-augmented chain
add_routes(
    app,
    rag_chain,
    path="/rag"
)

@app.get("/")
def home():
    return {
        "message": "Gemini RAG API is running successfully!",
        "endpoints": ["/crew/playground/", "/rag/playground/"]
    }
    