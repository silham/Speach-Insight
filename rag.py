import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# RAG DB Directory
CHROMA_PATH = "chroma_db"


def get_embeddings():
    """Return local, free embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_vectorstore():
    """Initialise or load the vector store."""
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())


def add_document_to_db(filepath: str):
    """Load a document, split it, and add it to the Chroma vector store."""
    if filepath.endswith('.pdf'):
        loader = PyPDFLoader(filepath)
    elif filepath.endswith('.txt'):
        loader = TextLoader(filepath)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    return len(chunks)


def _format_docs(docs):
    """Concatenate retrieved document contents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def ask_rag(query: str):
    """Query the vector database and generate an answer using Gemini (free tier)."""

    # We expect GOOGLE_API_KEY to be set in the environment or .env
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in environment variables.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_retries=2,
    )

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Modern LCEL-based RAG chain (replaces deprecated RetrievalQA)
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question based ONLY on the following context.
If the context does not contain enough information, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
    )

    # Build the chain using LCEL
    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Also retrieve raw docs for source display
    retrieved_docs = retriever.invoke(query)
    answer = rag_chain.invoke(query)

    return {
        "answer": answer,
        "context": [doc.page_content for doc in retrieved_docs]
    }
