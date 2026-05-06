import os
import json
from dotenv import load_dotenv
load_dotenv()
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

def evaluate_warmup_with_rag(warmup_text: str):
    """Evaluate warmup text against RAG DB and return JSON with score and suggestions."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[WARNING] GOOGLE_API_KEY not set, returning fallback RAG score.")
        return {"similarity_score_out_of_10": 0.0, "suggestions": "RAG evaluation failed: missing API key."}

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        max_retries=2,
    )

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """You are an evaluator. Compare the user's warmup text against the following knowledge base context.
Calculate a similarity score out of 10 based on how well the user followed the recommended warmup guidelines.
Also provide brief suggestions on how the user can improve based on the context.

Context:
{context}

Warmup Text: {question}

Return ONLY a valid JSON object matching this schema, without markdown formatting:
{{
  "similarity_score_out_of_10": 8.5,
  "suggestions": "Your suggestion here."
}}"""
    )

    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        response = rag_chain.invoke(warmup_text)
        # Clean potential markdown block formatting
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:-3].strip()
        elif response.startswith("```"):
            response = response[3:-3].strip()
        
        result = json.loads(response)
        return {
            "similarity_score_out_of_10": float(result.get("similarity_score_out_of_10", 0.0)),
            "suggestions": result.get("suggestions", "")
        }
    except Exception as e:
        print(f"[WARNING] RAG evaluation failed: {e}")
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Quota" in err_msg:
            return {"similarity_score_out_of_10": 0.0, "suggestions": "API quota exceeded (please wait a minute and try again)."}
        return {"similarity_score_out_of_10": 0.0, "suggestions": "Evaluation failed due to an error."}
