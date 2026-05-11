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

def evaluate_categories_with_rag(category_texts: dict):
    """
    Evaluate multiple category texts against the RAG DB in a SINGLE LLM call.

    Args:
        category_texts: dict mapping category name to combined transcript text,
                        e.g. {"warmup": "hello everyone...", "praise": "great work..."}

    Returns:
        dict mapping category name to {"similarity_score_out_of_10": float, "suggestions": str}
    """
    # Build fallback results
    fallback = {cat: {"similarity_score_out_of_10": 0.0, "suggestions": f"No {cat} segments detected."}
                for cat in category_texts}

    # Filter out empty categories
    active_cats = {cat: text for cat, text in category_texts.items() if text.strip()}
    if not active_cats:
        return fallback

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[WARNING] GOOGLE_API_KEY not set, returning fallback RAG scores.")
        for cat in active_cats:
            fallback[cat]["suggestions"] = "RAG evaluation failed: missing API key."
        return fallback

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        max_retries=2,
    )

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Build the category block for the prompt
    cat_block = "\n".join(
        f"--- {cat.upper()} ---\n{text}" for cat, text in active_cats.items()
    )

    prompt = ChatPromptTemplate.from_template(
        """You are an evaluator. Compare the user's spoken text for each category against the knowledge base context.
For EACH category, calculate a similarity score out of 10 based on how well the user followed the recommended guidelines.
Also provide brief suggestions on how the user can improve for each category.

Knowledge Base Context:
{context}

User's Spoken Text (by category):
{question}

Return ONLY a valid JSON object (no markdown) with one key per category. Example:
{{
  "warmup": {{"similarity_score_out_of_10": 8.5, "suggestions": "Your suggestion."}},
  "praise": {{"similarity_score_out_of_10": 7.0, "suggestions": "Your suggestion."}}
}}"""
    )

    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        response = rag_chain.invoke(cat_block)
        # Clean potential markdown formatting
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:-3].strip()
        elif response.startswith("```"):
            response = response[3:-3].strip()

        result = json.loads(response)

        for cat in active_cats:
            cat_result = result.get(cat, {})
            fallback[cat] = {
                "similarity_score_out_of_10": float(cat_result.get("similarity_score_out_of_10", 0.0)),
                "suggestions": cat_result.get("suggestions", "")
            }

        return fallback
    except Exception as e:
        print(f"[WARNING] RAG evaluation failed: {e}")
        err_msg = str(e)
        suggestion = "Evaluation failed due to an error."
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Quota" in err_msg:
            suggestion = "API quota exceeded (please wait a minute and try again)."
        for cat in active_cats:
            fallback[cat]["suggestions"] = suggestion
        return fallback

