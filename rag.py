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


def evaluate_categories_with_rag(category_texts: dict):
    """
    Evaluate multiple category texts against the RAG DB in a SINGLE LLM call.

    Args:
        category_texts: dict mapping category name to combined transcript text,
                        e.g. {"warmup": "hello...", "praise": "great...", "direct": "you need to..."}

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
For the "direct" category, also evaluate whether the directions given are clear and practical.
Also provide brief suggestions on how the user can improve for each category.

Knowledge Base Context:
{context}

User's Spoken Text (by category):
{question}

Return ONLY a valid JSON object (no markdown) with one key per category. Example:
{{
  "warmup": {{"similarity_score_out_of_10": 8.5, "suggestions": "Your suggestion."}},
  "praise": {{"similarity_score_out_of_10": 7.0, "suggestions": "Your suggestion."}},
  "direct": {{"similarity_score_out_of_10": 6.0, "suggestions": "Your suggestion."}}
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


def generate_report(evidence_data: list, score_data: dict):
    """
    Generate a detailed report from evidence.json data in a SINGLE LLM call.

    Args:
        evidence_data: list of evidence dicts (from evidence.json)
        score_data: dict of all scores (from score.json)

    Returns:
        dict with the full report structure, or a fallback on error.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[WARNING] GOOGLE_API_KEY not set, returning fallback report.")
        return _fallback_report(evidence_data, score_data)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_retries=2,
    )

    evidence_str = json.dumps(evidence_data, indent=2)
    score_str = json.dumps(score_data, indent=2)

    prompt = ChatPromptTemplate.from_template(
        """You are a professional meeting coach. Based on the scoring evidence below, generate a detailed performance report.

Scores:
{scores}

Evidence:
{evidence}

Return ONLY a valid JSON object (no markdown) with this exact structure:
{{
  "total_score": <number out of 100>,
  "categories": [
    {{
      "name": "template",
      "score": <number>,
      "max_score": 10,
      "description": "Brief description of performance in this category."
    }},
    {{
      "name": "warmup",
      "score": <number>,
      "max_score": 15,
      "description": "Brief description..."
    }},
    {{
      "name": "praise",
      "score": <number>,
      "max_score": 20,
      "description": "Brief description..."
    }},
    {{
      "name": "suggest",
      "score": <number>,
      "max_score": 20,
      "description": "Brief description..."
    }},
    {{
      "name": "listen",
      "score": <number>,
      "max_score": 15,
      "description": "Brief description..."
    }},
    {{
      "name": "direct",
      "score": <number>,
      "max_score": 20,
      "description": "Brief description..."
    }}
  ],
  "strengths": ["strength 1", "strength 2"],
  "improvements": ["improvement 1 with specific suggestion", "improvement 2 with specific suggestion"]
}}"""
    )

    try:
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"scores": score_str, "evidence": evidence_str})

        response = response.strip()
        if response.startswith("```json"):
            response = response[7:-3].strip()
        elif response.startswith("```"):
            response = response[3:-3].strip()

        return json.loads(response)
    except Exception as e:
        print(f"[WARNING] Report generation failed: {e}")
        return _fallback_report(evidence_data, score_data)


def _fallback_report(evidence_data: list, score_data: dict):
    """Build a basic report without LLM when the API is unavailable."""
    total = score_data.get("total_score", 0)
    categories = []
    cat_max = {
        "template": 10, "warmup": 15, "praise": 20,
        "suggest": 20, "listen": 15, "direct": 20
    }
    for ev in evidence_data:
        cat = ev.get("category", "unknown")
        categories.append({
            "name": cat,
            "score": ev.get("score", 0),
            "max_score": cat_max.get(cat, 0),
            "description": ev.get("evidence", "")
        })

    return {
        "total_score": total,
        "categories": categories,
        "strengths": ["Report generation unavailable — review evidence.json for details."],
        "improvements": ["Report generation unavailable — review evidence.json for details."]
    }
