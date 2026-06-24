import os
import json
import re
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# RAG DB Directory
CHROMA_PATH = "chroma_db"

# Valid RAG database category names
VALID_CATEGORIES = {"warmup", "praise", "suggest", "listen", "direct"}


def get_embeddings():
    """Return local, free embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_category_vectorstore(category: str):
    """Initialise or load the vector store for a specific category."""
    path = os.path.join(CHROMA_PATH, category.lower())
    return Chroma(persist_directory=path, embedding_function=get_embeddings())


def split_into_sentences(text: str) -> list[str]:
    """Split text into individual sentences using a robust regex pattern."""
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    sentences = []
    for line in lines:
        splits = sentence_endings.split(line)
        for s in splits:
            s = s.strip()
            if s:
                sentences.append(s)
    return sentences


def _classify_sentences_with_llm(sentences: list[str]) -> dict[str, list[str]]:
    """
    Classify a list of sentences into RAG categories using Gemini LLM.

    Returns a dict mapping category name to list of sentences, e.g.:
        {"warmup": ["Hello, how are you?"], "praise": ["Great job!"], ...}
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[WARNING] GOOGLE_API_KEY not set, cannot classify sentences.")
        return {cat: [] for cat in VALID_CATEGORIES}

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        max_retries=2,
    )

    # Build numbered sentence list for the prompt
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

    prompt = ChatPromptTemplate.from_template(
        """You are a text classifier for a coaching/appraisal conversation framework.
Classify whether each of the guidelines,directions,examples are for the warmup,praise,suggest,listen or direct:

- warmup: Guidelines, directions, or examples related to greetings, ice-breakers, casual conversation starters, or asking how someone is doing.
- praise: Guidelines, directions, or examples related to positive feedback, compliments, or recognition of good work or achievements.
- suggest: Guidelines, directions, or examples related to suggestions for improvement (both positive and constructive/negative), recommendations, or advice.
- listen: Guidelines, directions, or examples related to active listening cues, acknowledgements, paraphrasing what someone said, or empathetic responses.
- direct: Guidelines, directions, or examples related to direct instructions, action items, commands, or clear directives about what to do.

Guidelines/Directions/Examples:
{sentences}

Return ONLY a valid JSON object (no markdown, no explanation) mapping each category to a list of item numbers. Example:
{{
  "warmup": [1, 3],
  "praise": [2],
  "suggest": [4, 7],
  "listen": [5],
  "direct": [6]
}}

Every item number (1 to {count}) must appear in exactly one category. Do not skip any item."""
    )

    try:
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"sentences": numbered, "count": str(len(sentences))})

        # Clean potential markdown formatting
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:-3].strip()
        elif response.startswith("```"):
            response = response[3:-3].strip()

        raw_result = json.loads(response)

        # Map sentence numbers back to sentence text
        classified = {cat: [] for cat in VALID_CATEGORIES}
        for cat, indices in raw_result.items():
            cat_lower = cat.lower()
            if cat_lower in VALID_CATEGORIES and isinstance(indices, list):
                for idx in indices:
                    idx_int = int(idx) - 1  # Convert 1-based to 0-based
                    if 0 <= idx_int < len(sentences):
                        classified[cat_lower].append(sentences[idx_int])

        return classified

    except Exception as e:
        print(f"[WARNING] LLM sentence classification failed: {e}")
        return {cat: [] for cat in VALID_CATEGORIES}


def add_document_to_db(filepath: str):
    """
    Load a document (PDF, DOCX, or TXT), split it into sentences,
    classify each sentence using Gemini LLM,
    and route/add the sentences to their respective category RAG database.
    """
    if filepath.endswith('.pdf'):
        loader = PyPDFLoader(filepath)
    elif filepath.endswith('.docx'):
        loader = Docx2txtLoader(filepath)
    elif filepath.endswith('.txt'):
        loader = TextLoader(filepath)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

    docs = loader.load()
    full_text = "\n".join(doc.page_content for doc in docs)
    sentences = split_into_sentences(full_text)

    if not sentences:
        return 0

    from langchain_core.documents import Document

    # Classify all sentences using Gemini LLM
    classified = _classify_sentences_with_llm(sentences)

    # Group sentences by category and build debug log
    categorized_docs = {cat: [] for cat in VALID_CATEGORIES}
    debug_log = {cat: [] for cat in VALID_CATEGORIES}

    for category, cat_sentences in classified.items():
        for sentence in cat_sentences:
            doc = Document(
                page_content=sentence,
                metadata={"source": os.path.basename(filepath), "label": category}
            )
            categorized_docs[category].append(doc)
            debug_log[category].append({
                "sentence": sentence,
                "predicted_label": category,
            })

    total_added = 0
    for category, cat_docs in categorized_docs.items():
        if cat_docs:
            vectorstore = get_category_vectorstore(category)
            vectorstore.add_documents(cat_docs)
            total_added += len(cat_docs)

    # Save to a debug file for user verification of accuracy
    debug_path = "rag_debug.json"
    try:
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_log, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Saved RAG classification debug log to {debug_path}")
    except Exception as e:
        print(f"[WARNING] Failed to write {debug_path}: {e}")

    return total_added


def _format_docs(docs):
    """Concatenate retrieved document contents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_category_context(category: str, query: str, k: int = 3) -> str:
    """Safe retrieval of context for a category from its specific RAG database."""
    path = os.path.join(CHROMA_PATH, category.lower())
    if not os.path.exists(path) or not os.listdir(path):
        return f"No reference guidelines found in the {category} database. Please upload reference guidelines for this category."
    try:
        vstore = Chroma(persist_directory=path, embedding_function=get_embeddings())
        retrieved_docs = vstore.similarity_search(query, k=k)
        if not retrieved_docs:
            return f"No relevant reference guidelines found in the {category} database."
        return _format_docs(retrieved_docs)
    except Exception as e:
        print(f"[WARNING] Failed to query vectorstore for {category}: {e}")
        return f"No reference guidelines found in the {category} database."


def evaluate_categories_with_rag(category_texts: dict):
    """
    Evaluate multiple category texts against the category-specific RAG DBs in a SINGLE LLM call.

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

    # Retrieve specific context for each active category
    context_parts = []
    for cat, text in active_cats.items():
        # Query the specific category RAG database using the spoken text as the query
        cat_context = get_category_context(cat, text, k=3)
        context_parts.append(f"--- {cat.upper()} REFERENCE GUIDELINES ---\n{cat_context}")

    combined_context = "\n\n".join(context_parts)

    # Build the category block for the prompt
    cat_block = "\n".join(
        f"--- {cat.upper()} ---\n{text}" for cat, text in active_cats.items()
    )

    prompt = ChatPromptTemplate.from_template(
        """You are an expert speech evaluator.You have to provide a score for a speech based on the given details. Compare the user's spoken text for each category against the category-specific reference guidelines from the knowledge base context.
For EACH category, calculate a similarity score out of 10 based on how well the user followed the recommended guidelines for that category.
For the "direct" category, also evaluate whether the directions given are clear and practical. If context is missing generally do the evalutaion.
Also provide brief suggestions on how the user can improve for each category.

Knowledge Base Context (Category-Specific Guidelines):
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

    try:
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": combined_context, "question": cat_block})
        # Clean potential markdown formatting
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:-3].strip()
        elif response.startswith("```"):
            response = response[3:-3].strip()

        result = json.loads(response)

        for cat in active_cats:
            cat_result = result.get(cat, {})
            score_val = cat_result.get("similarity_score_out_of_10")
            try:
                score_val = float(score_val) if score_val is not None else 0.0
            except (ValueError, TypeError):
                score_val = 0.0
            fallback[cat] = {
                "similarity_score_out_of_10": score_val,
                "suggestions": str(cat_result.get("suggestions", ""))
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


def generate_report(evidence_data: list, score_data: dict, segments: list = None):
    """
    Generate a detailed report from evidence.json data in a SINGLE LLM call.

    Args:
        evidence_data: list of evidence dicts (from evidence.json)
        score_data: dict of all scores (from score.json)
        segments: list of all transcript segments

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
    
    # Minimize segment data to fit context limit safely
    minimized_segments = []
    if segments:
        for seg in segments:
            minimized_segments.append({
                "segment_id": seg.get("segment_id"),
                "speaker": seg.get("speaker"),
                "text": seg.get("text"),
                "emotion": seg.get("emotion")
            })
    segments_str = json.dumps(minimized_segments, indent=2)

    prompt = ChatPromptTemplate.from_template(
        """You are a professional speech coach.The speakers are given a speech template to follow and the evaluation is done based on that. The details for the reporting is given below. Based on the scoring evidence and transcript segments below, generate a detailed performance report and also give reasons for the scores and detailed suggestions, the generated report should be the final outcome. Additionally, provide a short, constructive comment for each individual audio segment based on its content and emotion.

Scores:
{scores}

Evidence:
{evidence}

Transcript Segments:
{segments}

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
  "improvements": ["improvement 1 with specific suggestion", "improvement 2 with specific suggestion"],
  "segment_comments": [
    {{
      "segment_id": <number>,
      "comment": "Constructive comment on this segment"
    }}
  ]
}}"""
    )

    try:
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"scores": score_str, "evidence": evidence_str, "segments": segments_str})

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
