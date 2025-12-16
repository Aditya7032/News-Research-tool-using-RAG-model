
#AI News & Document Research Tool with RAG, BM25, Cosine Similarity, and Summarization Includes RAG-powered Q&A on fetched articles (NewsAPI & Google Search)

import os
import re
import json
import tempfile
import hashlib
import requests
import numpy as np
import streamlit as st

from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Text extraction
from PyPDF2 import PdfReader
import docx

# Optional article extraction
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except Exception:
    NEWSPAPER_AVAILABLE = False

# NLP & Similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Spacy (used for entity extraction)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    st.warning("spaCy 'en_core_web_sm' model not found. "
               "Please run 'python -m spacy download en_core_web_sm' in your terminal.")




def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


def summarize_text(text, max_words=250):
    
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
       
        sentences_count = min(8, max(1, int(len(parser.document.sentences) * 0.15)))
        summary = summarizer(parser.document, sentences_count=sentences_count)
        summary_text = " ".join(str(s) for s in summary)
        return " ".join(summary_text.split()[:max_words])
    except Exception:
        
        return " ".join(text.split()[:max_words])


def extract_article_content(url):
    """
    Try to fetch full article text using newspaper. If unavailable or fails,
    return None to indicate fallback to snippet/description.
    """
    if not NEWSPAPER_AVAILABLE:
        return None
    try:
        art = Article(url)
        art.download()
        art.parse()
        txt = art.text
        return txt if txt and len(txt.strip()) > 50 else None
    except Exception:
        return None


def extract_document_metadata(text: str) -> Dict[str, Any]:
    return {
        "total_words": len(text.split()),
        "total_chars": len(text),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }



def structured_output(result: Dict[str, Any], format_type: str = "json") -> str:
    """
    Controlled generation of output formats.
    Supports 'json' and 'table' (markdown).
    """
    if format_type == "json":
        try:
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception:
            return json.dumps({"_error": "Serialization failed", "raw": str(result)}, indent=2)
    elif format_type == "table":
        # Flatten single-level dict to a simple table
        def to_rows(d: Dict[str, Any]) -> List[Tuple[str, Any]]:
            rows = []
            for k, v in d.items():
                if isinstance(v, (dict, list)):
                    rows.append((k, json.dumps(v, ensure_ascii=False)))
                else:
                    rows.append((k, v))
            return rows
        rows = to_rows(result)
        table = "| Key | Value |\n|-----|-------|\n"
        for k, v in rows:
            table += f"| {k} | {v} |\n"
        return table
    else:
        return str(result)
        
def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Extracts named entities from a given text using spaCy.
    Returns a dictionary with entity labels as keys and lists of entities as values.
    """
    if not SPACY_AVAILABLE:
        return {"error": "spaCy model not loaded."}
    
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
        
    return entities


#  News & Google Search 

def fetch_newsapi_articles(api_key, query, from_date=None, to_date=None, language="en", page_size=10):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": api_key
    }
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    r = requests.get(url, params=params)
    data = r.json()
    return data.get("articles", [])


def fetch_google_search(api_key, cx, query, num=10):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "cx": cx, "key": api_key, "num": num}
    r = requests.get(url, params=params)
    return r.json().get("items", [])

#  Ranking 

def rank_documents(docs, query):
    corpus = [d.get("content", "") for d in docs]
   
    tokenized_corpus = [c.split() for c in corpus]
    if any(tokenized_corpus):
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query.split())
    else:
        bm25_scores = np.zeros(len(docs))

    
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus + [query])
        cosine_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    except Exception:
        cosine_scores = np.zeros(len(docs))

    for i, d in enumerate(docs):
        d["bm25_score"] = float(bm25_scores[i]) if len(bm25_scores) > i else 0.0
        d["cosine_score"] = float(cosine_scores[i]) if len(cosine_scores) > i else 0.0

    # sort by combined score (cosine + bm25)
    docs = sorted(docs, key=lambda x: (x.get("bm25_score", 0) + x.get("cosine_score", 0)), reverse=True)
    return docs



@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


@dataclass
class VectorIndex:
    chunks: List[str]
    embeddings: np.ndarray  



if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = {}      
if "article_index_map" not in st.session_state:
    st.session_state["article_index_map"] = {} 
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def chunk_text(text: str, max_tokens: int = 220, overlap: int = 30) -> List[str]:
    sentences = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        toks = s.split()
        if cur_len + len(toks) > max_tokens and cur:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap > 0 else []
            cur_len = len(cur)
        cur += toks
        cur_len += len(toks)
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]

def build_vector_index(text: str) -> VectorIndex:
    doc_hash = _hash_text(text)
    if doc_hash in st.session_state["vector_store"]:
        return st.session_state["vector_store"][doc_hash]

    embedder = get_embedder()
    chunks = chunk_text(text)
    if not chunks:
        chunks = [text]
    embs = embedder.encode(chunks, convert_to_numpy=True)
    embs = normalize(embs)
    idx = VectorIndex(chunks=chunks, embeddings=embs)
    st.session_state["vector_store"][doc_hash] = idx
    return idx

def retrieve_top_k(query: str, index: VectorIndex, k: int = 5) -> List[Dict[str, Any]]:
    embedder = get_embedder()
    q = embedder.encode([query], convert_to_numpy=True)
    q = normalize(q)
    scores = (index.embeddings @ q.T).ravel()
    top_ids = np.argsort(-scores)[:k]
    return [{"chunk": index.chunks[i], "score": float(scores[i])} for i in top_ids]

#  RAG Pipeline 

def rag_summarize(text: str, user_goal="Provide a concise summary.", k=6, max_words=220):
    try:
        idx = build_vector_index(text)
        query = f"{user_goal} Identify key obligations/clauses/findings."
        top = retrieve_top_k(query, idx, k)
        context = "\n\n".join(t["chunk"] for t in top)
        return summarize_text(context, max_words)
    except Exception as e:
        return f"[RAG] Failed: {e}"

def rag_qa(text: str, question: str, k=5, max_words=160):
    try:
        idx = build_vector_index(text)
        top = retrieve_top_k(question, idx, k)
        evidence = "\n\n".join(t["chunk"] for t in top)
        prompt = f"Question: {question}\n\nEvidence:\n{evidence}"
        return summarize_text(prompt, max_words)
    except Exception as e:
        return f"[RAG-QA] Failed: {e}"



def rag_qa_fewshot(text: str, question: str, examples: List[Dict[str, str]], k=5, max_words=160):
    """
    Few-shot RAG Q&A: examples = [{"q": "...", "a": "..."}, ...]
    """
    try:
        idx = build_vector_index(text)
        top = retrieve_top_k(question, idx, k)
        evidence = "\n\n".join(t["chunk"] for t in top)
        example_str = "\n".join([f"Q: {ex.get('q','')}\nA: {ex.get('a','')}" for ex in examples[:5]])
        prompt = f"Follow the style of the examples.\n\nExamples:\n{example_str}\n\nQuestion: {question}\n\nEvidence:\n{evidence}"
        return summarize_text(prompt, max_words)
    except Exception as e:
        return f"[Few-shot RAG-QA] Failed: {e}"

# Document Typing & Agentic Pipelines 

def detect_doc_type(text: str) -> str:
    t = text.lower()
    if "agreement" in t or "party" in t or "termination" in t:
        return "contract"
    if "policy" in t or "compliance" in t:
        return "policy"
    if "abstract" in t or "methodology" in t or "references" in t:
        return "research"
    return "general"

def agentic_pipeline(text: str, task: str, question=""):
    out = {"doc_type": detect_doc_type(text), "task": task}
    if task == "summary":
        out["summary"] = rag_summarize(text, f"Summarize the {out['doc_type']} document.")
    elif task == "qa":
        out["answer"] = rag_qa(text, question)
    return out


def agentic_multistep(text: str, question: str = "") -> Dict[str, Any]:
    """
    Multi-step: detect -> summarize -> metadata -> optional Q&A.
    """
    out = {"doc_type": detect_doc_type(text)}
    out["summary"] = rag_summarize(text, f"Summarize the {out['doc_type']} document.")
    out["metadata"] = extract_document_metadata(text)
    if question:
        out["answer"] = rag_qa(text, question)
    return out

#  Vector Search

def search_across_documents(query: str, top_k=5):
    """
    Perform vector similarity search across all indexed documents/articles.
    """
    if not st.session_state["vector_store"]:
        return []
    embedder = get_embedder()
    q = embedder.encode([query], convert_to_numpy=True)
    q = normalize(q)
    results = []
    for doc_hash, index in st.session_state["vector_store"].items():
        scores = (index.embeddings @ q.T).ravel()
        top_ids = np.argsort(-scores)[:top_k]
        for i in top_ids:
            results.append({
                "doc_hash": doc_hash,
                "chunk": index.chunks[i],
                "score": float(scores[i])
            })
    return sorted(results, key=lambda x: -x["score"])[:top_k]

#  Knowledge Graph 

def extract_entities(text: str) -> List[str]:
    """
    Very lightweight 'entity' extraction:
    - Proper-like tokens (Capitalized words) and long all-caps tokens.
    """
    
    tokens = re.findall(r'\b([A-Z][a-zA-Z]+|[A-Z]{2,})\b', text)
    stops = {"The","And","For","With","From","This","That","In","On","Of","A","An","To","By","As","At","Be","It"}
    ents = [t for t in tokens if t not in stops and len(t) > 1]
   
    seen, out = set(), []
    for e in ents:
        if e not in seen:
            seen.add(e); out.append(e)
    return out[:200]

def knowledge_graph_from_text(text: str) -> Dict[str, Any]:
    """
    Build a co-occurrence graph of 'entities' within sentences.
    Nodes: entities; Edges: co-mentions in same sentence.
    """
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    nodes = {}
    edges = {}
    for sent in sentences:
        ents = extract_entities(sent)
        for e in ents:
            nodes[e] = nodes.get(e, 0) + 1
        # build undirected edges
        for i in range(len(ents)):
            for j in range(i+1, len(ents)):
                a, b = sorted((ents[i], ents[j]))
                key = (a, b)
                edges[key] = edges.get(key, 0) + 1
    node_list = [{"id": n, "weight": w} for n, w in sorted(nodes.items(), key=lambda kv: -kv[1])][:100]
    edge_list = [{"source": a, "target": b, "weight": w} for (a, b), w in sorted(edges.items(), key=lambda kv: -kv[1])][:200]
    return {"nodes": node_list, "edges": edge_list}



def clear_vector_cache():
    st.session_state["vector_store"].clear()

def cache_status():
    num_docs = len(st.session_state["vector_store"])
    num_chunks = sum(len(v.chunks) for v in st.session_state["vector_store"].values())
    return {"documents_indexed": num_docs, "total_chunks": num_chunks}

#  Chatbot 

def chatbot_ui(loaded_text: str, output_format: str = "text"):
    st.subheader("💬 Chat with Document (RAG)")
    for m in st.session_state["chat_history"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask about the uploaded document...")
    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context..."):
                ans = rag_qa(loaded_text, prompt)
                if output_format == "json":
                    payload = {"question": prompt, "answer": ans}
                    st.code(structured_output(payload, "json"), language="json")
                    display = json.dumps(payload)
                elif output_format == "table":
                    st.markdown(structured_output({"question": prompt, "answer": ans}, "table"))
                    display = ans
                else:
                    st.markdown(ans)
                    display = ans
        st.session_state["chat_history"].append({"role": "assistant", "content": display})

# Streamlit UI 

st.set_page_config(layout="wide")
st.title("📰 AI News & Document Research Tool — RAG & Q&A Enabled")

st.sidebar.title("Options")
mode = st.sidebar.radio("Choose Mode", [
    "Upload Document (LSA Summarizer)",
    "Fetch News",
    "Google Search",
    "Upload & RAG Summary",
    "Chat with Document (RAG)",
    "Vector Search (All Indexed)",         
    "Agentic Multi-step Analysis",          
    "Few-shot RAG Q&A",                     
    "Knowledge Graph (Lightweight)",         
    "Named Entity Recognition (NER)"         
])

st.sidebar.markdown("---")
st.sidebar.subheader("Structured Output")
out_fmt = st.sidebar.selectbox("Output format", ["text", "json", "table"])

st.sidebar.markdown("---")
st.sidebar.subheader("Context Cache")
colC1, colC2 = st.sidebar.columns(2)
with colC1:
    if st.button("View Cache"):
        st.sidebar.json(cache_status())
with colC2:
    if st.button("Clear Cache"):
        clear_vector_cache()
        st.sidebar.success("Cleared vector cache.")

# Document Upload + LSA 
if mode == "Upload Document (LSA Summarizer)":
    uploaded = st.file_uploader("Upload PDF/DOCX/TXT", type=["pdf", "docx", "txt"])
    if uploaded:
        ext = uploaded.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(uploaded.read())
            path = tmp.name
        if ext == "pdf":
            content = extract_text_from_pdf(path)
        elif ext == "docx":
            content = extract_text_from_docx(path)
        else:
            content = open(path, "r", encoding="utf-8").read()

        st.text_area("Extracted Text (preview)", content[:3000], height=200)
        if st.button("Summarize"):
            summary = summarize_text(content)
            meta = extract_document_metadata(content)
            result = {"summary": summary, "metadata": meta}
            st.subheader("Summary")
            if out_fmt == "json":
                st.code(structured_output(result, "json"), language="json")
            elif out_fmt == "table":
                st.markdown(structured_output(result, "table"))
            else:
                st.write(summary)
                st.subheader("Metadata")
                st.json(meta)
        st.session_state["last_doc_text"] = content
        
        build_vector_index(content)

# Fetch News 
elif mode == "Fetch News":
    query = st.text_input("Search Query (NewsAPI)")
    api_key = st.text_input("NewsAPI Key", type="password")
    page_size = st.slider("Max results", 1, 30, 10)
    if st.button("Fetch News") and api_key and query:
        articles = fetch_newsapi_articles(api_key, query, page_size=page_size)
        docs = []
        st.session_state["article_index_map"] = {} 
        for i, a in enumerate(articles):
            
            url = a.get("url", "")
            full_text = extract_article_content(url) if url else None
            content = full_text or (a.get("content") or a.get("description") or "")
            if not content:
                content = a.get("title", "")
            article_id = f"news_{i}"
            docs.append({
                "id": article_id,
                "title": a.get("title", ""),
                "content": content,
                "url": url,
                "publishedAt": a.get("publishedAt")
            })
            st.session_state["article_index_map"][article_id] = {"text": content, "hash": _hash_text(content)}
            
            build_vector_index(content)

        ranked = rank_documents(docs, query)
        st.success(f"Fetched {len(docs)} articles. Showing ranked results:")
        for art in ranked:
            aid = art["id"]
            with st.expander(f"🔎 {art['title']}"):
                st.markdown(f"Source: {art.get('url','N/A')}")
                st.write(art.get("content")[:2000] + ("..." if len(art.get("content",""))>2000 else ""))
                
                col1, col2 = st.columns([1,2])
                with col1:
                    if st.button(f"RAG Summarize {aid}", key=f"rag_sum_{aid}"):
                        with st.spinner("Running RAG summary..."):
                            text = st.session_state["article_index_map"][aid]["text"]
                            summary = rag_summarize(text, user_goal="Summarize article key points", k=6, max_words=220)
                            payload = {"id": aid, "summary": summary}
                            if out_fmt == "json":
                                st.code(structured_output(payload, "json"), language="json")
                            elif out_fmt == "table":
                                st.markdown(structured_output(payload, "table"))
                            else:
                                st.info(summary)
                with col2:
                    q = st.text_input("Ask a question about this article", key=f"qa_input_{aid}")
                    if st.button(f"Ask (RAG) {aid}", key=f"qa_btn_{aid}"):
                        with st.spinner("Running RAG Q&A..."):
                            text = st.session_state["article_index_map"][aid]["text"]
                            ans = rag_qa(text, q, k=6, max_words=200)
                            payload = {"id": aid, "question": q, "answer": ans}
                            if out_fmt == "json":
                                st.code(structured_output(payload, "json"), language="json")
                            elif out_fmt == "table":
                                st.markdown(structured_output(payload, "table"))
                            else:
                                st.success(ans)

# Google Search 
elif mode == "Google Search":
    query = st.text_input("Search Query (Google Custom Search)")
    api_key = st.text_input("Google API Key", type="password")
    cx = st.text_input("Custom Search CX ID")
    num = st.slider("Max results", 1, 10, 5)
    if st.button("Search") and api_key and cx and query:
        items = fetch_google_search(api_key, cx, query, num=num)
        docs = []
        st.session_state["article_index_map"] = {}
        for i, it in enumerate(items):
            url = it.get("link", "")
            full_text = extract_article_content(url) if url else None
            content = full_text or it.get("snippet", "") or it.get("title", "")
            aid = f"google_{i}"
            docs.append({"id": aid, "title": it.get("title",""), "content": content, "url": url})
            st.session_state["article_index_map"][aid] = {"text": content, "hash": _hash_text(content)}
            
            build_vector_index(content)

        ranked = rank_documents(docs, query)
        st.success(f"Found {len(docs)} results. Showing ranked order:")
        for res in ranked:
            aid = res["id"]
            with st.expander(f"🔎 {res['title']}"):
                st.markdown(f"Source: {res.get('url','N/A')}")
                st.write(res.get("content")[:2000] + ("..." if len(res.get("content",""))>2000 else ""))
                col1, col2 = st.columns([1,2])
                with col1:
                    if st.button(f"RAG Summarize {aid}", key=f"gs_rag_sum_{aid}"):
                        with st.spinner("Running RAG summary..."):
                            text = st.session_state["article_index_map"][aid]["text"]
                            summary = rag_summarize(text, user_goal="Summarize article key points", k=6, max_words=220)
                            payload = {"id": aid, "summary": summary}
                            if out_fmt == "json":
                                st.code(structured_output(payload, "json"), language="json")
                            elif out_fmt == "table":
                                st.markdown(structured_output(payload, "table"))
                            else:
                                st.info(summary)
                with col2:
                    q = st.text_input("Ask a question about this result", key=f"gs_qa_input_{aid}")
                    if st.button(f"Ask (RAG) {aid}", key=f"gs_qa_btn_{aid}"):
                        with st.spinner("Running RAG Q&A..."):
                            text = st.session_state["article_index_map"][aid]["text"]
                            ans = rag_qa(text, q, k=6, max_words=200)
                            payload = {"id": aid, "question": q, "answer": ans}
                            if out_fmt == "json":
                                st.code(structured_output(payload, "json"), language="json")
                            elif out_fmt == "table":
                                st.markdown(structured_output(payload, "table"))
                            else:
                                st.success(ans)

#  RAG Summary 
elif mode == "Upload & RAG Summary":
    uploaded = st.file_uploader("Upload Document (pdf/docx/txt)", type=["pdf", "docx", "txt"])
    doc_type = st.selectbox("Select doc type (helps guidance)", ["general", "contract", "policy", "research"])
    if uploaded:
        ext = uploaded.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(uploaded.read())
            path = tmp.name
        if ext == "pdf":
            content = extract_text_from_pdf(path)
        elif ext == "docx":
            content = extract_text_from_docx(path)
        else:
            content = open(path, "r", encoding="utf-8").read()

        st.text_area("Extracted Text (preview)", content[:4000], height=220)
        if st.button("Process with RAG"):
            st.subheader("Agentic Summary")
            result = agentic_pipeline(content, task="summary")
            if out_fmt == "json":
                st.code(structured_output(result, "json"), language="json")
            elif out_fmt == "table":
                st.markdown(structured_output(result, "table"))
            else:
                st.json(result)
        st.session_state["last_doc_text"] = content
        build_vector_index(content)

#  Chatbot 
elif mode == "Chat with Document (RAG)":
    if "last_doc_text" in st.session_state:
        chatbot_ui(st.session_state["last_doc_text"], output_format=out_fmt)
    else:
        st.warning("Please upload a document first in 'Upload & RAG Summary' mode.")

# Vector Search 
elif mode == "Vector Search (All Indexed)":
    q = st.text_input("🔍 Enter a semantic search query")
    top_k = st.slider("Top K", 1, 15, 5)
    if q:
        results = search_across_documents(q, top_k=top_k)
        st.subheader("Vector Search Results")
        for r in results:
            st.write(f"Score: {r['score']:.3f} — Doc Hash: {r['doc_hash']}")
            st.info(r["chunk"])

# Agentic Multi-step Analysis 
elif mode == "Agentic Multi-step Analysis":
    if "last_doc_text" not in st.session_state:
        st.warning("Please upload a document first in 'Upload & RAG Summary' mode.")
    else:
        question = st.text_input("Optional guiding question for the agent")
        if st.button("Run Agent"):
            with st.spinner("Running multi-step agent..."):
                result = agentic_multistep(st.session_state["last_doc_text"], question)
                if out_fmt == "json":
                    st.code(structured_output(result, "json"), language="json")
                elif out_fmt == "table":
                    st.markdown(structured_output(result, "table"))
                else:
                    st.json(result)

# Few-shot RAG Q&A 
elif mode == "Few-shot RAG Q&A":
    if "last_doc_text" not in st.session_state:
        st.warning("Please upload a document first in 'Upload & RAG Summary' mode.")
    else:
        st.write("Provide up to 3 example Q&A pairs to steer answers.")
        ex_q1 = st.text_input("Example 1 - Q", key="ex_q1")
        ex_a1 = st.text_area("Example 1 - A", key="ex_a1")
        ex_q2 = st.text_input("Example 2 - Q", key="ex_q2")
        ex_a2 = st.text_area("Example 2 - A", key="ex_a2")
        ex_q3 = st.text_input("Example 3 - Q", key="ex_q3")
        ex_a3 = st.text_area("Example 3 - A", key="ex_a3")
        question = st.text_input("Your question")
        if st.button("Ask with Few-shot"):
            examples = []
            for q_, a_ in [(ex_q1, ex_a1), (ex_q2, ex_a2), (ex_q3, ex_a3)]:
                if q_ or a_:
                    examples.append({"q": q_ or "", "a": a_ or ""})
            with st.spinner("Running Few-shot RAG Q&A..."):
                ans = rag_qa_fewshot(st.session_state["last_doc_text"], question, examples, k=6, max_words=200)
                payload = {"question": question, "answer": ans, "examples_used": len(examples)}
                if out_fmt == "json":
                    st.code(structured_output(payload, "json"), language="json")
                elif out_fmt == "table":
                    st.markdown(structured_output(payload, "table"))
                else:
                    st.success(ans)

# Knowledge Graph 
elif mode == "Knowledge Graph (Lightweight)":
    if "last_doc_text" not in st.session_state:
        st.warning("Please upload a document first in 'Upload & RAG Summary' mode.")
    else:
        kg = knowledge_graph_from_text(st.session_state["last_doc_text"])
        st.subheader("Entity Co-occurrence Graph (JSON)")
        st.code(json.dumps(kg, indent=2, ensure_ascii=False), language="json")
        st.info("Tip: Export this JSON to visualize with tools like D3, Cytoscape, or Gephi.")
        

elif mode == "Named Entity Recognition (NER)":
    if "last_doc_text" not in st.session_state:
        st.warning("Please upload a document first in 'Upload & RAG Summary' mode.")
    else:
        st.subheader("Named Entities Extracted")
        if SPACY_AVAILABLE:
            entities = extract_named_entities(st.session_state["last_doc_text"])
            st.json(entities)
        else:
            st.error("spaCy is not available. Please install it as instructed in the warning.")


st.sidebar.markdown("---")

st.sidebar.write("Tip: For best RAG results, provide full article text via NewsAPI 'content' or enable newspaper library to fetch article bodies.")
