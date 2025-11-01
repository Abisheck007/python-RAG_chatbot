# retriever.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from google import genai
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from rich import print as rprint
import numpy as np
import textwrap
import os

# ============ ENV SETUP ============
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found. Please set it in your .env file")

# ============ CONFIG ============
PERSIST_DIR = "chroma_store"  # Must match ingest.py
HF_MODEL = "sentence-transformers/all-mpnet-base-v2"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_K = 10
GEMINI_MODEL = "gemini-2.5-flash"

# ============ INITIALIZE ============
rprint("[bold cyan]🔹 Initializing Retriever and Embeddings...[/bold cyan]")
embedder = HuggingFaceEmbeddings(model_name=HF_MODEL)
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedder)
total_docs = vectordb._collection.count()
rprint(f"[green]✅ Loaded Chroma with {total_docs} document chunks.[/green]")

retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": RETRIEVAL_K, "fetch_k": 30})
cross_encoder = CrossEncoder(CROSS_MODEL)
client = genai.Client(api_key=GOOGLE_API_KEY)

# ============ RERANK ============
def rerank(question, docs, top_k=8):
    """Re-rank retrieved docs using cross-encoder."""
    if not docs:
        return [], []
    texts = [d.page_content for d in docs]
    pairs = [[question, t] for t in texts]
    scores = cross_encoder.predict(pairs)
    order = np.argsort(scores)[::-1]
    ranked = [docs[i] for i in order]
    ranked_scores = [scores[i] for i in order]
    return ranked[:top_k], ranked_scores[:top_k]


def ask_gemini(question: str, depth: str = "extended"):
    """
    Ask Gemini using retrieved and re-ranked context.
    depth: 'concise' | 'standard' | 'extended'
    """

    
    docs = retriever.get_relevant_documents(question)
    if not docs:
        rprint("[red]⚠️ No relevant chunks retrieved! Check persist dir or ingestion.[/red]")
        return

   
    ranked_docs, scores = rerank(question, docs, top_k=8)
    rprint(f"\n[cyan]🔁 Retrieved & reranked {len(ranked_docs)} top chunks.[/cyan]\n")

    
    context = "\n\n---\n\n".join(
        [f"Page {d.metadata.get('page')}: {d.page_content}" for d in ranked_docs]
    )

    
    if depth == "concise":
        detail_level = "Answer briefly and precisely."
    elif depth == "standard":
        detail_level = "Give a well-explained, paragraph-level answer."
    else:
        detail_level = (
            "Provide a long, structured, and well-organized answer with detailed points. "
            "Explain background, reasoning, and key insights. Use bullets or numbered lists "
            "where appropriate, and provide a short summary at the end."
        )

    
    prompt = f"""
You are a professional assistant that summarizes and explains information ONLY from the document context below.

Rules:
- Base your answer strictly on the provided context.
- Mention the page numbers of the most relevant parts.
- Be formal, clear, and logically structured.
- Do not invent or assume missing data.
- {detail_level}

Context:
{context}

Question: {question}

Answer:
""".strip()

 
    rprint("[bold yellow]✨ Sending refined prompt to Gemini...[/bold yellow]\n")
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    answer = getattr(response, "text", str(response)).strip()

    
    if depth == "extended":
        follow_prompt = f"""
The previous answer was good but too short. Please expand it with deeper explanation,
examples, and a summary section at the end. Make it sound professional and comprehensive.

Previous answer:
{answer}
"""
        response2 = client.models.generate_content(model=GEMINI_MODEL, contents=follow_prompt)
        answer = getattr(response2, "text", str(response2)).strip()

 
    formatted_answer = (
        answer.replace("•", "\n  •")
              .replace(" - ", "\n  - ")
              .replace("\\n", "\n")
              .strip()
    )
    wrapper = textwrap.TextWrapper(width=100)
    formatted_answer = "\n".join(wrapper.wrap(formatted_answer))

   
    print("\n" + "=" * 90)
    rprint("[bold bright_cyan]📘 Gemini Answer[/bold bright_cyan]\n")
    rprint(f"[bold bright_white]{formatted_answer}[/bold bright_white]")
    print("=" * 90 + "\n")

  
    rprint("[bold cyan]📑 Sources Used:[/bold cyan]")
    unique_pages = sorted(set([d.metadata.get("page") for d in ranked_docs if d.metadata.get("page")]))
    for i, page in enumerate(unique_pages, 1):
        print(f"  {i}. Page {page}")
    print("\n")


if __name__ == "__main__":
    rprint("[bold green]🚀 Retriever Ready. Ask your questions below![/bold green]")
    while True:
        q = input("\n🔹 Ask something (or type 'exit'): ").strip()
        if q.lower() == "exit":
            rprint("[bold red]👋 Exiting Retriever. Have a good one![/bold red]")
            break
        depth = input("🧠 Depth (concise / standard / extended) [default=extended]: ").strip().lower()
        if depth not in ["concise", "standard", "extended"]:
            depth = "extended"
        ask_gemini(q, depth)
