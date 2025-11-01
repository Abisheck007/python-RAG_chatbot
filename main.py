from ingest import extract_pdf_text_and_images, build_corpus, chunk_documents, embed_and_persist
from retriever import ask_gemini

if __name__ == "__main__":
    print("🧩 Starting complete pipeline...")
    pages = extract_pdf_text_and_images("/home/abisheck/Downloads/CSR.pdf")
    docs = build_corpus(pages)
    chunks = chunk_documents(docs)
    embed_and_persist(chunks)

    print("\n🎯 Chroma DB ready. Let's query it!\n")
    ask_gemini("What Makes Someone a Customer?")

