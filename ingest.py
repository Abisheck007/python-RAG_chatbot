import fitz  
import pytesseract
from PIL import Image
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

load_dotenv()

PDF_PATH = Path("/home/abisheck/Downloads/CSR.pdf")
PERSIST_DIR = Path("chroma_store")
PERSIST_DIR.mkdir(exist_ok=True)



def extract_pdf_text_and_images(pdf_path):
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text") or ""
        image_texts = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 4:
                mode = "RGB"
                img_pil = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            else:
                pix = fitz.Pixmap(fitz.csRGB, pix)
                img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img_pil)
            if ocr_text.strip():
                image_texts.append(ocr_text)
        pages.append({"page": page_num + 1, "text": text, "image_texts": image_texts})
    return pages


def clean_text(s):
    s = s.replace("\x0c", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_corpus(pages):
    docs = []
    for p in pages:
        combined = p["text"]
        if p["image_texts"]:
            combined += "\n\n" + "\n\n".join(p["image_texts"])
        combined = clean_text(combined)
        if combined:
            docs.append({"page": p["page"], "content": combined})
    return docs


def chunk_documents(docs, chunk_size=1200, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for d in docs:
        parts = splitter.split_text(d["content"])
        for i, part in enumerate(parts):
            chunks.append({"page": d["page"], "chunk_id": f"{d['page']}_{i}", "text": part})
    return chunks


def embed_and_persist(chunks, persist_dir=PERSIST_DIR):
    print("Creating embeddings and saving to Chroma DB...")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    texts = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"], "chunk_id": c["chunk_id"]} for c in chunks]

    vectordb = Chroma.from_texts(
        texts=texts, embedding=embedder, metadatas=metadatas, persist_directory=str(persist_dir)
    )
    vectordb.persist()
    print(f"✅ Chroma DB persisted at: {persist_dir}")


if __name__ == "__main__":
    print("📄 Extracting text and images from PDF...")
    pages = extract_pdf_text_and_images(PDF_PATH)
    print(f"Pages extracted: {len(pages)}")

    docs = build_corpus(pages)
    chunks = chunk_documents(docs)

    embed_and_persist(chunks)
    print("✅ Ingestion complete.")
