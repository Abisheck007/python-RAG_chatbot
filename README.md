# 🧠 PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows you to **upload PDFs**, **retrieve relevant content**, and **chat interactively** with your documents — powered by **LangChain**, **Chroma**, and an **LLM**.

---

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that combines **document retrieval** with **language model reasoning**.  
You can ask natural language questions, and the bot will fetch the most relevant text chunks from the uploaded PDFs before generating a contextual answer.

**Workflow summary:**
1. Load and preprocess PDF documents.  
2. Split them into manageable text chunks.  
3. Convert chunks into vector embeddings.  
4. Store them in a **Chroma** vector database.  
5. Use a retriever to find relevant chunks per query.  
6. Combine the retrieved content with an **LLM prompt** to generate the final answer.

---

## ⚙️ Tech Stack

| Component | Technology Used |
|------------|----------------|
| **Language** | Python |
| **Framework** | LangChain |
| **Vector Database** | Chroma |
| **Embedding Model** | (e.g. OpenAI / SentenceTransformers / HuggingFace) |
| **LLM** | OpenAI / HuggingFace model |
| **Document Processing** | PyMuPDF or pdfplumber |
| **Environment** | Virtualenv or Conda 



---

## 🧩 How It Works

### 1. Load and Chunk PDFs
The script reads all PDFs from the `data/` folder and splits them into text chunks for efficient retrieval.

### 2. Create Embeddings
Each chunk is converted into vector embeddings and stored in **Chroma**.

### 3. Ask Questions
When you run the chatbot, it retrieves the top relevant chunks for each query and uses the LLM to generate responses.

### 4. Interactive Chat
You can chat directly through the terminal interface:


```bash
$ python retriever.py

Loaded Chroma with 415 document chunks.
🚀 Retriever Ready. Ask your questions below!

🔹 Ask something (or type 'exit'): What is the purpose of this document?
🧩 Relevant context fetched from PDFs...
💬 Answer: This document explains how to identify and engage with customers...
