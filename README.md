# RAG Chatbot with LangChain & Ollama

This is a **Retrieval-Augmented Generation (RAG) chatbot** using **LangChain**, **FAISS**, and **Ollama** for **local LLM inference**. The chatbot can load and process `.txt` and `.pdf` documents from the `docs/` folder, retrieve relevant content, and generate responses. In ~60 lines of code.

## Features
✅ **Retrieval-Augmented Generation (RAG)** – Enhances responses using document retrieval.  
✅ **Supports `.txt` and `.pdf` files** – Loads all documents from the `docs/` folder.  
✅ **Uses FAISS for efficient vector search** – Enables fast and scalable retrieval.  
✅ **Local AI-powered chatbot** – Runs offline using `sentence-transformers` & `Ollama`.  
✅ **Gradio Web Interface** – Provides a simple UI for user interaction.  

---

## Installation

### 1. **Install Dependencies**
Make sure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Add Your Documents
Place .txt and .pdf files inside the docs/ folder.

### 3. Start the Web Interface
To run the Gradio UI:
```bash
python main.py
```

## How It Works
### Document Loading

* Reads all .txt and .pdf files from docs/.
### Text Splitting & Embeddings

* Splits documents into chunks.
* Converts text into vector embeddings using sentence-transformers.
* Stores embeddings in FAISS for retrieval.
### Query Processing

Uses FAISS to retrieve relevant document chunks.
Feeds them into the Mistral LLM (via Ollama) for response generation.
