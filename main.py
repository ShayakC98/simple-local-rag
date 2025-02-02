import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import gradio as gr

# Directory containing documents
DOCS_FOLDER = "docs"

# Load all .txt and .pdf files
def load_documents(folder):
    documents = []
    
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        
        if filename.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    return documents

# Load and split documents
docs = load_documents(DOCS_FOLDER)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

# Create vector store
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Choose a model from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Create vector store with FAISS
vector_db = FAISS.from_documents(documents, embeddings)

# Save vector store
vector_db.save_local("faiss_index")

vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

llm = Ollama(model="deepseek-r1:1.5b")  # Choose the model you pulled earlier

qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever())

def chatbot(query):
    return qa_chain.run(query)

iface = gr.Interface(fn=chatbot, inputs="text", outputs="text")
iface.launch()
