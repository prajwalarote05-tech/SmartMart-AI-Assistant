import os
import gradio as gr

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# -----------------------------
# API Key
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

# -----------------------------
# Load PDF
# -----------------------------
loader = PyPDFLoader("SmartMart Policy.pdf")
documents = loader.load()

# -----------------------------
# Split PDF
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# -----------------------------
# Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

retriever = vector_db.as_retriever(
    search_kwargs={"k":2}
)

# -----------------------------
# Chat Function
# -----------------------------
def smartmart_chat(question):

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are SmartMart AI Assistant.

Context:
{context}

Question:
{question}

Answer ONLY using the context.
If the answer is not available in the PDF, reply:
Sorry, I couldn't find that information in the SmartMart knowledge base.
"""

    response = llm.invoke(prompt)

    return response.content

# -----------------------------
# Gradio UI
# -----------------------------
demo = gr.Interface(
    fn=smartmart_chat,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask anything about SmartMart..."
    ),
    outputs="text",
    title="🛒 SmartMart AI Assistant",
    description="Ask questions about SmartMart Policy, Orders, Refunds, Shipping and Products."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
