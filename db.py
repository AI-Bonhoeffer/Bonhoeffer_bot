# db.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load .env file
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "bonhoeffer-bot"
def load_vector_store():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Connect to Pinecone
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key,
    )
    return vectorstore


