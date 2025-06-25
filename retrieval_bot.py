import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()


# Pinecone setup
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "bonhoeffer-bot"

# Safety check
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not set in .env")

# Example: English Flyers as documents
docs = [
    """  Price of Silent Generator Models

Model: BON-DG-11KW-1P (220V, 50HZ)
Price: $3,974.15

Model: BON-DG-17KW-1P (220V, 50HZ)
Price: $4,899.44

Model: BON-DG-20KW-1P (220V, 50HZ)
Price: $5,314.67

Model: BON-DG-25KW-1P (220V, 50HZ)
Price: $5,977.02

Model: BON-DG-30KW-1P (220V, 50HZ)
Price: $5,676.73

Model: BON-DG-36KW-1P (220V, 50HZ)
Price: $6,482.77

Model: BON-DG-40KW-1P (220V, 50HZ)
Price: $6,482.77

Model: BON-DG-45KW-1P (220V, 50HZ)
Price: $7,554.60

Model: BON-DG-55KW-1P (220V, 50HZ)
Price: $8,314.67

Model: BON-DG-11KW-3P (220/380V, 50HZ)
Price: $3,998.58

Model: BON-DG-17KW-3P (220/380V, 50HZ)
Price: $4,127.88

Model: BON-DG-20KW-3P (220/380V, 50HZ)
Price: $4,856.33

Model: BON-DG-25KW-3P (220/380V, 50HZ)
Price: $5,053.17

Model: BON-DG-30KW-3P (220/380V, 50HZ)
Price: $5,889.38

Model: BON-DG-36KW-3P (220/380V, 50HZ)
Price: $6,201.15

Model: BON-DG-40KW-3P (220/380V, 50HZ)
Price: $6,285.93

Model: BON-DG-45KW-3P (220/380V, 50HZ)
Price: $6,893.69

Model: BON-DG-55KW-3P (220/380V, 50HZ)
Price: $7,764.38

Model: BON-DG-11KW-1P (120/240V, 60HZ)
Price: $3,577.59

Model: BON-DG-17KW-1P (120/240V, 60HZ)
Price: $3,814.67

Model: BON-DG-20KW-1P (120/240V, 60HZ)
Price: $4,077.59

Model: BON-DG-25KW-1P (120/240V, 60HZ)
Price: $5,402.30

Model: BON-DG-30KW-1P (120/240V, 60HZ)
Price: $5,646.57

Model: BON-DG-36KW-1P (120/240V, 60HZ)
Price: $6,794.55

Model: BON-DG-40KW-1P (120/240V, 60HZ)
Price: $6,880.75

Model: BON-DG-45KW-1P (120/240V, 60HZ)
Price: $8,318.98

Model: BON-DG-55KW-1P (120/240V, 60HZ)
Price: $9,001.44

Model: BON-DG-11KW-1P (220V, 60HZ)
Price: $3,577.59

Model: BON-DG-17KW-1P (220V, 50HZ)
Price: $3,814.67

Model: BON-DG-20KW-1P (220V, 60HZ)
Price: $4,077.59

Model: BON-DG-25KW-1P (220V, 60HZ)
Price: $5,402.30

Model: BON-DG-30KW-1P (220V, 60HZ)
Price: $5,646.57

Model: BON-DG-36KW-1P (220V, 60HZ)
Price: $6,794.55

Model: BON-DG-40KW-1P (220V, 60HZ)
Price: $6,880.75

Model: BON-DG-45KW-1P (220V, 60HZ)
Price: $8,318.98

Model: BON-DG-55KW-1P (220V, 60HZ)
Price: $9,001.44

Model: BON-DG-11KW-3P (208/120V, 60HZ)
Price: $3,577.59

Model: BON-DG-17KW-3P (208/120V, 60HZ)
Price: $3,893.69

Model: BON-DG-20KW-3P (208/120V, 60HZ)
Price: $3,962.65

Model: BON-DG-25KW-3P (208/120V, 60HZ)
Price: $5,020.13

Model: BON-DG-30KW-3P (208/120V, 60HZ)
Price: $5,373.58

Model: BON-DG-36KW-3P (208/120V, 60HZ)
Price: $5,813.23

Model: BON-DG-40KW-3P (208/120V, 60HZ)
Price: $6,629.32

Model: BON-DG-45KW-3P (208/120V, 60HZ)
Price: $6,794.55

Model: BON-DG-55KW-3P (208/120V, 60HZ)
Price: $8,298.87

Model: BON-DG-11KW-3P (220/127V, 60HZ)
Price: $3,556.04

Model: BON-DG-17KW-3P (220/127V, 60HZ)
Price: $3,893.69

Model: BON-DG-20KW-3P (220/127V, 60HZ)
Price: $3,962.65

Model: BON-DG-25KW-3P (220/127V, 60HZ)
Price: $5,020.13

Model: BON-DG-30KW-3P (220/127V, 60HZ)
Price: $5,373.58

Model: BON-DG-36KW-3P (220/127V, 60HZ)
Price: $5,755.75

Model: BON-DG-40KW-3P (220/127V, 60HZ)
Price: $6,482.77

Model: BON-DG-45KW-3P (220/127V, 60HZ)
Price: $6,794.55

Model: BON-DG-55KW-3P (220/127V, 60HZ)
Price: $8,255.75
"""
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Reduced for speed
    chunk_overlap=30
)
split_docs = text_splitter.create_documents(docs)

print(f"Total chunks created: {len(split_docs)}")
# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store flyers in Pinecone
vectorstore = PineconeVectorStore.from_texts(
    texts=docs,
    embedding=embeddings,
    index_name=index_name,
    pinecone_api_key=pinecone_api_key,
)

# Perform a test query
query = "give basic info"
results = vectorstore.similarity_search(query, k=2)

# Output the results
print("\nTop Matching Documents:\n")
for i, result in enumerate(results, start=1):
    print(f"Result {i}:\n{result.page_content}\n{'-'*60}")