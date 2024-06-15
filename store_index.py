import os
import time
from src.helper import PINECONE_API_KEY, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as LangchainPinecone  # Alias to avoid confusion
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from PyPDF2 import PdfReader

# Define the load_pdf function
def load_pdf(file_path):
    all_text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            all_text += page.extract_text() + "\n"
    return all_text if all_text else None

# Define the text_split function
def text_split(text):
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_text(text)

# Load environment variables if not already set
load_dotenv()

# Load and process data
pdf_file_path = "data/Kingdon Field Guide to African Mammals -- Jonathan Kingdon -- 2nd Revised edition, 2015 -- Bloomsbury Natural History -- 9781472912367 -- 2b54816a0e2b7188d843e2356a60fb61 -- Anna’s Archive.pdf"  # Update this path to your single PDF file
extracted_data = load_pdf(pdf_file_path)
if extracted_data is None:
    raise ValueError("The extracted data is None. Please check the load_pdf function.")

print(f"Extracted Data: {extracted_data}")

# Split the extracted text into chunks
text_chunks = text_split(extracted_data)
if text_chunks is None:
    raise ValueError("The text_chunks is None. Please check the text_split function.")

print(f"Text Chunks: {text_chunks}")

embeddings = download_hugging_face_embeddings()
if embeddings is None:
    raise ValueError("The embeddings is None. Please check the download_hugging_face_embeddings function.")

print(f"Embeddings: {embeddings}")

# Ensure Pinecone API key is available
api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set.")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Specify cloud and region for the serverless index
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

# Define the index name
index_name = "wildlife-bot"

# Create the index if it does not exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=spec
    )
    # Wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Connect to the created index
index = pc.Index(index_name)
time.sleep(1)

# Example: Add data to the index with reduced metadata
# Create a dictionary to simulate external storage of text chunks
text_chunk_store = {}

# Function to simulate storing text chunk and returning a reference ID
def store_text_chunk(text_chunk):
    chunk_id = f"chunk_{len(text_chunk_store)}"
    text_chunk_store[chunk_id] = text_chunk
    return chunk_id

# Add text chunks to Pinecone with reference IDs
for i, text_chunk in enumerate(text_chunks):
    chunk_id = store_text_chunk(text_chunk)
    embedding = embeddings.embed_query(text_chunk)  # Embed the text chunk
    index.upsert(
        vectors=[
            {
                "id": f"vec_{i}", 
                "values": embedding, 
                "metadata": {"chunk_id": chunk_id}  # Only store the reference ID as metadata
            }
        ],
        namespace="ns1"
    )

print("Indexing completed successfully.")