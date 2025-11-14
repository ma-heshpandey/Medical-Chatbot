import os
from pinecone import Pinecone
from  dotenv import load_dotenv
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embeddings, filter_to_minimal_docs, load_pdf_files, text_split

load_dotenv(override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"]= PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"]= GOOGLE_API_KEY

# Load all pdf files inside the fodler data
extracted_data = load_pdf_files("data")

#  Filter only the necessary metadata and content
filter_data = filter_to_minimal_docs(extracted_data)

# Separate the text into chunks
text_chunk = text_split(filter_data)

# Downloading embeding model
embedding = download_embeddings()


pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


index_name="medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

#call the client's Index constructor
index = pc.Index(index_name)

# Store the docs data in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embedding,
    index_name=index_name
)

# for retrieving embedings vector
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embedding)
