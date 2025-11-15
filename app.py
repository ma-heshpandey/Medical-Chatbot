import os
from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from src.prompt import *
from  dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"]= PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"]= GOOGLE_API_KEY

import os
from dotenv import load_dotenv
from flask import Flask, render_template, request

from src.helper import download_embeddings
from src.prompt import system_prompt
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv(override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY or ""

# Create embedding model object
embedding = download_embeddings()

# Since we have already created the embedding in Pinecone, use that index.
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)

# Search on similarity basis, retrieving the 3 most similar documents
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Using Gemini 2.5 flash for LLM model
chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

app = Flask(__name__)


@app.route("/")
def index():
    """Render chat UI."""
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    """Handle chat requests from the frontend."""
    msg = request.form.get("msg", "")
    user_input = msg
    print(user_input)
    response = rag_chain.invoke({"input": user_input})
    print("Response:", response)
    return str(response.get("answer", ""))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)