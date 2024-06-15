import os
import time
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.llms import CTransformers
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

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
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=25)
    document = Document(page_content=text)
    return text_splitter.split_documents([document])

# Load and process data
pdf_file_path = "data/Kingdon Field Guide to African Mammals -- Jonathan Kingdon -- 2nd Revised edition, 2015 -- Bloomsbury Natural History -- 9781472912367 -- 2b54816a0e2b7188d843e2356a60fb61 -- Anna’s Archive.pdf"  # Update this path to your single PDF file
extracted_data = load_pdf(pdf_file_path)
if extracted_data is None:
    raise ValueError("The extracted data is None. Please check the load_pdf function.")

print(f"Extracted Data: {extracted_data}")

# Split the extracted text into chunks
text_chunks = text_split(extracted_data)
if not text_chunks:
    raise ValueError("The text_chunks is None or empty. Please check the text_split function.")

print(f"Text Chunks: {text_chunks}")

embeddings = download_hugging_face_embeddings()
if embeddings is None:
    raise ValueError("The embeddings is None. Please check the download_hugging_face_embeddings function.")

print(f"Embeddings: {embeddings}")

# Setup CTransformers LLM
llm=CTransformers(model="model",
                  model_type="llama",
                  config={'max_new_tokens':100,
                          'temperature':0.8})


# Flask routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        input_text = msg
        print(f"Received message: {input_text}")
        
        # Display spinner
        result = {"generated_text": "Thinking..."}
        
        # Simulate processing delay
        time.sleep(1)
        
        # Retrieve response from the model
        result = llm.generate([input_text])
        print(f"LLMResult: {result}")
        
        # Access the generated text from the result object
        if result.generations and result.generations[0]:
            generated_text = result.generations[0][0].text
        else:
            generated_text = "No response generated."
        
        print(f"Response: {generated_text}")
        
        return str(generated_text)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)