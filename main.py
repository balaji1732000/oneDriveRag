from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import logging
import json
import os
import requests
import pinecone
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders.onedrive import OneDriveLoader

app = FastAPI()
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv


load_dotenv()

#trigger workflow
# Set environment variables for authentication
os.environ['O365_CLIENT_ID'] = os.environ.get('O365_CLIENT_ID')
os.environ['O365_CLIENT_SECRET'] = os.environ.get("O365_CLIENT_SECRET")

print(os.environ['O365_CLIENT_ID'])
print(os.environ['O365_CLIENT_SECRET'])
# === Global Initialization for RAG System ===
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
print(PINECONE_API_KEY)
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = os.environ.get("INDEX_NAME", "rag")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Adjust based on the embedding model
        metric="cosine",
        spec=pinecone.ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )
index = pc.Index(INDEX_NAME)

# Create a cache folder for the embedding model
cache_dir = os.path.join(os.getcwd(), "cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Load embedding model globally
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=cache_dir)

def retrieve_similar_chunks(query: str, embedding_model, top_k: int = 3) -> str:
    """Finds and returns concatenated text of the top_k relevant chunks."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    retrieved_texts = [match["metadata"]["text"] for match in results.get("matches", [])]
    return "\n".join(retrieved_texts)

def query_gemini(query: str, context: str) -> str:
    """Queries the Gemini API using the retrieved context and returns the answer."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Give an overall answer for the following context:\n{context}\n\nUser Question: {query}. if there is not context in the document written 'No answer found'"
                    }
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"Error: {response.text}"
    
def chunk_text(text, chunk_size=500):
    """Splits text into overlapping chunks of chunk_size characters."""
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size // 2):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# =================== 3. Embed Chunks with Hugging Face =================== #
def embed_text(text_chunks):
    """Generates embeddings for each chunk using a Hugging Face model."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(text_chunks, convert_to_numpy=True)
    return embeddings


@app.post("/process_onedrive_document")
async def process_onedrive_document(request: Request):
    """
    API endpoint to process a document from OneDrive, generate embeddings, and store them in Pinecone.
    """
    try:
        req_body = await request.json()
        drive_id = req_body.get("drive_id")
        object_id = req_body.get("object_id")
        
        if not drive_id or not object_id:
            raise HTTPException(status_code=400, detail="Missing 'drive_id' or 'object_id' in the request body.")
        
        # Initialize OneDriveLoader
        loader = OneDriveLoader(
            drive_id=drive_id,
            object_ids=[object_id],
            auth_with_token=True
        )
        
        # Load document
        documents = loader.load()
        document_text = " ".join([doc.page_content for doc in documents])
        
        # Chunk text
        text_chunks = chunk_text(document_text)
        
        # Generate embeddings
        embeddings = embed_text(text_chunks)
        
        # Store embeddings in Pinecone
        for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
            index.upsert([(f"{object_id}_{i}", embedding.tolist(), {"text": chunk})])
        
        return JSONResponse(content={"message": "Document processed and embeddings stored successfully."}, status_code=200)
    
    except Exception as e:
        logging.error(f"Error processing OneDrive document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/datatransformation")
async def datatransformation(request: Request):
    logging.info("datatransformation endpoint processed a request.")
    try:
        req_body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    
    records = req_body.get("body", [])
    duration_map = {}
    for item in records:
        app_name = item.get("application name", "")
        try:
            exec_duration = float(item.get("execution duration", 0))
        except Exception:
            exec_duration = 0
        duration_map[app_name] = duration_map.get(app_name, 0) + exec_duration

    processed_data = [
        {"application name": app_name, "execution duration": total_duration}
        for app_name, total_duration in duration_map.items()
    ]
    return JSONResponse(content={"records": processed_data}, status_code=200)

@app.post("/datatransformation_summary")
async def datatransformation_summary(request: Request):
    logging.info("datatransformation_summary endpoint processed a request.")
    try:
        req_body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    
    # Accept either a list as a root JSON object or an object with a 'body' field.
    records = req_body if isinstance(req_body, list) else req_body.get("body", [])
    duration_map = {}
    for item in records:
        app_name = item.get("application name", "")
        try:
            exec_duration = float(item.get("execution duration", 0))
        except Exception:
            exec_duration = 0
        if app_name:
            duration_map[app_name] = duration_map.get(app_name, 0) + exec_duration
    return JSONResponse(content=duration_map, status_code=200)

@app.post("/rag_query")
async def rag_query(request: Request):
    logging.info("rag_query endpoint processed a request.")
    try:
        req_body = await request.json()
        user_query = req_body.get("query", "")
        if not user_query:
            raise ValueError("Missing query in the request body.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    
    # Retrieve similar text chunks from Pinecone
    context = retrieve_similar_chunks(user_query, embedding_model)
    # Query Gemini API using the retrieved context and the user's query
    answer = query_gemini(user_query, context)
    
    return JSONResponse(content={"answer": answer}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
