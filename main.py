from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from docling.document_converter import DocumentConverter
from docling.datamodel.document import TableItem, TextItem

import tempfile
import os
import time
import logging
import uuid
import json 
from datetime import datetime

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue

import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pymongo import MongoClient

# ---------------------------------------------------
# 1. Logging Configuration
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# 2. Environment & Application Setup
# ---------------------------------------------------
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# 3. Initialize Docling Converter
# ---------------------------------------------------
converter = DocumentConverter()

# ---------------------------------------------------
# 4. Qdrant Setup (Persistent Storage)
# ---------------------------------------------------
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docling_parser_demo")
EMBEDDING_DIMENSION = 3072 # gemini-embedding-001 produces 3072 dimensions

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    prefer_grpc=False,
    timeout=120  # Increased timeout for large uploads
)

# Ensure the collection exists with correct vector size
if qdrant.collection_exists(QDRANT_COLLECTION):
    collection_info = qdrant.get_collection(QDRANT_COLLECTION)
    current_size = collection_info.config.params.vectors.size
    if current_size != EMBEDDING_DIMENSION:
        logger.warning(f"Collection dimension mismatch ({current_size} vs {EMBEDDING_DIMENSION}). Recreating collection...")
        qdrant.delete_collection(QDRANT_COLLECTION)
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )
else:
    logger.info(f"Creating new collection: {QDRANT_COLLECTION}")
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=Distance.COSINE
        )
    )

# Explicitly verify/create indexes to prevent '400 Bad Request'
try:
    logger.info("Verifying payload indexes...")
    qdrant.create_payload_index(QDRANT_COLLECTION, "file_id", "keyword")
    qdrant.create_payload_index(QDRANT_COLLECTION, "file_name", "keyword")
    qdrant.create_payload_index(QDRANT_COLLECTION, "lender_name", "keyword")
    qdrant.create_payload_index(QDRANT_COLLECTION, "lender_type", "keyword")
except Exception as e:
    logger.info(f"Indexing note: {e}")

# ---------------------------------------------------
# 5. MongoDB Setup (Chat Storage)
# ---------------------------------------------------
MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB = os.getenv("MONGO_DB", "docling_parser")

mongo_client = MongoClient(MONGO_URL)
mongo_db = mongo_client[MONGO_DB]
conversations_collection = mongo_db["conversations"]
messages_collection = mongo_db["messages"]

# Create indexes for efficient queries
conversations_collection.create_index("conversation_id", unique=True)
messages_collection.create_index("conversation_id")
messages_collection.create_index("timestamp")

logger.info(f"MongoDB connected: {MONGO_DB}")

# ---------------------------------------------------
# 6. Gemini Setup (AI & Embeddings)
# ---------------------------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def embed_text(text: str, is_query: bool = False):
    """
    Converts text into a 3072-dimension vector.
    Uses models/gemini-embedding-001.
    """
    if is_query:
        task = "retrieval_query"
    else:
        task = "retrieval_document"
    
    return genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type=task
    )["embedding"]

llm = genai.GenerativeModel("gemini-2.0-flash")

# ---------------------------------------------------
# 7. Intent Extractor (AI Detective)
# ---------------------------------------------------
def extract_metadata_from_question(question: str):
    """
    Asks Gemini to check if the user specified a lender or product type.
    """
    prompt = f"""
    Look at this mortgage guideline question: "{question}"
    
    Extract the following entities if they are explicitly mentioned:
    1. lender_name
    2. lender_type (e.g., VA, FHA, USDA, Conventional, Freddie Mac)
    
    Return the result ONLY as a JSON object. If not mentioned, use null.
    Example: {{"lender_name": "theLender", "lender_type": "VA"}}
    """
    try:
        response = llm.generate_content(prompt)
        json_str = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        return {"lender_name": None, "lender_type": None}

# ---------------------------------------------------
# 8. Docling Parsing Helper
# ---------------------------------------------------
def parse_item(item, doc):
    """
    Logic for processing individual document items (Text vs Table).
    """
    if item.prov and len(item.prov) > 0:
        page_no = item.prov[0].page_no
    else:
        page_no = 1

    if isinstance(item, TextItem):
        text = item.text.strip()
        if not text:
            return None
        return {
            "clean_text": text,
            "type": "text",
            "page_no": page_no
        }

    if isinstance(item, TableItem):
        df = item.export_to_dataframe(doc)
        csv_text = df.to_csv(index=False)
        return {
            "clean_text": csv_text,
            "type": "table",
            "page_no": page_no
        }

    return None

# ---------------------------------------------------
# 9. Upload Endpoint (Detailed Logic)
# ---------------------------------------------------
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    lender_name: str = Form(...),
    lender_type: str = Form(...)
):
    start_total_time = time.perf_counter()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Generate a unique file_id for this document
    file_id = str(uuid.uuid4())
    logger.info(f"Upload received: {file.filename} | Lender: {lender_name} | File ID: {file_id}")

    try:
        # STEP 1: Overwrite Protection - delete by file_name (for re-uploads of same file)
        qdrant.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="file_name", match=MatchValue(value=file.filename))]
            )
        )

        # STEP 2: Conversion
        result = converter.convert(tmp_path)
        doc = result.document

        # STEP 3: Iterate and Parse
        parsed_items = []
        for item, _ in doc.iterate_items():
            parsed = parse_item(item, doc)
            if parsed is not None:
                parsed_items.append(parsed)

        # STEP 4: Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=100
        )

        # STEP 5: Prepare Points for Qdrant
        points = []
        for idx, p in enumerate(parsed_items):
            text_chunks = splitter.split_text(p["clean_text"])
            
            for chunk_text in text_chunks:
                point_id = str(uuid.uuid4())
                vector = embed_text(chunk_text, is_query=False)
                
                payload = {
                    "text": chunk_text,
                    "type": p["type"],
                    "file_id": file_id,
                    "file_name": file.filename,
                    "order": idx,
                    "page_no": p["page_no"],
                    "lender_name": lender_name,
                    "lender_type": lender_type
                }
                
                points.append({
                    "id": point_id,
                    "vector": vector,
                    "payload": payload
                })

        # STEP 6: Upsert in batches to avoid timeout
        BATCH_SIZE = 50
        total_points = len(points)
        
        for i in range(0, total_points, BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=batch)
            logger.info(f"Uploaded batch {i // BATCH_SIZE + 1}/{(total_points + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch)} points)")
        
        total_time = time.perf_counter() - start_total_time
        logger.info(f"Upload finished in {total_time:.2f}s | Total chunks: {total_points}")

        return {
            "status": "success",
            "file_id": file_id,
            "file": file.filename,
            "lender": lender_name,
            "type": lender_type,
            "chunks_count": total_points
        }

    except Exception as e:
        logger.exception("Upload process failed")
        return {"error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------------------------------------------------
# 10. Collection Status Endpoint
# ---------------------------------------------------
@app.get("/collection/status")
def get_collection_status():
    try:
        collection_info = qdrant.get_collection(QDRANT_COLLECTION)
        scroll_result = qdrant.scroll(
            collection_name=QDRANT_COLLECTION, 
            limit=1000, 
            with_payload=["file_id", "file_name", "lender_name", "lender_type"]
        )
        
        # Build a dict to get unique documents with their metadata
        documents_dict = {}
        for point in scroll_result[0]:
            if point.payload and "file_id" in point.payload:
                file_id = point.payload["file_id"]
                if file_id not in documents_dict:
                    documents_dict[file_id] = {
                        "file_id": file_id,
                        "file_name": point.payload.get("file_name"),
                        "lender_name": point.payload.get("lender_name"),
                        "lender_type": point.payload.get("lender_type")
                    }
        
        documents_list = list(documents_dict.values())
        
        return {
            "status": "ready" if len(documents_list) > 0 else "empty",
            "collection_name": QDRANT_COLLECTION,
            "points_count": collection_info.points_count,
            "documents": documents_list,
            "documents_count": len(documents_list)
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {"status": "empty", "documents": [], "documents_count": 0, "points_count": 0, "error": str(e)}

# ---------------------------------------------------
# 11. Delete Document Endpoint
# ---------------------------------------------------
@app.delete("/document/{file_id}")
def delete_document(file_id: str):
    """Delete all chunks associated with a specific document by file_id from Qdrant"""
    logger.info(f"Delete request for document ID: {file_id}")
    
    try:
        # Delete all points with matching file_id
        qdrant.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
            )
        )
        
        logger.info(f"Successfully deleted document ID: {file_id}")
        return {
            "status": "success",
            "message": f"Document and all its chunks have been deleted"
        }
    except Exception as e:
        logger.exception(f"Failed to delete document ID: {file_id}")
        return {
            "status": "error",
            "message": str(e)
        }

# ---------------------------------------------------
# 12. Conversation Endpoints (MongoDB)
# ---------------------------------------------------

# Pydantic models for conversations
class ConversationCreate(BaseModel):
    title: Optional[str] = None

class MessageCreate(BaseModel):
    conversation_id: str
    content: str

@app.post("/conversations")
def create_conversation(data: ConversationCreate = None):
    """Create a new conversation"""
    conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow()
    
    conversation = {
        "conversation_id": conversation_id,
        "title": data.title if data and data.title else "New Conversation",
        "created_at": now,
        "updated_at": now,
        "message_count": 0
    }
    
    conversations_collection.insert_one(conversation)
    logger.info(f"Created conversation: {conversation_id}")
    
    return {
        "conversation_id": conversation_id,
        "title": conversation["title"],
        "created_at": conversation["created_at"].isoformat(),
        "updated_at": conversation["updated_at"].isoformat(),
        "message_count": 0
    }

@app.get("/conversations")
def list_conversations():
    """List all conversations, sorted by most recent"""
    conversations = conversations_collection.find().sort("updated_at", -1)
    
    result = []
    for conv in conversations:
        result.append({
            "conversation_id": conv["conversation_id"],
            "title": conv.get("title", "Untitled"),
            "created_at": conv["created_at"].isoformat(),
            "updated_at": conv["updated_at"].isoformat(),
            "message_count": conv.get("message_count", 0)
        })
    
    return {"conversations": result}

@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    """Get a conversation with all its messages"""
    conversation = conversations_collection.find_one({"conversation_id": conversation_id})
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = messages_collection.find(
        {"conversation_id": conversation_id}
    ).sort("timestamp", 1)
    
    messages_list = []
    for msg in messages:
        messages_list.append({
            "message_id": msg["message_id"],
            "role": msg["role"],
            "content": msg["content"],
            "sources": msg.get("sources", []),
            "timestamp": msg["timestamp"].isoformat()
        })
    
    return {
        "conversation_id": conversation["conversation_id"],
        "title": conversation.get("title", "Untitled"),
        "created_at": conversation["created_at"].isoformat(),
        "updated_at": conversation["updated_at"].isoformat(),
        "messages": messages_list
    }

@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages"""
    # Delete messages first
    messages_collection.delete_many({"conversation_id": conversation_id})
    
    # Delete conversation
    result = conversations_collection.delete_one({"conversation_id": conversation_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    logger.info(f"Deleted conversation: {conversation_id}")
    return {"status": "success", "message": "Conversation deleted"}

# ---------------------------------------------------
# 13. Chat Endpoint (with MongoDB storage)
# ---------------------------------------------------
class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    top_k: int = 5

@app.post("/chat")
def chat(req: ChatRequest):
    logger.info(f"Processing question: {req.question}")
    
    now = datetime.utcnow()
    conversation_id = req.conversation_id
    
    # Create new conversation if not provided
    if not conversation_id:
        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
        # Auto-generate title from first message (first 50 chars)
        title = req.question[:50] + "..." if len(req.question) > 50 else req.question
        
        conversations_collection.insert_one({
            "conversation_id": conversation_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0
        })
        logger.info(f"Auto-created conversation: {conversation_id}")
    
    # Save user message to MongoDB
    user_message_id = f"msg_{uuid.uuid4().hex[:12]}"
    messages_collection.insert_one({
        "message_id": user_message_id,
        "conversation_id": conversation_id,
        "role": "user",
        "content": req.question,
        "sources": [],
        "timestamp": now
    })

    # STEP 1: AI Detective
    intent = extract_metadata_from_question(req.question)
    lender_filter = intent.get("lender_name")
    type_filter = intent.get("lender_type")

    # STEP 2: Filtering Logic
    must_conditions = []
    if lender_filter is not None:
        must_conditions.append(FieldCondition(key="lender_name", match=MatchValue(value=lender_filter)))
    if type_filter is not None:
        must_conditions.append(FieldCondition(key="lender_type", match=MatchValue(value=type_filter)))

    if len(must_conditions) > 0:
        query_filter = Filter(must=must_conditions)
    else:
        query_filter = None

    # STEP 3: Vector Search
    query_embedding = embed_text(req.question, is_query=True)
    
    search_result = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        query_filter=query_filter, 
        limit=req.top_k,
        with_payload=True
    )
    
    hits = sorted(search_result.points, key=lambda x: x.payload.get("order", 0))

    # Print all chunks used for the response
    logger.info("=" * 80)
    logger.info(f"CHUNKS RETRIEVED FOR QUESTION: {req.question}")
    logger.info("=" * 80)
    for i, hit in enumerate(hits, 1):
        p = hit.payload
        logger.info(f"\n--- CHUNK {i} (Score: {hit.score:.4f}) ---")
        logger.info(f"LENDER: {p.get('lender_name')}")
        logger.info(f"TYPE: {p.get('lender_type')}")
        logger.info(f"FILE: {p.get('file_name')}")
        logger.info(f"PAGE: {p.get('page_no')}")
        logger.info(f"TEXT:\n{p.get('text')}")
        logger.info("-" * 40)
    logger.info("=" * 80)

    # STEP 4: Context Building
    context_blocks = []
    for hit in hits:
        p = hit.payload
        block = (
            f"--- SOURCE ---\n"
            f"LENDER: {p.get('lender_name')}\n"
            f"PRODUCT TYPE: {p.get('lender_type')}\n"
            f"DOCUMENT: {p.get('file_name')}\n"
            f"PAGE: {p.get('page_no')}\n"
            f"CONTENT: {p.get('text')}\n"
            f"--- END SOURCE ---"
        )
        context_blocks.append(block)

    full_context = "\n\n".join(context_blocks)

    # STEP 5: Generation
    prompt = f"""
You are an expert Mortgage AI assistant. Answer the user's question using ONLY the provided context.

Context:
{full_context}

Question:
{req.question}

RULES FOR FORMATTING YOUR ANSWER:
1. Provide a direct, professional, structured answer.
2. Every time you cite information, you MUST wrap the citation in a Markdown link format exactly like this:
   [Lender: <lender_name> | Type: <lender_type> | Doc: <file_name> | Page: <page_no>](#)
3. Put the citation immediately after the sentence it supports.
4. If the answer is not in the context, say "This information isn't available in the document".
"""

    response = llm.generate_content(prompt)

    # STEP 6: Sources list
    response_sources = []
    for h in hits:
        response_sources.append({
            "lender_name": h.payload.get("lender_name"),
            "lender_type": h.payload.get("lender_type"),
            "file_name": h.payload.get("file_name"),
            "page_no": h.payload.get("page_no"),
            "score": h.score
        })

    # Save assistant message to MongoDB
    assistant_message_id = f"msg_{uuid.uuid4().hex[:12]}"
    messages_collection.insert_one({
        "message_id": assistant_message_id,
        "conversation_id": conversation_id,
        "role": "assistant",
        "content": response.text,
        "sources": response_sources,
        "timestamp": datetime.utcnow()
    })
    
    # Update conversation metadata
    conversations_collection.update_one(
        {"conversation_id": conversation_id},
        {
            "$set": {"updated_at": datetime.utcnow()},
            "$inc": {"message_count": 2}  # user + assistant
        }
    )

    return {
        "answer": response.text,
        "conversation_id": conversation_id,
        "intent_detected": intent, 
        "sources": response_sources
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)