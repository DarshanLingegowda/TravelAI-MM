"""
TravelAI-MM: Production-Ready Multimodal RAG Application
Author: Darshan Linge Gowda
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager
from typing import Optional, List, Dict
import numpy as np
import logging
from datetime import datetime
import uuid
from io import BytesIO
from PIL import Image
import soundfile as sf
import threading

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travelai-mm")

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    APP_NAME = "TravelAI-MM"
    VERSION = "1.0.0"
    MAX_FILE_SIZE = 10 * 1024 * 1024
    ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
    ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/mp3", "audio/mpeg"}
    EMBEDDING_DIM = 512
    TOP_K = 5
    MIN_SIMILARITY = 0.3

config = Config()

# ============================================================================
# ENCODERS (Azure swap-ready)
# ============================================================================

class BaseEncoder:
    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, data) -> np.ndarray:
        raise NotImplementedError


class TextEncoder(BaseEncoder):
    def encode(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Empty text input")
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.random(self.dim, dtype=np.float32)


class ImageEncoder(BaseEncoder):
    def encode(self, image_bytes: bytes) -> np.ndarray:
        try:
            img = Image.open(BytesIO(image_bytes))
            img.verify()
            rng = np.random.default_rng(len(image_bytes))
            return rng.random(self.dim, dtype=np.float32)
        except Exception:
            raise ValueError("Invalid image data")


class AudioEncoder(BaseEncoder):
    def encode(self, audio_bytes: bytes) -> np.ndarray:
        try:
            sf.read(BytesIO(audio_bytes))
            rng = np.random.default_rng(len(audio_bytes))
            return rng.random(self.dim, dtype=np.float32)
        except Exception:
            raise ValueError("Invalid audio data")

# ============================================================================
# VECTOR STORE
# ============================================================================

class Document:
    def __init__(self, vector: np.ndarray, metadata: Dict):
        self.id = str(uuid.uuid4())
        self.vector = vector
        self.metadata = metadata
        self.created_at = datetime.utcnow()


class VectorStore:
    def __init__(self):
        self.docs: Dict[str, Document] = {}
        self.vectors = []
        self.ids = []
        self.lock = threading.Lock()

    def add(self, vector: np.ndarray, metadata: Dict) -> str:
        with self.lock:
            doc = Document(vector, metadata)
            self.docs[doc.id] = doc
            self.vectors.append(vector)
            self.ids.append(doc.id)
            return doc.id

    def search(self, query: np.ndarray, k: int, min_score: float):
        if not self.vectors:
            return []

        vectors = np.array(self.vectors)
        q = query / (np.linalg.norm(query) + 1e-8)
        v = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

        scores = np.dot(v, q)
        idxs = np.argsort(scores)[::-1][:k]

        results = []
        for i in idxs:
            if scores[i] >= min_score:
                doc = self.docs[self.ids[i]]
                results.append({
                    "id": doc.id,
                    "score": float(scores[i]),
                    "metadata": doc.metadata,
                    "created_at": doc.created_at.isoformat()
                })
        return results

    def stats(self):
        by_type = {}
        for d in self.docs.values():
            t = d.metadata.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        return {"total_documents": len(self.docs), "by_type": by_type}

# ============================================================================
# APP STATE
# ============================================================================

class AppState:
    text_encoder: TextEncoder
    image_encoder: ImageEncoder
    audio_encoder: AudioEncoder
    store: VectorStore

state = AppState()

# ============================================================================
# SCHEMAS
# ============================================================================

class TextIngest(BaseModel):
    text: str
    metadata: Optional[Dict] = {}

    @validator("text")
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v


class PlanRequest(BaseModel):
    query: str
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    include_types: Optional[List[str]] = None


# ============================================================================
# LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting TravelAI-MM")
    state.text_encoder = TextEncoder(config.EMBEDDING_DIM)
    state.image_encoder = ImageEncoder(config.EMBEDDING_DIM)
    state.audio_encoder = AudioEncoder(config.EMBEDDING_DIM)
    state.store = VectorStore()
    yield
    logger.info("Shutting down TravelAI-MM")

# ============================================================================
# APP INIT
# ============================================================================

app = FastAPI(
    title=config.APP_NAME,
    version=config.VERSION,
    description="Multimodal Travel Intelligence with RAG",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve landing page
app.mount("/", StaticFiles(directory="web", html=True), name="web")

# ============================================================================
# HELPERS
# ============================================================================

async def validate_upload(file: UploadFile, allowed: set):
    if file.content_type not in allowed:
        raise HTTPException(400, "Invalid file type")
    data = await file.read()
    if len(data) > config.MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
    return data


def generate_itinerary(query: str, citations: List[Dict]) -> str:
    modalities = {c["metadata"]["type"] for c in citations}
    text = f"Travel plan for: {query}\n\n"
    if "image" in modalities:
        text += "• Visual preferences detected\n"
    if "audio" in modalities:
        text += "• Audio preferences considered\n"
    text += "\nDay 1–2: Relaxed exploration\nDay 3–4: Local culture & food\nDay 5: Leisure and return"
    return text

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": config.VERSION,
        "stats": state.store.stats()
    }


@app.post("/ingest/text")
def ingest_text(req: TextIngest):
    vec = state.text_encoder.encode(req.text)
    doc_id = state.store.add(vec, {"type": "text", **req.metadata})
    return {"status": "ok", "document_id": doc_id}


@app.post("/ingest/image")
async def ingest_image(file: UploadFile = File(...)):
    data = await validate_upload(file, config.ALLOWED_IMAGE_TYPES)
    vec = state.image_encoder.encode(data)
    doc_id = state.store.add(vec, {"type": "image", "filename": file.filename})
    return {"status": "ok", "document_id": doc_id}


@app.post("/ingest/audio")
async def ingest_audio(file: UploadFile = File(...)):
    data = await validate_upload(file, config.ALLOWED_AUDIO_TYPES)
    vec = state.audio_encoder.encode(data)
    doc_id = state.store.add(vec, {"type": "audio", "filename": file.filename})
    return {"status": "ok", "document_id": doc_id}


@app.post("/plan")
def plan(req: PlanRequest):
    qvec = state.text_encoder.encode(req.query)
    results = state.store.search(qvec, req.top_k, config.MIN_SIMILARITY)

    if req.include_types:
        results = [r for r in results if r["metadata"]["type"] in req.include_types]

    itinerary = generate_itinerary(req.query, results)

    breakdown = {}
    for r in results:
        t = r["metadata"]["type"]
        breakdown[t] = breakdown.get(t, 0) + 1

    return {
        "query": req.query,
        "itinerary": itinerary,
        "citations": results,
        "num_sources": len(results),
        "modality_breakdown": breakdown
    }


@app.get("/stats")
def stats():
    return state.store.stats()


@app.delete("/reset")
def reset():
    state.store = VectorStore()
    return {"status": "reset"}

