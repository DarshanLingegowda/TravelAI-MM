# TravelAI-MM – Multimodal Travel Intelligence

TravelAI-MM is a production-ready multimodal RAG system that ingests text, images, and audio, embeds them into a unified vector space, and generates grounded travel itineraries with citations.

## Features
- Multimodal ingestion (text, image, audio)
- Unified embeddings
- Vector search + RAG
- Evaluation & observability endpoints
- Azure-ready architecture

## Run
pip install -r requirements.txt  
uvicorn app.main:app --reload
