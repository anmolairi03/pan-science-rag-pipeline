from fastapi import FastAPI
from app.api import router as api_router

app = FastAPI(
    title="RAG API",
    description="Upload documents and query them using LLMs with RAG.",
    version="1.0.0"
)

# Include API routes
app.include_router(api_router)
