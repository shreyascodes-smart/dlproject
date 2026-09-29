"""
FastAPI backend for Disaster Insights — Climate-Induced Natural Disaster Assessment.
Main entry point for the API server.
"""
from pathlib import Path
from dotenv import load_dotenv

# Load .env from backend directory
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.flood_routes import router as flood_router

app = FastAPI(
    title="Disaster Insights API",
    description="AI-Powered Climate-Induced Natural Disaster Assessment using Satellite Imagery",
    version="1.0.0",
)

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "flood-assessment-api",
        "version": "0.1.0",
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Flood Assessment API",
        "docs": "/docs",
        "health": "/health",
    }


# Register routers
app.include_router(flood_router)

