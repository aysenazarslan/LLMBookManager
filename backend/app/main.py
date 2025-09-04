# backend/app/main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()
# Router'ı içe al
from app.api.routes import router as api_router






APP_TITLE = os.getenv("APP_TITLE", "LLM Turkish Book Assistant API")
APP_DESC = "Book loading, processing, embedding and RAG services"
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

app = FastAPI(title=APP_TITLE, description=APP_DESC, version=APP_VERSION)

# CORS
allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API router
app.include_router(api_router)

# Health endpoint
@app.get("/")
def root():
    return {"status": "ok", "service": APP_TITLE, "version": APP_VERSION}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=bool(int(os.getenv("UVICORN_RELOAD", "1")))
    )