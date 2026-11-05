"""Ace AI - localhost backend (Phases 3, 4, 7: chat engine, model manager, hardware)."""
import json
import os
import platform
import sys
from pathlib import Path
from typing import List, Optional

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

FROZEN = getattr(sys, "frozen", False)
# Bundled resources (static files) vs. user data (models) live in different places in an .exe
RES_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
DATA_DIR = Path(sys.executable).parent if FROZEN else Path(__file__).parent
MODELS_DIR = Path(os.environ.get("ACEAI_MODELS_DIR", DATA_DIR / "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Ace AI")

# ---------- Hardware & performance ----------

def detect_hardware() -> dict:
    ram_gb = psutil.virtual_memory().total / 1024**3
    return {
        "os": platform.system(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_gb": round(ram_gb, 1),
        "ram_available_gb": round(psutil.virtual_memory().available / 1024**3, 1),
        "low_memory_mode": ram_gb <= 9,  # 8GB machines report slightly under 8
    }

def perf_settings() -> dict:
    hw = detect_hardware()
    threads = max(1, (hw["cpu_threads"] or 2) - 1)
    if hw["low_memory_mode"]:  # 8GB optimization mode
        return {"n_ctx": 2048, "n_batch": 128, "n_threads": threads, "use_mmap": True, "mode": "8gb"}
    return {"n_ctx": 4096, "n_batch": 512, "n_threads": threads, "use_mmap": True, "mode": "standard"}

# ---------- Model manager ----------

def list_models() -> List[dict]:
    return [
        {"name": p.name, "size_mb": round(p.stat().st_size / 1024**2, 1)}
        for p in sorted(MODELS_DIR.glob("*.gguf"))
    ]

def safe_model_path(name: str) -> Path:
    path = (MODELS_DIR / Path(name).name).resolve()
    if path.suffix != ".gguf" or not path.is_file():
        raise HTTPException(404, "Model not found")
    return path

class Engine:
    """Lazy-loaded llama.cpp model; unloadable to free RAM."""
    def __init__(self):
        self.llm = None
        self.name: Optional[str] = None

    def load(self, name: str):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise HTTPException(503, "llama-cpp-python not installed. Run: pip install -r requirements-llm.txt")
        path = safe_model_path(name)
        self.unload()
        s = perf_settings()
        self.llm = Llama(model_path=str(path), n_ctx=s["n_ctx"], n_batch=s["n_batch"],
                         n_threads=s["n_threads"], use_mmap=s["use_mmap"], verbose=False)
        self.name = name

    def unload(self):
        self.llm = None
        self.name = None
        import gc; gc.collect()

engine = Engine()

# ---------- API ----------

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7

@app.get("/api/health")
def health():
    return {"status": "ok", "loaded_model": engine.name}

@app.get("/api/hardware")
def hardware():
    return {"hardware": detect_hardware(), "settings": perf_settings()}

@app.get("/api/models")
def models():
    return {"models": list_models(), "loaded": engine.name}

@app.post("/api/models/{name}/load")
def load_model(name: str):
    engine.load(name)
    return {"loaded": engine.name}

@app.post("/api/models/unload")
def unload_model():
    engine.unload()
    return {"loaded": None}

@app.delete("/api/models/{name}")
def delete_model(name: str):
    path = safe_model_path(name)
    if engine.name == path.name:
        engine.unload()
    path.unlink()
    return {"deleted": path.name}

@app.post("/api/chat")
def chat(req: ChatRequest):
    if req.model and req.model != engine.name:
        engine.load(req.model)
    if engine.llm is None:
        raise HTTPException(400, "No model loaded. Add a .gguf file to the models folder and load it.")

    def stream():
        for chunk in engine.llm.create_chat_completion(
            messages=[m.model_dump() for m in req.messages],
            max_tokens=req.max_tokens, temperature=req.temperature, stream=True,
        ):
            token = chunk["choices"][0]["delta"].get("content")
            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.post("/api/image")
@app.post("/api/video")
def not_implemented():
    raise HTTPException(501, "Not implemented yet (planned for Phases 5 and 6).")

# Frontend (Phase 2): index.html sits next to app.py
@app.get("/", include_in_schema=False)
def index():
    page = RES_DIR / "index.html"
    if not page.exists():
        raise HTTPException(404, "index.html missing")
    return FileResponse(page)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
