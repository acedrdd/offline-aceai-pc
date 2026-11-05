import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
import app as ace

client = TestClient(ace.app)

def test_health():
    assert client.get("/api/health").json()["status"] == "ok"

def test_hardware():
    data = client.get("/api/hardware").json()
    assert data["hardware"]["ram_gb"] > 0
    assert data["settings"]["n_ctx"] in (2048, 4096)

def test_models_empty_list_shape():
    assert "models" in client.get("/api/models").json()

def test_chat_without_model_errors():
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400

def test_missing_model_404():
    assert client.post("/api/models/nope.gguf/load").status_code in (404, 503)

def test_image_video_honestly_unimplemented():
    assert client.post("/api/image").status_code == 501

def test_index_page():
    assert client.get("/").status_code == 200
