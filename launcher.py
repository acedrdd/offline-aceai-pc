"""Ace AI launcher: starts the local server and opens the browser. Entry point for the .exe."""
import socket
import threading
import time
import webbrowser

import uvicorn

from app import app

HOST, PORT = "127.0.0.1", 8000

def port_in_use() -> bool:
    with socket.socket() as s:
        return s.connect_ex((HOST, PORT)) == 0

def open_browser_when_ready():
    for _ in range(100):
        if port_in_use():
            webbrowser.open(f"http://localhost:{PORT}")
            return
        time.sleep(0.2)

if __name__ == "__main__":
    if port_in_use():  # already running: just open it
        webbrowser.open(f"http://localhost:{PORT}")
    else:
        print(f"Ace AI running at http://localhost:{PORT}  (close this window to stop)")
        print("Put .gguf models in the 'models' folder next to this file.")
        threading.Thread(target=open_browser_when_ready, daemon=True).start()
        uvicorn.run(app, host=HOST, port=PORT)
