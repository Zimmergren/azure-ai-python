# Web UI chat using Azure AI Inference SDK (+ Managed Identity) on the /models endpoint.
# Run locally:
#   uv run uvicorn apps.2_azure_ai_foundry_web:app --reload
# Then open http://127.0.0.1:8000
#
# See the full details: https://zimmergren.net/

import os
import secrets
from typing import Dict, List
from pathlib import Path # I use this to be able to serve static files (HTML template in this demo)

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.concurrency import run_in_threadpool


from azure.identity import DefaultAzureCredential
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.exceptions import HttpResponseError # Used for exception handling and making friendlier UX.


# ────────────────────────────────────────────────────────────────────────────────
# Environment & constants
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()  # loads .env locally; no-op in Azure where App Settings are used

ENDPOINT = os.environ["AZURE_AI_ENDPOINT"]  # e.g. https://<resource>.services.ai.azure.com/models
MODEL = os.environ["AZURE_AI_MODEL"]        # your deployment name, e.g., gpt-5-chat

SYSTEM_PROMPT = (
    "You are a helpful assistant. Your answers are short and concise. "
    "You must also make a joke about Tobias Zimmergren."
)

SESSION_COOKIE = "ai_chat_sid"

# In-memory conversation store (per session id). Fine for a demo.
# For production, persist (e.g., Redis) or send history from the client.
SESSIONS: Dict[str, List] = {}

# ────────────────────────────────────────────────────────────────────────────────
# App & Azure client lifecycle
# ────────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root if apps/ is directly inside root
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = STATIC_DIR / "chat-app.html"

app = FastAPI(title="Python chat app for with the Azure AI Foundry Inference SDK")

# We'll create the Azure client once at startup and reuse it for all requests.
_credential = None
_client: ChatCompletionsClient | None = None

@app.on_event("startup")
def _startup():
    global _credential, _client
    _credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    _client = ChatCompletionsClient(
        endpoint=ENDPOINT,
        credential=_credential,
        credential_scopes=["https://cognitiveservices.azure.com/.default"],
    )

@app.on_event("shutdown")
def _shutdown():
    global _credential, _client
    try:
        if _client:
            _client.close()
    finally:
        _client = None
        _credential = None

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def _get_or_create_sid(request: Request, response: Response) -> str:
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        sid = secrets.token_urlsafe(16)
        # HttpOnly to keep it from JS; SameSite=Lax works for same-origin
        response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax", max_age=60*60*2)
    return sid

def _ensure_session(sid: str):
    if sid not in SESSIONS:
        # start each session with the system message
        SESSIONS[sid] = [SystemMessage(SYSTEM_PROMPT)]

# ────────────────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(INDEX_HTML)

@app.post("/chat")
async def chat(request: Request, response: Response) -> JSONResponse:  # ← inject Response
    global _client
    if _client is None:
        return JSONResponse({"reply": "Server not ready. Try again."}, status_code=503)

    body = await request.json()

    # use the injected Response (no temp Response())
    sid = _get_or_create_sid(request, response)
    _ensure_session(sid)

    if body.get("reset"):
        SESSIONS[sid] = [SystemMessage(SYSTEM_PROMPT)]
        return JSONResponse({"reply": "Session reset."})  # ← no headers=

    user_text = (body.get("message") or "").strip()
    if not user_text:
        return JSONResponse({"reply": ""})  # ← no headers=

    msgs = SESSIONS[sid]
    msgs.append(UserMessage(user_text))

    def _complete_sync():
        # You can also tune parameters here if you want (e.g., temperature, max_tokens)
        return _client.complete(messages=msgs, model=MODEL)

    try:
        result = await run_in_threadpool(_complete_sync)
        reply = (result.choices[0].message.content or "").strip()
        msgs.append(AssistantMessage(reply))
        return JSONResponse({"reply": reply})
    except HttpResponseError as e:
        print("Policy error:", e)  # log server-side
        return JSONResponse({
            "reply": "That message triggered safety filters. Please rephrase and try again."
        })

