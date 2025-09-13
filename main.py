#!/usr/bin/env python3
import os
import re
import io
import cohere
import uvicorn
import asyncio
from typing import Optional, List, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from interactive_audio_processor import InteractiveAudioProcessor

# ---------- Behavior / Cohere ----------
PREAMBLE = (
    "You are VibeVideo.AI, a concise expert assistant for audio and video. "
    "Answer clearly and briefly. If the user asks for an edit, confirm what will be done "
    "in plain language and provide any helpful tips. Do not expose internal commands."
)

MODEL = os.getenv("COHERE_MODEL", "command-r")
COHERE_KEY = os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY")

# ---------- Phrase → command mapping ----------
PHRASE_TO_CMD: List[Tuple[str, str]] = [
    (r"\b(remove|reduce|clean|denoise).*(background\s+noise|noise|hums?|buzz|hiss)\b", "rm bg"),
    (r"\b(remove|trim|cut).*(long\s+)?silence(s)?\b", "rm silence"),
    (r"\b(remove).*(stutter|stutters|stuttering)\b", "rm stutter"),
    (r"\b(remove).*(filler|um|uh|like|you know|sort of|kinda)\b", "rm filler"),
    (r"\b(remove).*(mouth\s*(sound|click|smack|noise)s?)\b", "rm mouth"),
    (r"\b(remove).*(hesitation|hesitations|hesitant)\b", "rm hesitation"),
    (r"\b(reduce|remove).*(breath|breathing)\b", "rm breath"),
    (r"\bnormalize(d|)\b|\b(level match|level-match)\b", "normalize"),
    (r"\b(ai\s*enhance|enhance.*(with\s*ai)?|ai\s*cleanup)\b", "ai enhance"),
    (r"\b(preserve|keep).*(music)\b", "preserve music"),
    (r"\btranscribe|transcription|speech\s*to\s*text\b", "transcribe"),
    (r"\b(apply|do|run|perform|process).*\b(all|comprehensive|everything)\b", "comprehensive"),
    # power users typing exact tokens:
    (r"\brm bg\b", "rm bg"),
    (r"\brm silence\b", "rm silence"),
    (r"\brm stutter\b", "rm stutter"),
    (r"\brm filler\b", "rm filler"),
    (r"\brm mouth\b", "rm mouth"),
    (r"\brm hesitation\b", "rm hesitation"),
    (r"\brm breath\b", "rm breath"),
    (r"\bnormalize\b", "normalize"),
    (r"\bai enhance\b", "ai enhance"),
    (r"\bpreserve music\b", "preserve music"),
    (r"\btranscribe\b", "transcribe"),
    (r"\bcomprehensive\b", "comprehensive"),
]

INTENT_VERBS = [
    "run", "apply", "execute", "perform", "process", "start", "begin", "do",
    "remove", "reduce", "normalize", "enhance", "transcribe", "preserve", "clean", "denoise"
]

def _user_has_execute_intent(text: str) -> bool:
    t = text.strip().lower()
    return any(v in t for v in INTENT_VERBS)

def _extract_command_from_user(text: str) -> Optional[str]:
    t = text.strip().lower()
    for pattern, token in PHRASE_TO_CMD:
        if re.search(pattern, t):
            return token
    return None

# ---------- App ----------
app = FastAPI(title="VibeVideo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down to your domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLEANVOICE_KEY = os.getenv("CLEANVOICE_API_KEY", "FJB8s8nbmY9UQcfeXFeB6tqJmjwDUkKN")
iap = InteractiveAudioProcessor(CLEANVOICE_KEY)  # uses your existing processing pipeline  :contentReference[oaicite:4]{index=4}

# Cohere client (optional; only needed for /chat)
co = cohere.Client(COHERE_KEY) if COHERE_KEY else None

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(message: str = Form(...)):
    """Simple chatbot endpoint: returns assistant text."""
    if not COHERE_KEY:
        raise HTTPException(500, "COHERE_API_KEY / CO_API_KEY not set")

    # try new SDK signature first; fallback to legacy if needed
    try:
        resp = co.chat(
            model=MODEL,
            messages=[{"role": "user", "content": message}],
            preamble=PREAMBLE,
            temperature=0.3,
        )
        text = getattr(resp, "text", None) or getattr(getattr(resp, "message", None), "content", "")
        return {"message": (text or "").strip()}
    except TypeError:
        # legacy signature
        resp = co.chat(
            model=MODEL,
            message=message,
            chat_history=[],
            preamble=PREAMBLE,
            temperature=0.3,
        )
        text = getattr(resp, "text", None) or getattr(getattr(resp, "message", None), "content", "")
        return {"message": (text or "").strip()}

@app.post("/process")
async def process(
    file: UploadFile = File(...),
    message: str = Form(...),
    command: Optional[str] = Form(None)  # optional override if you want to force a specific token
):
    """
    Accept a file + user instruction, run the mapped command via InteractiveAudioProcessor,
    and stream the processed file back to the frontend.
    """
    # 1) Decide the command from user message (unless forced)
    if command is None:
        if not _user_has_execute_intent(message):
            return JSONResponse(
                status_code=400,
                content={"detail": "No executable intent detected. Say things like 'remove background noise' or 'normalize audio'."}
            )
        command = _extract_command_from_user(message)
        if not command:
            return JSONResponse(
                status_code=400,
                content={"detail": "Couldn't map your request to a known command."}
            )

    # Validate command available
    if command not in iap.function_map:
        return JSONResponse(status_code=400, content={"detail": f"Unknown command '{command}'."})

    # 2) Save upload to /tmp
    os.makedirs("/tmp", exist_ok=True)
    ext = os.path.splitext(file.filename or "input.bin")[1] or ".m4a"
    local_in = os.path.join("/tmp", f"input{ext}")
    with open(local_in, "wb") as f:
        f.write(await file.read())

    # 3) Run processing
    iap.set_input_file(local_in)  # <— tell the processor to use the uploaded file next
    try:
        await iap.process_audio_file(command)
    finally:
        # clear override so future calls don't reuse this file by accident
        iap.set_input_file(None)  # type: ignore

    # 4) Find processed output (the class names output as input-stem + "-<command>" + ext)  :contentReference[oaicite:5]{index=5}
    stem = os.path.splitext(os.path.basename(local_in))[0]
    guess_name = f"{stem}-{command.replace(' ', '-')}{ext}"
    local_out = os.path.join("/tmp", guess_name)
    if not os.path.exists(local_out):
        # fallback: search /tmp for any file starting with 'input-' and same ext
        candidates = [p for p in os.listdir("/tmp") if p.startswith("input-") and p.endswith(ext)]
        if not candidates:
            raise HTTPException(500, "Processing finished but output file not found.")
        candidates.sort(key=lambda n: os.path.getmtime(os.path.join("/tmp", n)))
        local_out = os.path.join("/tmp", candidates[-1])

    # 5) Stream file back to frontend
    def _iterfile():
        with open(local_out, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk

    download_name = os.path.basename(local_out)
    return StreamingResponse(
        _iterfile(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'}
    )

if __name__ == "__main__":
    # Local run: uvicorn main:app --reload --port 8080
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=True)
