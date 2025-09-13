#!/usr/bin/env python3
import os
import re
import asyncio
from typing import Optional, List, Tuple

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cohere  # pip install cohere

from interactive_audio_processor import InteractiveAudioProcessor

# ---------------- Behavior / model ----------------
PREAMBLE = (
    "You are VibeVideo.AI, a concise expert assistant for audio and video. "
    "Answer clearly and briefly. If the user asks for an edit, confirm what will be done "
    "in plain language and provide any helpful tips. Do not expose internal commands."
)
MODEL = os.getenv("COHERE_MODEL", "command-r")
COHERE_KEY = os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY")

# Natural-language → command mapping
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
    # power-user tokens
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
    t = (text or "").strip().lower()
    return any(v in t for v in INTENT_VERBS)

def _extract_command_from_user(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    for pattern, token in PHRASE_TO_CMD:
        if re.search(pattern, t):
            return token
    return None

# ---------------- Flask app ----------------
app = Flask(__name__)
# Heroku router limit is ~32MB; keep uploads under that
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
CORS(app)  # keep permissive during dev; restrict to your domain in prod

CLEANVOICE_KEY = os.getenv("CLEANVOICE_API_KEY", "FJB8s8nbmY9UQcfeXFeB6tqJmjwDUkKN")
iap = InteractiveAudioProcessor(CLEANVOICE_KEY)

co = cohere.Client(COHERE_KEY) if COHERE_KEY else None

@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "service": "VibeVideo Flask API"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/chat", methods=["POST"])
def chat():
    if not co:
        return jsonify({"error": "COHERE_API_KEY / CO_API_KEY not set"}), 500
    message = request.form.get("message", "").strip()
    if not message:
        return jsonify({"error": "Missing 'message'"}), 400

    # Try new Cohere signature first; fallback to legacy if needed
    try:
        resp = co.chat(
            model=MODEL,
            messages=[{"role": "user", "content": message}],
            preamble=PREAMBLE,
            temperature=0.3,
        )
        text = getattr(resp, "text", None) or getattr(getattr(resp, "message", None), "content", "")
        return jsonify({"message": (text or "").strip()})
    except TypeError:
        resp = co.chat(
            model=MODEL,
            message=message,
            chat_history=[],
            preamble=PREAMBLE,
            temperature=0.3,
        )
        text = getattr(resp, "text", None) or getattr(getattr(resp, "message", None), "content", "")
        return jsonify({"message": (text or "").strip()})

@app.route("/process", methods=["POST"])
def process():
    """
    Multipart form:
      file:   uploaded audio/video
      message: natural language instruction (e.g., 'remove background noise')
      command (optional): exact token like 'rm bg'
    Returns: processed file as attachment
    """
    uploaded = request.files.get("file")
    message = request.form.get("message", "")
    command = request.form.get("command", None)

    if not uploaded or uploaded.filename == "":
        return jsonify({"detail": "No file uploaded"}), 400

    # Decide command unless explicitly provided
    if not command:
        if not _user_has_execute_intent(message):
            return jsonify({"detail": "No executable intent detected. Try 'remove background noise'."}), 400
        command = _extract_command_from_user(message)
        if not command:
            return jsonify({"detail": "Couldn't map your request to a known command."}), 400

    if command not in iap.function_map:
        return jsonify({"detail": f"Unknown command '{command}'."}), 400

    # Save upload to /tmp in chunks (Heroku-safe)
    os.makedirs("/tmp", exist_ok=True)
    _, ext = os.path.splitext(uploaded.filename or "input.bin")
    ext = ext or ".m4a"
    local_in = os.path.join("/tmp", f"input{ext}")
    with open(local_in, "wb") as out:
        while True:
            chunk = uploaded.stream.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    # Run processing and CAPTURE the returned output path
    iap.set_input_file(local_in)
    try:
        out_path = asyncio.run(iap.process_audio_file(command))
    finally:
        iap.set_input_file(None)

    # If the processor didn't return a path, error out
    if not out_path:
        return jsonify({"detail": "Processing failed (no output path returned)."}), 500

    # Resolve relative vs absolute path; check common locations
    candidates = []
    if os.path.isabs(out_path):
        candidates.append(out_path)
    else:
        # try CWD (where InteractiveAudioProcessor likely saved)
        candidates.append(os.path.join(os.getcwd(), out_path))
        # also try /tmp just in case
        candidates.append(os.path.join("/tmp", out_path))

    # final fallback: pattern search for guessed name
    stem = os.path.splitext(os.path.basename(local_in))[0]  # "input"
    guess_name = f"{stem}-{command.replace(' ', '-')}{ext}"
    candidates.append(os.path.join(os.getcwd(), guess_name))
    candidates.append(os.path.join("/tmp", guess_name))

    # pick the first existing file
    existing = next((p for p in candidates if os.path.exists(p)), None)
    if not existing:
        return jsonify({"detail": "Processing finished but output file not found."}), 500

    # Stream as download
    return send_file(
        existing,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=os.path.basename(existing)
    )

# ----- Local dev entry (Heroku uses Procfile/Gunicorn) -----
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)

