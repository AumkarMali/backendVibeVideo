#!/usr/bin/env python3
import os
import re
import asyncio
from typing import Optional, List, Tuple

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
try:
    import cohere  # pip install cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    print("Warning: cohere module not available. Chat functionality will be disabled.")

try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("Warning: pymongo module not available. Authentication functionality will be disabled.")

from interactive_audio_processor import InteractiveAudioProcessor
from file_merger import merge_files

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
    (r"\b(merge|combine|join|concatenate|splice).*(audio|video|file|files|clips?)\b", "merge"),
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
    (r"\bmerge\b", "merge"),
]
INTENT_VERBS = [
    "run", "apply", "execute", "perform", "process", "start", "begin", "do",
    "remove", "reduce", "normalize", "enhance", "transcribe", "preserve", "clean", "denoise",
    "merge", "combine", "join", "concatenate", "splice"
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

co = cohere.Client(COHERE_KEY) if COHERE_KEY and COHERE_AVAILABLE else None

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://chatapp_user:AM20060305!_ilovesushi@cluster0.rw6klmv.mongodb.net/chatapp?retryWrites=true&w=majority&appName=Cluster0")
DB_NAME = os.getenv("DB_NAME", "chatapp")

# Initialize MongoDB client
if MONGODB_AVAILABLE:
    try:
        mongo_client = MongoClient(MONGODB_URI)
        mongo_db = mongo_client[DB_NAME]
        # Test connection
        mongo_client.admin.command('ping')
        print("✅ MongoDB connection successful")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        MONGODB_AVAILABLE = False
        mongo_client = None
        mongo_db = None
else:
    mongo_client = None
    mongo_db = None

# Helper functions for MongoDB operations
def get_user_chats(username: str, limit: int = 50):
    """Fetch chats for a given username, sorted by updatedAt desc."""
    if not MONGODB_AVAILABLE or not mongo_db:
        return []
    
    try:
        chats = list(
            mongo_db.chats
            .find({"username": username})
            .sort("updatedAt", -1)  # -1 for descending order
            .limit(limit)
        )
        return chats
    except Exception as e:
        print(f"Error fetching chats for {username}: {e}")
        return []

def get_user_library_items(username: str, limit: int = 100):
    """Fetch library items for a given username, sorted by updatedAt desc."""
    if not MONGODB_AVAILABLE or not mongo_db:
        return []
    
    try:
        library_items = list(
            mongo_db.library_items
            .find({"username": username})
            .sort("updatedAt", -1)  # -1 for descending order
            .limit(limit)
        )
        return library_items
    except Exception as e:
        print(f"Error fetching library items for {username}: {e}")
        return []

@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "service": "VibeVideo Flask API"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/login", methods=["POST"])
def login():
    """
    Authenticate user against MongoDB database.
    
    Expected JSON payload:
    {
        "username": "AumkarM",
        "password": "MySecret123!"
    }
    
    Returns:
    - 200: Login successful with user info
    - 401: Invalid credentials
    - 500: Server error
    """
    if not MONGODB_AVAILABLE or not mongo_db:
        return jsonify({"error": "Database connection not available"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()
        
        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400
        
        # Query MongoDB for user
        user = mongo_db.users.find_one({"username": username})
        
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Verify password (in production, you should hash passwords)
        if user.get("password") != password:
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Remove password from response for security
        user_response = {
            "username": user.get("username"),
            "email": user.get("email"),
            "createdAt": user.get("createdAt")
        }
        
        # Fetch user's chats and library items
        chats = get_user_chats(username)
        library_items = get_user_library_items(username)
        
        return jsonify({
            "message": "Login successful",
            "user": user_response,
            "chats": chats,
            "library_items": library_items
        }), 200
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/chats", methods=["GET"])
def get_chats():
    """
    Get chats for a specific user.
    
    Query parameters:
    - username: The username to fetch chats for
    - limit: Maximum number of chats to return (default: 50)
    
    Returns:
    - 200: List of chats for the user
    - 400: Missing username parameter
    - 500: Server error
    """
    if not MONGODB_AVAILABLE or not mongo_db:
        return jsonify({"error": "Database connection not available"}), 500
    
    username = request.args.get("username")
    if not username:
        return jsonify({"error": "Username parameter is required"}), 400
    
    try:
        limit = int(request.args.get("limit", 50))
        chats = get_user_chats(username, limit)
        
        return jsonify({
            "username": username,
            "chats": chats,
            "count": len(chats)
        }), 200
        
    except ValueError:
        return jsonify({"error": "Invalid limit parameter"}), 400
    except Exception as e:
        print(f"Get chats error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/library", methods=["GET"])
def get_library():
    """
    Get library items for a specific user.
    
    Query parameters:
    - username: The username to fetch library items for
    - limit: Maximum number of items to return (default: 100)
    
    Returns:
    - 200: List of library items for the user
    - 400: Missing username parameter
    - 500: Server error
    """
    if not MONGODB_AVAILABLE or not mongo_db:
        return jsonify({"error": "Database connection not available"}), 500
    
    username = request.args.get("username")
    if not username:
        return jsonify({"error": "Username parameter is required"}), 400
    
    try:
        limit = int(request.args.get("limit", 100))
        library_items = get_user_library_items(username, limit)
        
        return jsonify({
            "username": username,
            "library_items": library_items,
            "count": len(library_items)
        }), 200
        
    except ValueError:
        return jsonify({"error": "Invalid limit parameter"}), 400
    except Exception as e:
        print(f"Get library error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    if not co:
        return jsonify({"error": "COHERE_API_KEY / CO_API_KEY not set or cohere module not available"}), 500
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

@app.route("/merge", methods=["POST"])
def merge():
    """
    Multipart form:
      files: multiple uploaded audio/video files (same type)
      message: natural language instruction (e.g., 'merge these audio files')
      order (optional): comma-separated order of files (e.g., '0,2,1,3')
    Returns: merged file as attachment
    """
    uploaded_files = request.files.getlist("files")
    message = request.form.get("message", "")
    order = request.form.get("order", None)

    if not uploaded_files or len(uploaded_files) < 2:
        return jsonify({"detail": "Need at least 2 files to merge"}), 400

    # Filter out empty files
    uploaded_files = [f for f in uploaded_files if f.filename and f.filename.strip()]
    
    if len(uploaded_files) < 2:
        return jsonify({"detail": "Need at least 2 valid files to merge"}), 400

    # Check if user wants to merge (if message is provided)
    if message and not _user_has_execute_intent(message):
        return jsonify({"detail": "No executable intent detected. Try 'merge these audio files'."}), 400

    # Save uploads to /tmp in chunks (Heroku-safe)
    os.makedirs("/tmp", exist_ok=True)
    local_files = []
    
    try:
        for i, uploaded in enumerate(uploaded_files):
            _, ext = os.path.splitext(uploaded.filename or f"input_{i}.bin")
            ext = ext or ".m4a"
            local_path = os.path.join("/tmp", f"merge_input_{i}{ext}")
            
            with open(local_path, "wb") as out:
                while True:
                    chunk = uploaded.stream.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            
            local_files.append(local_path)

        # Handle custom ordering if provided
        if order:
            try:
                order_indices = [int(x.strip()) for x in order.split(',')]
                if len(order_indices) != len(local_files):
                    return jsonify({"detail": f"Order list length ({len(order_indices)}) doesn't match number of files ({len(local_files)})"}), 400
                if set(order_indices) != set(range(len(local_files))):
                    return jsonify({"detail": "Order indices must be unique and cover all files (0-based)"}), 400
                local_files = [local_files[i] for i in order_indices]
            except (ValueError, IndexError) as e:
                return jsonify({"detail": f"Invalid order format: {e}"}), 400

        # Generate output filename
        first_ext = os.path.splitext(local_files[0])[1]
        output_filename = f"merged{first_ext}"
        output_path = os.path.join("/tmp", output_filename)

        # Merge the files
        merged_path = merge_files(local_files, output_path)
        
        # Return the merged file
        return send_file(
            merged_path,
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name=output_filename
        )
        
    except Exception as e:
        return jsonify({"detail": f"Merge failed: {str(e)}"}), 500
    finally:
        # Clean up temporary files
        for local_file in local_files:
            try:
                if os.path.exists(local_file):
                    os.remove(local_file)
            except:
                pass

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
