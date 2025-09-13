#!/usr/bin/env python3
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Simple Flask app to test routing
app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "message": "Root endpoint working"}), 200


@app.route("/test", methods=["GET"])
def test():
    return jsonify({"message": "Test endpoint working"}), 200


@app.route("/chats", methods=["GET"])
def get_chats():
    username = request.args.get("username", "unknown")
    return jsonify({
        "message": "Chats endpoint working",
        "username": username,
        "chats": [],
        "count": 0
    }), 200


@app.route("/debug/routes", methods=["GET"])
def debug_routes():
    """Show all registered routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "rule": str(rule)
        })

    return jsonify({
        "message": "All registered routes",
        "routes": routes,
        "total_routes": len(routes)
    }), 200


# ===================== VibeVideo Mongo + Debug add-on (APPEND-ONLY) =====================
# This block registers new routes without modifying any existing lines above.

# Lazy import so we don't break your app if pymongo isn't installed
try:
    from pymongo import MongoClient as _VV_MongoClient
except Exception:
    _VV_MongoClient = None

import os
from flask import request, jsonify

# Read env (doesn't override any of your existing config)
_VV_MONGODB_URI = os.getenv("MONGODB_URI")
_VV_DB_NAME = os.getenv("DB_NAME", "chatapp")

# Best-effort connection; all routes handle the "no DB" case gracefully
_vv_client = _VV_MongoClient(_VV_MONGODB_URI) if (_VV_MONGODB_URI and _VV_MongoClient) else None
_vv_db = _vv_client[_VV_DB_NAME] if _vv_client else None

def _vv_ser(doc):
    """Minimal serializer that won't change your schema—just stringifies _id and datetimes."""
    if not doc:
        return doc
    d = dict(doc)
    try:
        if "_id" in d:
            d["_id"] = str(d["_id"])
    except Exception:
        pass
    for k in ("createdAt", "updatedAt"):
        v = d.get(k)
        try:
            if hasattr(v, "isoformat"):
                d[k] = v.isoformat()
        except Exception:
            pass
    return d

# ---------- Debug: list registered routes ----------
@app.get("/debug/routes")
def _vv_debug_routes():
    try:
        rules = sorted([str(r) for r in app.url_map.iter_rules()])
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "routes": rules}), 200

# ---------- Debug: Mongo connectivity/status ----------
@app.get("/debug/mongo")
def _vv_debug_mongo():
    info = {
        "pymongo_imported": bool(_VV_MongoClient),
        "uri_present": bool(_VV_MONGODB_URI),
        "db_name": _VV_DB_NAME,
        "connected": False,
        "ping_ok": False,
        "collections": [],
    }
    try:
        if _vv_client:
            info["connected"] = True
            try:
                _vv_client.admin.command("ping")
                info["ping_ok"] = True
            except Exception as e:
                info["ping_error"] = str(e)
        if _vv_db:
            info["collections"] = sorted(_vv_db.list_collection_names())
    except Exception as e:
        info["error"] = str(e)
        return jsonify(info), 500
    return jsonify(info), 200

# ---------- Login (matches your plaintext user docs) ----------
@app.post("/login")
def vv_login_api():
    """
    JSON body: { "username": "AumkarM", "password": "MySecret123!" }
    Returns: { "user": {...}, "chats": [...], "library_items": [...] }
    """
    if not _vv_db:
        return jsonify({"detail": "Database connection not available"}), 500

    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"detail": "Invalid JSON"}), 400

    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    # Your user docs are plaintext, exactly like you showed
    user = _vv_db.users.find_one({"username": username, "password": password}, {"password": 0})
    if not user:
        return jsonify({"detail": "Invalid credentials"}), 401

    chats_cur = _vv_db.chats.find({"username": username}).sort("updatedAt", -1).limit(50)
    lib_cur = _vv_db.library_items.find({"username": username}).sort("updatedAt", -1).limit(100)

    return jsonify({
        "user": _vv_ser(user),
        "chats": [_vv_ser(x) for x in chats_cur],
        "library_items": [_vv_ser(x) for x in lib_cur],
    }), 200

# ---------- Get chats ----------
@app.get("/chats")
def vv_chats_api():
    """
    GET /chats?username=AumkarM&limit=50
    Returns: { username, chats: [...], count }
    """
    if not _vv_db:
        return jsonify({"detail": "Database connection not available"}), 500

    username = (request.args.get("username") or "").strip()
    try:
        limit = int(request.args.get("limit", "50"))
    except ValueError:
        limit = 50

    cur = _vv_db.chats.find({"username": username}).sort("updatedAt", -1).limit(limit)
    docs = [_vv_ser(x) for x in cur]
    return jsonify({"username": username, "chats": docs, "count": len(docs)}), 200

# ---------- Get library (gallery) ----------
@app.get("/library")
def vv_library_api():
    """
    GET /library?username=AumkarM&limit=100
    Returns: { username, library_items: [...], count }
    """
    if not _vv_db:
        return jsonify({"detail": "Database connection not available"}), 500

    username = (request.args.get("username") or "").strip()
    try:
        limit = int(request.args.get("limit", "100"))
    except ValueError:
        limit = 100

    cur = _vv_db.library_items.find({"username": username}).sort("updatedAt", -1).limit(limit)
    docs = [_vv_ser(x) for x in cur]
    return jsonify({"username": username, "library_items": docs, "count": len(docs)}), 200

# =================== end append-only block ===================


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting simple Flask test app on port {port}")
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.methods} {rule.rule}")

    app.run(host="0.0.0.0", port=port, debug=True)