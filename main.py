#!/usr/bin/env python3
import os
import re
import asyncio
import copy
from typing import Optional, List, Tuple
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

try:
    import cohere

    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    print("Warning: cohere module not available. Chat functionality will be disabled.")

try:
    from pymongo import MongoClient, DESCENDING
    from bson import ObjectId

    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("Warning: pymongo module not available. Authentication functionality will be disabled.")

from interactive_audio_processor import InteractiveAudioProcessor
from file_merger import merge_files

# MongoDB configuration
MONGODB_URI = "mongodb+srv://chatapp_user:AM20060305!_ilovesushi@cluster0.rw6klmv.mongodb.net/chatapp?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "chatapp"

# Global MongoDB variables
mongo_client = None
mongo_db = None

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


def serialize_mongo_doc(doc):
    """Convert MongoDB document to JSON-serializable format WITHOUT modifying the original."""
    if doc is None:
        return None

    # Create a deep copy to avoid modifying the original document
    doc_copy = copy.deepcopy(doc)

    # Convert ObjectId to string
    if '_id' in doc_copy:
        doc_copy['_id'] = str(doc_copy['_id'])

    # Convert datetime objects to ISO format strings
    for key, value in doc_copy.items():
        if isinstance(value, datetime):
            doc_copy[key] = value.isoformat()
        elif isinstance(value, list):
            # Handle nested documents in arrays (like messages in chats)
            for item in value:
                if isinstance(item, dict):
                    for nested_key, nested_value in item.items():
                        if isinstance(nested_value, datetime):
                            item[nested_key] = nested_value.isoformat()

    return doc_copy


def get_user_by_credentials(username: str, password: str):
    """Authenticate user by username and password - using the same approach as your working script."""
    if not MONGODB_AVAILABLE or not mongo_db:
        print("MongoDB not available for authentication")
        return None

    try:
        print(f"Searching for user: {username}")
        user = mongo_db.users.find_one({
            "username": username,
            "password": password
        })

        if user:
            print(f"User found: {user.get('username')}")
            return serialize_mongo_doc(user)
        else:
            print(f"User not found or wrong password")
            return None

    except Exception as e:
        print(f"Error authenticating user {username}: {e}")
        return None


def get_user_chats(username: str, limit: int = 50):
    """Fetch chats for a given username - using the EXACT same approach as your working script."""
    if not MONGODB_AVAILABLE or not mongo_db:
        print("MongoDB not available for chats")
        return []

    try:
        print(f"Fetching chats for username: {username}")

        # Use the EXACT same query as your working script
        chats = list(
            mongo_db.chats
            .find({"username": username})
            .sort("updatedAt", DESCENDING)  # Use DESCENDING constant like your working script
            .limit(limit)
        )

        print(f"Found {len(chats)} chats for {username}")

        # Serialize each chat
        serialized_chats = []
        for chat in chats:
            serialized_chat = serialize_mongo_doc(chat)
            serialized_chats.append(serialized_chat)

        return serialized_chats

    except Exception as e:
        print(f"Error fetching chats for {username}: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_user_library_items(username: str, limit: int = 100):
    """Fetch library items for a given username - using the same approach."""
    if not MONGODB_AVAILABLE or not mongo_db:
        print("MongoDB not available for library")
        return []

    try:
        print(f"Fetching library items for username: {username}")

        library_items = list(
            mongo_db.library_items
            .find({"username": username})
            .sort("updatedAt", DESCENDING)
            .limit(limit)
        )

        print(f"Found {len(library_items)} library items for {username}")

        # Serialize each item
        serialized_items = []
        for item in library_items:
            serialized_item = serialize_mongo_doc(item)
            serialized_items.append(serialized_item)

        return serialized_items

    except Exception as e:
        print(f"Error fetching library items for {username}: {e}")
        import traceback
        traceback.print_exc()
        return []


# Flask app setup
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
CORS(app)

CLEANVOICE_KEY = os.getenv("CLEANVOICE_API_KEY", "FJB8s8nbmY9UQcfeXFeB6tqJmjwDUkKN")
iap = InteractiveAudioProcessor(CLEANVOICE_KEY)

# Cohere setup
COHERE_KEY = os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_KEY) if COHERE_KEY and COHERE_AVAILABLE else None


@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "service": "VibeVideo Flask API"}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "mongodb": MONGODB_AVAILABLE})


@app.route("/debug/mongo", methods=["GET"])
def debug_mongo():
    """Debug endpoint to test MongoDB connection and data."""
    if not MONGODB_AVAILABLE or not mongo_db:
        return jsonify({"error": "MongoDB not available"}), 500

    try:
        # Test basic connection
        mongo_client.admin.command('ping')

        # Count documents
        user_count = mongo_db.users.count_documents({})
        chat_count = mongo_db.chats.count_documents({})
        library_count = mongo_db.library_items.count_documents({})

        # Get sample user
        sample_user = mongo_db.users.find_one()
        sample_chat = mongo_db.chats.find_one()

        return jsonify({
            "connection": "OK",
            "counts": {
                "users": user_count,
                "chats": chat_count,
                "library_items": library_count
            },
            "sample_user": serialize_mongo_doc(sample_user) if sample_user else None,
            "sample_chat": serialize_mongo_doc(sample_chat) if sample_chat else None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/login", methods=["POST"])
def login():
    """Authenticate user against MongoDB database."""
    print("=== LOGIN ATTEMPT ===")

    if not MONGODB_AVAILABLE or not mongo_db:
        return jsonify({"error": "Database connection not available"}), 500

    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        username = data.get("username", "").strip()
        password = data.get("password", "").strip()

        print(f"Login attempt for username: {username}")

        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400

        # Authenticate user
        user = get_user_by_credentials(username, password)

        if not user:
            return jsonify({"error": "Invalid credentials"}), 401

        print(f"Login successful for: {username}")

        # Remove password from response for security
        user_response = {
            "_id": user.get("_id"),
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
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


@app.route("/chats", methods=["GET"])
def get_chats():
    """Get chats for a specific user."""
    print("=== GET CHATS REQUEST ===")

    if not MONGODB_AVAILABLE or not mongo_db:
        return jsonify({"error": "Database connection not available"}), 500

    username = request.args.get("username")
    print(f"Requested chats for username: {username}")

    if not username:
        return jsonify({"error": "Username parameter is required"}), 400

    try:
        limit = int(request.args.get("limit", 50))
        chats = get_user_chats(username, limit)

        response = {
            "username": username,
            "chats": chats,
            "count": len(chats)
        }

        print(f"Returning {len(chats)} chats")
        return jsonify(response), 200

    except ValueError:
        return jsonify({"error": "Invalid limit parameter"}), 400
    except Exception as e:
        print(f"Get chats error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


@app.route("/library", methods=["GET"])
def get_library():
    """Get library items for a specific user."""
    print("=== GET LIBRARY REQUEST ===")

    if not MONGODB_AVAILABLE or not mongo_db:
        return jsonify({"error": "Database connection not available"}), 500

    username = request.args.get("username")
    print(f"Requested library for username: {username}")

    if not username:
        return jsonify({"error": "Username parameter is required"}), 400

    try:
        limit = int(request.args.get("limit", 100))
        library_items = get_user_library_items(username, limit)

        response = {
            "username": username,
            "library_items": library_items,
            "count": len(library_items)
        }

        print(f"Returning {len(library_items)} library items")
        return jsonify(response), 200

    except ValueError:
        return jsonify({"error": "Invalid limit parameter"}), 400
    except Exception as e:
        print(f"Get library error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting Flask app on port {port}")
    print(f"MongoDB available: {MONGODB_AVAILABLE}")

    if MONGODB_AVAILABLE:
        print("Testing MongoDB connection...")
        try:
            # Test the connection
            test_user = mongo_db.users.find_one()
            if test_user:
                print(f"MongoDB test successful. Sample user: {test_user.get('username')}")
            else:
                print("MongoDB connected but no users found")
        except Exception as e:
            print(f"MongoDB test failed: {e}")

    app.run(host="0.0.0.0", port=port, debug=True)