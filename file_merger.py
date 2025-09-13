#!/usr/bin/env python3
import os, re, pathlib, requests, sys
import cloudinary
from cloudinary.uploader import upload
from cloudinary.utils import cloudinary_url
from typing import List

# Cloudinary configuration - these should be set via environment variables in production
CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dar0atu6x")
API_KEY = os.getenv("CLOUDINARY_API_KEY", "514998552917611")
API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "MSkq8QMFFuFjjzkYPnhcLFr9S74")

# Supported file extensions
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg"}

def init_cloudinary():
    """Initialize Cloudinary with the configured credentials."""
    cloudinary.config(
        cloud_name=CLOUD_NAME,
        api_key=API_KEY,
        api_secret=API_SECRET,
        secure=True,
    )

def as_public_id(path: str) -> str:
    """Convert a file path to a safe Cloudinary public_id."""
    name = pathlib.Path(path).stem
    # safe-ish public_id
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)

def detect_mode(ext: str) -> str:
    """Detect if the file extension is video or audio."""
    e = ext.lower()
    if e in VIDEO_EXTS: 
        return "video"
    if e in AUDIO_EXTS: 
        return "audio"
    raise ValueError(f"Unsupported file extension: {ext}")

def ensure_files_exist(paths: List[str]):
    """Ensure all files in the list exist."""
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing local file: {p}")

def ensure_same_extension(paths: List[str]) -> str:
    """Ensure all files share the exact same extension; return that extension (like '.m4a')."""
    exts = {pathlib.Path(p).suffix.lower() for p in paths}
    if len(exts) != 1:
        raise ValueError(f"All files must have the SAME type/extension. Got: {sorted(exts)}")
    return exts.pop()

def upload_as_video_resource(local_path: str, public_id: str) -> str:
    """
    Upload to Cloudinary as resource_type='video' (works for both video *and* audio).
    Returns the public_id (same as provided), which we'll use in transformations.
    """
    print(f"📤 Uploading {local_path} as public_id '{public_id}' …")
    res = upload(
        local_path,
        resource_type="video",
        public_id=public_id,
        overwrite=True,
        eager_async=False,
    )
    # res includes 'public_id' and 'secure_url' among other fields.
    print(f"✅ Uploaded -> {res.get('secure_url')}")
    return res["public_id"]

def build_splice_url(base_public_id: str, tail_public_ids: List[str], output_format: str) -> str:
    """
    Build a Cloudinary delivery URL that concatenates all clips in order:
    base_public_id + splice(overlay=tail1) + splice(overlay=tail2) + …
    """
    # Each extra clip is added as an overlay with flag 'splice'
    transforms = [{"overlay": f"video:{pid}", "flags": "splice"} for pid in tail_public_ids]

    # Build the delivery URL on top of the first clip (base)
    merged_url, _ = cloudinary_url(
        base_public_id,
        resource_type="video",
        format=output_format,       # keep SAME extension as inputs
        transformation=transforms,
        secure=True,
    )
    return merged_url

def download(url: str, out_path: str):
    """Download a file from URL to the specified path."""
    print(f"⬇️  Downloading merged result …")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
    print(f"🎉 Saved -> {out_path}")

def merge_files(local_files: List[str], output_path: str = None) -> str:
    """
    Merge multiple audio/video files of the same type using Cloudinary.
    
    Args:
        local_files: List of file paths to merge (must be same extension)
        output_path: Optional output path. If not provided, generates one.
        
    Returns:
        Path to the merged file
        
    Raises:
        ValueError: If files have different extensions
        FileNotFoundError: If any file doesn't exist
        requests.HTTPError: If download fails
    """
    if len(local_files) < 2:
        raise ValueError("Need at least 2 files to merge.")

    # Ensure all files exist
    ensure_files_exist(local_files)

    # Initialize Cloudinary
    init_cloudinary()

    # 1) Enforce SAME extension across all inputs
    ext = ensure_same_extension(local_files)            # e.g., ".m4a"
    mode = detect_mode(ext)                             # "audio" or "video"
    output_format = ext.lstrip(".")                     # e.g., "m4a"
    
    # Generate output path if not provided
    if not output_path:
        output_path = f"merged{ext}"

    print(f"✅ All files share extension {ext} ({mode}); output will be {output_path}")

    # 2) Upload all files as "video" resources (Cloudinary handles audio in video pipeline)
    public_ids = []
    for p in local_files:
        pid = as_public_id(p)
        # avoid collision if two files share same stem; append index if needed
        if pid in public_ids:
            pid = f"{pid}_{len(public_ids)}"
        upload_as_video_resource(p, pid)
        public_ids.append(pid)

    # 3) Build the splice URL: base is first clip, then splice each subsequent clip
    base, tail = public_ids[0], public_ids[1:]
    merged_delivery_url = build_splice_url(base, tail, output_format=output_format)

    print("🔗 Merged delivery URL:")
    print(merged_delivery_url)

    # 4) Trigger generation and download the derived result
    try:
        download(merged_delivery_url, output_path)
        return output_path
    except requests.HTTPError as e:
        # Helpful hint if Cloudinary plan/format isn't supported
        error_msg = f"❌ Download failed ({e}). If this is an audio format like {ext} and your plan doesn't allow it, " \
                   f"try converting to 'mp3' by setting output_format='mp3' and OUTPUT_FILE='merged.mp3'."
        print(error_msg, file=sys.stderr)
        raise
