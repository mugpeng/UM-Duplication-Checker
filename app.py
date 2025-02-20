import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List
import tempfile
import shutil
from clip_sift_search import analyze_images
from fastapi.responses import StreamingResponse, JSONResponse
import json
import asyncio
from sse_starlette.sse import EventSourceResponse
import queue
import threading
from datetime import datetime, timedelta

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_BASE_DIR = 'uploads'
CLEANUP_THRESHOLD_HOURS = 24  # Cleanup folders older than 24 hours

# Global progress queue
progress_queue = queue.Queue()

# Ensure base upload directory exists
os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)

def get_user_upload_dir(session_id: str) -> str:
    """Create and return a user-specific upload directory"""
    user_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def cleanup_old_uploads():
    """Clean up upload directories older than threshold"""
    try:
        current_time = datetime.now()
        for session_id in os.listdir(UPLOAD_BASE_DIR):
            session_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
            if os.path.isdir(session_dir):
                dir_modified_time = datetime.fromtimestamp(os.path.getmtime(session_dir))
                if current_time - dir_modified_time > timedelta(hours=CLEANUP_THRESHOLD_HOURS):
                    shutil.rmtree(session_dir)
    except Exception as e:
        print(f"Error during old uploads cleanup: {str(e)}")

def analyze_files_with_progress(folder_path):
    """Run analysis in a separate thread and put progress updates in the queue"""
    try:
        results = analyze_images(folder_path, progress_callback=lambda p: progress_queue.put({"progress": p}))
        progress_queue.put({"done": True, "results": results})
    except Exception as e:
        progress_queue.put({"error": str(e)})

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        # Generate unique session ID for this upload
        session_id = str(uuid.uuid4())
        user_upload_dir = get_user_upload_dir(session_id)
        
        # Clean up old upload directories
        cleanup_old_uploads()
        
        # Save uploaded files preserving directory structure
        saved_files = []
        for file in files:
            # Handle potential directory structure in filename
            file_path = os.path.join(user_upload_dir, file.filename)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
            
        print(f"Saved files in session {session_id}: {saved_files}")  # Debug log
        
        # Start analysis in a separate thread
        if saved_files:
            # Clear the queue
            while not progress_queue.empty():
                progress_queue.get()
            
            thread = threading.Thread(target=analyze_files_with_progress, args=(user_upload_dir,))
            thread.start()
            return {"message": "Analysis started", "session_id": session_id}
        else:
            return {"error": "No files were saved successfully"}
        
    except Exception as e:
        print(f"Error during upload: {str(e)}")  # Debug log
        return {"error": str(e)}

@app.post("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up a specific session's uploaded files"""
    try:
        session_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        return {"message": "Session cleanup successful"}
    except Exception as e:
        return {"error": f"Session cleanup failed: {str(e)}"}

@app.post("/cleanup")
async def cleanup_files():
    """Clean up all uploaded files"""
    try:
        if os.path.exists(UPLOAD_BASE_DIR):
            # Instead of removing the base directory, clean up its contents
            for item in os.listdir(UPLOAD_BASE_DIR):
                item_path = os.path.join(UPLOAD_BASE_DIR, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        return {"message": "Cleanup successful"}
    except Exception as e:
        return {"error": f"Cleanup failed: {str(e)}"}

@app.get("/progress")
async def progress_stream():
    async def event_generator():
        while True:
            if not progress_queue.empty():
                data = progress_queue.get()
                if "error" in data:
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": data["error"]})
                    }
                    break
                elif "done" in data:
                    yield {
                        "event": "complete",
                        "data": json.dumps(data["results"])
                    }
                    break
                else:
                    yield {
                        "event": "progress",
                        "data": json.dumps(data)
                    }
            await asyncio.sleep(0.1)  # Small delay to prevent CPU overload

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 