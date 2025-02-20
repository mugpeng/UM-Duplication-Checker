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
UPLOAD_FOLDER = 'uploads'

# Global progress queue
progress_queue = queue.Queue()

# Ensure upload directory exists
if os.path.exists(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        # Clean previous files if any
        for item in os.listdir(UPLOAD_FOLDER):
            item_path = os.path.join(UPLOAD_FOLDER, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
        
        # Save uploaded files
        saved_files = []
        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
            
        print(f"Saved files: {saved_files}")  # Debug log
        
        # Start analysis in a separate thread
        if saved_files:
            # Clear the queue
            while not progress_queue.empty():
                progress_queue.get()
            
            thread = threading.Thread(target=analyze_files_with_progress, args=(UPLOAD_FOLDER,))
            thread.start()
            return {"message": "Analysis started"}
        else:
            return {"error": "No files were saved successfully"}
        
    except Exception as e:
        print(f"Error during upload: {str(e)}")  # Debug log
        return {"error": str(e)}

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

@app.post("/cleanup")
async def cleanup_files():
    """Clean up all uploaded files"""
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER)
        return {"message": "Cleanup successful"}
    except Exception as e:
        return {"error": f"Cleanup failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 