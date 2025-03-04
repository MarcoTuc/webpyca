from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
import numpy as np
import base64
import io
import asyncio
import time
from typing import Optional, Literal
from PIL import Image
from automatas import GoL, MultiLenia, Lenia
from pydantic import BaseModel
import json
import inspect
import os
from pathlib import Path
import struct

app = FastAPI()

# Enable CORS - let's improve the configuration
origins = [
    "http://localhost:3000",  # React default
    "http://localhost:5173",  # Vite default
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "*"  # Allow all origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Global variables to store the current automata
current_automata = None
automata_type = None
streaming_active = False
binary_mode = False  # New flag to toggle between binary and image modes

class AutomataRequest(BaseModel):
    automata_type: Literal["gol", "lenia", "multilenia"]
    size: int = 256
    binary_mode: bool = False

@app.post("/select")
async def select_automata(request: AutomataRequest):
    """Select and initialize an automata type"""
    global current_automata, binary_mode
    
    try:
        if request.automata_type == "gol":
            current_automata = GoL(request.size)
        elif request.automata_type == "lenia":
            current_automata = Lenia(request.size)
        elif request.automata_type == "multilenia":
            current_automata = MultiLenia()
        else:
            raise HTTPException(status_code=400, detail="Invalid automata type")
        
        # Set the binary mode flag
        binary_mode = request.binary_mode
        
        # Return initial state
        if binary_mode:
            return {"status": "success", "binary_mode": True}
        else:
            return {"status": "success", "data": encode_grid(current_automata.grid)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/step")
async def step():
    """Perform a single step on the current automata"""
    global current_automata, binary_mode
    
    if current_automata is None:
        raise HTTPException(status_code=400, detail="No automata selected, use /select first")
    
    try:
        grid = current_automata.draw()
        
        if binary_mode:
            # Return binary data directly
            return Response(content=prepare_binary_grid(grid), 
                           media_type="application/octet-stream")
        else:
            return {"data": encode_grid(grid)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/spray")
async def spray(x: int, y: int):
    """Spray at the specified coordinates"""
    global current_automata
    
    if current_automata is None:
        raise HTTPException(status_code=400, detail="No automata selected, use /select first")
    
    try:
        current_automata.spray(y, x)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/run")
async def run(fps: int = 30, max_fps: int = 60):
    """Continuously run the automata and stream results"""
    global current_automata, streaming_active, binary_mode
    
    # Cap FPS at a reasonable maximum
    fps = min(fps, max_fps)
    
    if current_automata is None:
        raise HTTPException(status_code=400, detail="No automata selected, use /select first")
    
    streaming_active = True
    
    if binary_mode:
        # Use binary streaming
        async def generate_binary():
            frame_time = 1.0 / fps
            last_frame_time = time.time()
            
            try:
                while streaming_active:
                    current_time = time.time()
                    elapsed_since_last = current_time - last_frame_time
                    
                    if elapsed_since_last >= frame_time:
                        grid = current_automata.draw()
                        binary_data = prepare_binary_grid(grid)
                        
                        # First send the frame size as 4 bytes
                        size_bytes = struct.pack('>I', len(binary_data))
                        yield size_bytes
                        
                        # Then send the actual frame data
                        yield binary_data
                        
                        last_frame_time = current_time
                    
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                print(f"Streaming error: {e}")
        
        return StreamingResponse(generate_binary(), media_type="application/octet-stream")
    
    else:
        # Use Server-Sent Events (SSE) with image data
        async def generate():
            frame_time = 1.0 / fps
            last_frame_time = time.time()
            frames_sent = 0
            start_time = time.time()
            
            # Send initial connection event
            yield f"event: connected\ndata: {{}}\n\n"
            
            try:
                while streaming_active:
                    current_time = time.time()
                    elapsed_since_last = current_time - last_frame_time
                    
                    # Only send a new frame if enough time has passed
                    if elapsed_since_last >= frame_time:
                        grid = current_automata.draw()
                        encoded = encode_grid(grid)
                        
                        # Send the data with the update event type
                        yield f"event: update\ndata: {encoded}\n\n"
                        
                        # Update statistics
                        frames_sent += 1
                        last_frame_time = current_time
                        
                        # Calculate actual FPS every 30 frames
                        if frames_sent % 30 == 0:
                            actual_fps = frames_sent / (time.time() - start_time)
                            yield f"event: stats\ndata: {{\"actual_fps\": {actual_fps:.2f}}}\n\n"
                    
                    # Always provide some yield time to prevent CPU overload
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                error_data = json.dumps({"message": str(e)})
                yield f"event: error\ndata: {error_data}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/stop")
async def stop():
    """Stop the streaming if it's running"""
    global streaming_active
    streaming_active = False
    return {"status": "stopped"}

@app.get("/get_implementation_code/{automata_type}")
async def get_implementation_code(automata_type: Literal["gol", "lenia", "multilenia"]):
    """Get the source code of the specified automata implementation"""
    try:
        if automata_type == "gol":
            module_path = inspect.getsourcefile(GoL)
        elif automata_type == "lenia":
            module_path = inspect.getsourcefile(Lenia)
        elif automata_type == "multilenia":
            module_path = inspect.getsourcefile(MultiLenia)
        else:
            raise HTTPException(status_code=400, detail="Invalid automata type")
        
        # Read the file content
        with open(module_path, "r") as file:
            source_code = file.read()
            
        return {"code": source_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def prepare_binary_grid(grid):
    """Convert grid data to efficient binary representation"""
    # Check if grid is a PyTorch tensor
    try:
        import torch
        if isinstance(grid, torch.Tensor):
            grid = grid.detach().cpu().numpy()
    except ImportError:
        pass  # torch not available, continue with numpy processing
    
    if isinstance(grid, np.ndarray):
        # Get shape information
        shape = grid.shape
        dtype = grid.dtype
        
        # Convert shape to binary header (dimensions and type info)
        header = struct.pack('>BI', len(shape), np.dtype(dtype).num)
        for dim in shape:
            header += struct.pack('>I', dim)
            
        # Get the raw binary data
        binary_data = grid.tobytes()
        
        # Combine header and data
        return header + binary_data
    else:
        return b''

def encode_grid(grid):
    """Convert grid data to base64 encoded image"""
    # Check if grid is a PyTorch tensor
    try:
        import torch
        if isinstance(grid, torch.Tensor):
            grid = grid.detach().cpu().numpy()
    except ImportError:
        pass  # torch not available, continue with numpy processing
    
    if isinstance(grid, np.ndarray):
        # Handle different grid dimensions
        if grid.ndim == 2:
            # Convert to RGB by repeating across channels
            img_data = np.stack([grid, grid, grid], axis=2)
            img_data = (img_data * 255).astype(np.uint8)
        elif grid.ndim == 3 and grid.shape[2] == 3:
            # Already RGB format
            img_data = (grid * 255).astype(np.uint8)
        else:
            # Take the first 3 channels or pad if needed
            if grid.shape[2] >= 3:
                img_data = (grid[:, :, :3] * 255).astype(np.uint8)
            else:
                # Pad to 3 channels if less than 3
                img_data = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
                for i in range(min(3, grid.shape[2])):
                    img_data[:, :, i] = (grid[:, :, i] * 255).astype(np.uint8)
        
        # Convert to base64
        img = Image.fromarray(img_data)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        return ""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

