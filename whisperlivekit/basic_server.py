from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from whisperlivekit import WhisperLiveKit
from whisperlivekit.audio_processor import AudioProcessor

import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

kit = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global kit
    kit = WhisperLiveKit()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return HTMLResponse(kit.web_interface())


async def handle_websocket_results(websocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response)
    except Exception as e:
        logger.warning(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    audio_processor = AudioProcessor()

    await websocket.accept()
    logger.info("WebSocket connection opened.")
            
    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected.")
    finally:
        websocket_task.cancel()
        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up.")

def main():
    """Entry point for the CLI command."""
    import uvicorn
    
    temp_kit = WhisperLiveKit(transcription=False, diarization=False)
    
    uvicorn.run(
        "whisperlivekit.basic_server:app", 
        host=temp_kit.args.host, 
        port=temp_kit.args.port, 
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
