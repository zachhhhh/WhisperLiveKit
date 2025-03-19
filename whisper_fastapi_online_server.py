from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from whisper_streaming_custom.whisper_online import backend_factory, warmup_asr
import asyncio
import logging
from parse_args import parse_args
from audio import AudioProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr, tokenizer, diarization
    if args.transcription:
        asr, tokenizer = backend_factory(args)
        warmup_asr(asr, args.warmup_file)
    else:
        asr, tokenizer = None, None

    if args.diarization:
        from diarization.diarization_online import DiartDiarization
        diarization = DiartDiarization()
    else :
        diarization = None
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load demo HTML for the root endpoint
with open("web/live_transcription.html", "r", encoding="utf-8") as f:
    html = f.read()

@app.get("/")
async def get():
    return HTMLResponse(html)


async def handle_websocket_results(websocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response)
    except Exception as e:
        logger.warning(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    audio_processor = AudioProcessor(args, asr, tokenizer)

    await websocket.accept()
    logger.info("WebSocket connection opened.")
            
    results_generator = await audio_processor.create_tasks(diarization)
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected.")
    finally:
        websocket_task.cancel()
        audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up.")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "whisper_fastapi_online_server:app", host=args.host, port=args.port, reload=True,
        log_level="info"
    )