import io
import argparse
import asyncio
import numpy as np
import ffmpeg
from time import time, sleep
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from whisper_streaming_custom.whisper_online import backend_factory, online_factory, add_shared_args, warmup_asr
from timed_objects import ASRToken

import math
import logging
from datetime import timedelta
import traceback
from state import SharedState
from formatters import format_time
from parse_args import parse_args
from audio import AudioProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



args = parse_args()

SAMPLE_RATE = 16000
# CHANNELS = 1
# SAMPLES_PER_SEC = int(SAMPLE_RATE * args.min_chunk_size)
# BYTES_PER_SAMPLE = 2  # s16le = 2 bytes per sample
# BYTES_PER_SEC = SAMPLES_PER_SEC * BYTES_PER_SAMPLE
# MAX_BYTES_PER_SEC = 32000 * 5  # 5 seconds of audio at 32 kHz


##### LOAD APP #####

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
        diarization = DiartDiarization(SAMPLE_RATE)
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








@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    audio_processor = AudioProcessor(args, asr, tokenizer)

    await websocket.accept()
    logger.info("WebSocket connection opened.")

    ffmpeg_process = None
    pcm_buffer = bytearray()
    
    transcription_queue = asyncio.Queue() if args.transcription else None
    diarization_queue = asyncio.Queue() if args.diarization else None
    
    online = None

    ffmpeg_process, online, pcm_buffer = await audio_processor.restart_ffmpeg(ffmpeg_process, online, pcm_buffer)
    tasks = []    
    if args.transcription and online:
        tasks.append(asyncio.create_task(
            audio_processor.transcription_processor(transcription_queue, online)))    
    if args.diarization and diarization:
        tasks.append(asyncio.create_task(
            audio_processor.diarization_processor(diarization_queue, diarization)))
    formatter_task = asyncio.create_task(audio_processor.results_formatter(websocket))
    tasks.append(formatter_task)
    stdout_reader_task = asyncio.create_task(audio_processor.ffmpeg_stdout_reader(ffmpeg_process, pcm_buffer, diarization_queue, transcription_queue))
    tasks.append(stdout_reader_task)
    
    try:
        while True:
            # Receive incoming WebM audio chunks from the client
            message = await websocket.receive_bytes()
            try:
                ffmpeg_process.stdin.write(message)
                ffmpeg_process.stdin.flush()
            except (BrokenPipeError, AttributeError) as e:
                logger.warning(f"Error writing to FFmpeg: {e}. Restarting...")
                ffmpeg_process, online, pcm_buffer = await audio_processor.restart_ffmpeg(ffmpeg_process, online, pcm_buffer)
                ffmpeg_process.stdin.write(message)
                ffmpeg_process.stdin.flush()
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected.")
    finally:
        for task in tasks:
            task.cancel()
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        if args.diarization and diarization:
            diarization.close()
        logger.info("WebSocket endpoint cleaned up.")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "whisper_fastapi_online_server:app", host=args.host, port=args.port, reload=True,
        log_level="info"
    )