import asyncio
import numpy as np
import ffmpeg
from time import time, sleep
import math
import logging
import traceback
from datetime import timedelta
from typing import List, Dict, Any
from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.whisper_streaming_custom.whisper_online import online_factory
from whisperlivekit.core import WhisperLiveKit

# Set up logging once
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

class AudioProcessor:
    """
    Processes audio streams for transcription and diarization.
    Handles audio processing, state management, and result formatting.
    """
    
    def __init__(self):
        """Initialize the audio processor with configuration, models, and state."""
        
        models = WhisperLiveKit()
        
        # Audio processing settings
        self.args = models.args
        self.sample_rate = 16000
        self.channels = 1
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size)
        self.bytes_per_sample = 2
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        
        # State management
        self.tokens = []
        self.buffer_transcription = ""
        self.buffer_diarization = ""
        self.full_transcription = ""
        self.end_buffer = 0
        self.end_attributed_speaker = 0
        self.lock = asyncio.Lock()
        self.beg_loop = time()
        self.sep = " "  # Default separator
        self.last_response_content = ""
        
        # Models and processing
        self.asr = models.asr
        self.tokenizer = models.tokenizer
        self.diarization = models.diarization
        self.ffmpeg_process = self.start_ffmpeg_decoder()
        self.transcription_queue = asyncio.Queue() if self.args.transcription else None
        self.diarization_queue = asyncio.Queue() if self.args.diarization else None
        self.pcm_buffer = bytearray()
        
        # Initialize transcription engine if enabled
        if self.args.transcription:
            self.online = online_factory(self.args, models.asr, models.tokenizer)

    def convert_pcm_to_float(self, pcm_buffer):
        """Convert PCM buffer in s16le format to normalized NumPy array."""
        return np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

    def start_ffmpeg_decoder(self):
        """Start FFmpeg process for WebM to PCM conversion."""
        return (ffmpeg.input("pipe:0", format="webm")
                .output("pipe:1", format="s16le", acodec="pcm_s16le", 
                        ac=self.channels, ar=str(self.sample_rate))
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True))

    async def restart_ffmpeg(self):
        """Restart the FFmpeg process after failure."""
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.kill()
                await asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.wait)
            except Exception as e:
                logger.warning(f"Error killing FFmpeg process: {e}")
            self.ffmpeg_process = self.start_ffmpeg_decoder()
            self.pcm_buffer = bytearray()

    async def update_transcription(self, new_tokens, buffer, end_buffer, full_transcription, sep):
        """Thread-safe update of transcription with new data."""
        async with self.lock:
            self.tokens.extend(new_tokens)
            self.buffer_transcription = buffer
            self.end_buffer = end_buffer
            self.full_transcription = full_transcription
            self.sep = sep
            
    async def update_diarization(self, end_attributed_speaker, buffer_diarization=""):
        """Thread-safe update of diarization with new data."""
        async with self.lock:
            self.end_attributed_speaker = end_attributed_speaker
            if buffer_diarization:
                self.buffer_diarization = buffer_diarization
            
    async def add_dummy_token(self):
        """Placeholder token when no transcription is available."""
        async with self.lock:
            current_time = time() - self.beg_loop
            self.tokens.append(ASRToken(
                start=current_time, end=current_time + 1,
                text=".", speaker=-1, is_dummy=True
            ))
            
    async def get_current_state(self):
        """Get current state."""
        async with self.lock:
            current_time = time()
            
            # Calculate remaining times
            remaining_transcription = 0
            if self.end_buffer > 0:
                remaining_transcription = max(0, round(current_time - self.beg_loop - self.end_buffer, 2))
                
            remaining_diarization = 0
            if self.tokens:
                latest_end = max(self.end_buffer, self.tokens[-1].end if self.tokens else 0)
                remaining_diarization = max(0, round(latest_end - self.end_attributed_speaker, 2))
                
            return {
                "tokens": self.tokens.copy(),
                "buffer_transcription": self.buffer_transcription,
                "buffer_diarization": self.buffer_diarization,
                "end_buffer": self.end_buffer,
                "end_attributed_speaker": self.end_attributed_speaker,
                "sep": self.sep,
                "remaining_time_transcription": remaining_transcription,
                "remaining_time_diarization": remaining_diarization
            }
            
    async def reset(self):
        """Reset all state variables to initial values."""
        async with self.lock:
            self.tokens = []
            self.buffer_transcription = self.buffer_diarization = ""
            self.end_buffer = self.end_attributed_speaker = 0
            self.full_transcription = self.last_response_content = ""
            self.beg_loop = time()

    async def ffmpeg_stdout_reader(self):
        """Read audio data from FFmpeg stdout and process it."""
        loop = asyncio.get_event_loop()
        beg = time()
        
        while True:
            try:
                # Calculate buffer size based on elapsed time
                elapsed_time = math.floor((time() - beg) * 10) / 10  # Round to 0.1 sec
                buffer_size = max(int(32000 * elapsed_time), 4096)
                beg = time()

                # Read chunk with timeout
                try:
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(None, self.ffmpeg_process.stdout.read, buffer_size),
                        timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg read timeout. Restarting...")
                    await self.restart_ffmpeg()
                    beg = time()
                    continue

                if not chunk:
                    logger.info("FFmpeg stdout closed.")
                    break
                    
                self.pcm_buffer.extend(chunk)
                        
                # Send to diarization if enabled
                if self.args.diarization and self.diarization_queue:
                    await self.diarization_queue.put(
                        self.convert_pcm_to_float(self.pcm_buffer).copy()
                    )

                # Process when we have enough data
                if len(self.pcm_buffer) >= self.bytes_per_sec:
                    if len(self.pcm_buffer) > self.max_bytes_per_sec:
                        logger.warning(
                            f"Audio buffer too large: {len(self.pcm_buffer) / self.bytes_per_sec:.2f}s. "
                            f"Consider using a smaller model."
                        )

                    # Process audio chunk
                    pcm_array = self.convert_pcm_to_float(self.pcm_buffer[:self.max_bytes_per_sec])
                    self.pcm_buffer = self.pcm_buffer[self.max_bytes_per_sec:]
                    
                    # Send to transcription if enabled
                    if self.args.transcription and self.transcription_queue:
                        await self.transcription_queue.put(pcm_array.copy())
                    
                    # Sleep if no processing is happening
                    if not self.args.transcription and not self.args.diarization:
                        await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                break

    async def transcription_processor(self):
        """Process audio chunks for transcription."""
        self.full_transcription = ""
        self.sep = self.online.asr.sep
        
        while True:
            try:
                pcm_array = await self.transcription_queue.get()
                
                logger.info(f"{len(self.online.audio_buffer) / self.online.SAMPLING_RATE} seconds of audio to process.")
                
                # Process transcription
                self.online.insert_audio_chunk(pcm_array)
                new_tokens = self.online.process_iter()
                
                if new_tokens:
                    self.full_transcription += self.sep.join([t.text for t in new_tokens])
                    
                # Get buffer information
                _buffer = self.online.get_buffer()
                buffer = _buffer.text
                end_buffer = _buffer.end if _buffer.end else (
                    new_tokens[-1].end if new_tokens else 0
                )
                
                # Avoid duplicating content
                if buffer in self.full_transcription:
                    buffer = ""
                    
                await self.update_transcription(
                    new_tokens, buffer, end_buffer, self.full_transcription, self.sep
                )
                
            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
            finally:
                self.transcription_queue.task_done()

    async def diarization_processor(self, diarization_obj):
        """Process audio chunks for speaker diarization."""
        buffer_diarization = ""
        
        while True:
            try:
                pcm_array = await self.diarization_queue.get()
                
                # Process diarization
                await diarization_obj.diarize(pcm_array)
                
                # Get current state and update speakers
                state = await self.get_current_state()
                new_end = diarization_obj.assign_speakers_to_tokens(
                    state["end_attributed_speaker"], state["tokens"]
                )
                
                await self.update_diarization(new_end, buffer_diarization)
                
            except Exception as e:
                logger.warning(f"Exception in diarization_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
            finally:
                self.diarization_queue.task_done()

    async def results_formatter(self):
        """Format processing results for output."""
        while True:
            try:
                # Get current state
                state = await self.get_current_state()
                tokens = state["tokens"]
                buffer_transcription = state["buffer_transcription"]
                buffer_diarization = state["buffer_diarization"]
                end_attributed_speaker = state["end_attributed_speaker"]
                sep = state["sep"]
                
                # Add dummy tokens if needed
                if (not tokens or tokens[-1].is_dummy) and not self.args.transcription and self.args.diarization:
                    await self.add_dummy_token()
                    sleep(0.5)
                    state = await self.get_current_state()
                    tokens = state["tokens"]
                
                # Format output
                previous_speaker = -1
                lines = []
                last_end_diarized = 0
                undiarized_text = []
                
                # Process each token
                for token in tokens:
                    speaker = token.speaker
                    
                    # Handle diarization
                    if self.args.diarization:
                        if (speaker in [-1, 0]) and token.end >= end_attributed_speaker:
                            undiarized_text.append(token.text)
                            continue
                        elif (speaker in [-1, 0]) and token.end < end_attributed_speaker:
                            speaker = previous_speaker
                        if speaker not in [-1, 0]:
                            last_end_diarized = max(token.end, last_end_diarized)

                    # Group by speaker
                    if speaker != previous_speaker or not lines:
                        lines.append({
                            "speaker": speaker,
                            "text": token.text,
                            "beg": format_time(token.start),
                            "end": format_time(token.end),
                            "diff": round(token.end - last_end_diarized, 2)
                        })
                        previous_speaker = speaker
                    elif token.text:  # Only append if text isn't empty
                        lines[-1]["text"] += sep + token.text
                        lines[-1]["end"] = format_time(token.end)
                        lines[-1]["diff"] = round(token.end - last_end_diarized, 2)
                
                # Handle undiarized text
                if undiarized_text:
                    combined = sep.join(undiarized_text)
                    if buffer_transcription:
                        combined += sep
                    await self.update_diarization(end_attributed_speaker, combined)
                    buffer_diarization = combined
                
                # Create response object
                if not lines:
                    lines = [{
                        "speaker": 1,
                        "text": "",
                        "beg": format_time(0),
                        "end": format_time(tokens[-1].end if tokens else 0),
                        "diff": 0
                    }]
                
                response = {
                    "lines": lines, 
                    "buffer_transcription": buffer_transcription,
                    "buffer_diarization": buffer_diarization,
                    "remaining_time_transcription": state["remaining_time_transcription"],
                    "remaining_time_diarization": state["remaining_time_diarization"]
                }
                
                # Only yield if content has changed
                response_content = ' '.join([f"{line['speaker']} {line['text']}" for line in lines]) + \
                                  f" | {buffer_transcription} | {buffer_diarization}"
                
                if response_content != self.last_response_content and (lines or buffer_transcription or buffer_diarization):
                    yield response
                    self.last_response_content = response_content
                
                await asyncio.sleep(0.1)  # Avoid overwhelming the client
                
            except Exception as e:
                logger.warning(f"Exception in results_formatter: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.5)  # Back off on error
                
    async def create_tasks(self):
        """Create and start processing tasks."""
            
        tasks = []    
        if self.args.transcription and self.online:
            tasks.append(asyncio.create_task(self.transcription_processor()))
            
        if self.args.diarization and self.diarization:
            tasks.append(asyncio.create_task(self.diarization_processor(self.diarization)))
        
        tasks.append(asyncio.create_task(self.ffmpeg_stdout_reader()))
        self.tasks = tasks
        
        return self.results_formatter()
        
    async def cleanup(self):
        """Clean up resources when processing is complete."""
        for task in self.tasks:
            task.cancel()
            
        try:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
            
        if self.args.diarization and hasattr(self, 'diarization'):
            self.diarization.close()

    async def process_audio(self, message):
        """Process incoming audio data."""
        try:
            self.ffmpeg_process.stdin.write(message)
            self.ffmpeg_process.stdin.flush()
        except (BrokenPipeError, AttributeError) as e:
            logger.warning(f"Error writing to FFmpeg: {e}. Restarting...")
            await self.restart_ffmpeg()
            self.ffmpeg_process.stdin.write(message)
            self.ffmpeg_process.stdin.flush()