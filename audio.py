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


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AudioProcessor:
    
    def __init__(self, args, asr, tokenizer):
        self.args = args
        self.sample_rate = 16000
        self.channels = 1
        self.samples_per_sec = int(self.sample_rate * args.min_chunk_size)
        self.bytes_per_sample = 2
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        self.shared_state = SharedState()
        self.asr = asr
        self.tokenizer = tokenizer

    def convert_pcm_to_float(self, pcm_buffer):
        """
        Converts a PCM buffer in s16le format to a normalized NumPy array.
        Arg: pcm_buffer. PCM buffer containing raw audio data in s16le format
        Returns: np.ndarray. NumPy array of float32 type normalized between -1.0 and 1.0
        """
        pcm_array = (np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) 
                    / 32768.0)
        return pcm_array

    async def start_ffmpeg_decoder(self):
        """
        Start an FFmpeg process in async streaming mode that reads WebM from stdin
        and outputs raw s16le PCM on stdout. Returns the process object.
        """
        process = (
            ffmpeg.input("pipe:0", format="webm")
            .output(
                "pipe:1",
                format="s16le",
                acodec="pcm_s16le",
                ac=self.channels,
                ar=str(self.sample_rate),
            )
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        return process

    async def restart_ffmpeg(self, ffmpeg_process, online, pcm_buffer):
        if ffmpeg_process:
            try:
                ffmpeg_process.kill()
                await asyncio.get_event_loop().run_in_executor(None, ffmpeg_process.wait)
            except Exception as e:
                logger.warning(f"Error killing FFmpeg process: {e}")
        ffmpeg_process = await self.start_ffmpeg_decoder()
        pcm_buffer = bytearray()
        
        if self.args.transcription:
            online = online_factory(self.args, self.asr, self.tokenizer)
        
        await self.shared_state.reset()
        logger.info("FFmpeg process started.")
        return ffmpeg_process, online, pcm_buffer



    async def ffmpeg_stdout_reader(self, ffmpeg_process, pcm_buffer, diarization_queue, transcription_queue):
        loop = asyncio.get_event_loop()
        beg = time()
        
        while True:
            try:
                elapsed_time = math.floor((time() - beg) * 10) / 10 # Round to 0.1 sec
                ffmpeg_buffer_from_duration = max(int(32000 * elapsed_time), 4096)
                beg = time()

                # Read chunk with timeout
                try:
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, ffmpeg_process.stdout.read, ffmpeg_buffer_from_duration
                        ),
                        timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg read timeout. Restarting...")
                    ffmpeg_process, online, pcm_buffer = await self.restart_ffmpeg(ffmpeg_process, online, pcm_buffer)
                    beg = time()
                    continue  # Skip processing and read from new process

                if not chunk:
                    logger.info("FFmpeg stdout closed.")
                    break
                pcm_buffer.extend(chunk)
                        
                if self.args.diarization and diarization_queue:
                    await diarization_queue.put(self.convert_pcm_to_float(pcm_buffer).copy())

                if len(pcm_buffer) >= self.bytes_per_sec:
                    if len(pcm_buffer) > self.max_bytes_per_sec:
                        logger.warning(
                            f"""Audio buffer is too large: {len(pcm_buffer) / self.bytes_per_sec:.2f} seconds.
                            The model probably struggles to keep up. Consider using a smaller model.
                            """)

                    pcm_array = self.convert_pcm_to_float(pcm_buffer[:self.max_bytes_per_sec])
                    pcm_buffer = pcm_buffer[self.max_bytes_per_sec:]
                    
                    if self.args.transcription and transcription_queue:
                        await transcription_queue.put(pcm_array.copy())
                    
                    
                    if not self.args.transcription and not self.args.diarization:
                        await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                break
        logger.info("Exiting ffmpeg_stdout_reader...")




    async def transcription_processor(self, pcm_queue, online):
        full_transcription = ""
        sep = online.asr.sep
        
        while True:
            try:
                pcm_array = await pcm_queue.get()
                
                logger.info(f"{len(online.audio_buffer) / online.SAMPLING_RATE} seconds of audio will be processed by the model.")
                
                # Process transcription
                online.insert_audio_chunk(pcm_array)
                new_tokens = online.process_iter()
                
                if new_tokens:
                    full_transcription += sep.join([t.text for t in new_tokens])
                    
                _buffer = online.get_buffer()
                buffer = _buffer.text
                end_buffer = _buffer.end if _buffer.end else (new_tokens[-1].end if new_tokens else 0)
                
                if buffer in full_transcription:
                    buffer = ""
                    
                await self.shared_state.update_transcription(
                    new_tokens, buffer, end_buffer, full_transcription, sep)
                
            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
            finally:
                pcm_queue.task_done()

    async def diarization_processor(self, pcm_queue, diarization_obj):
        buffer_diarization = ""
        
        while True:
            try:
                pcm_array = await pcm_queue.get()
                
                # Process diarization
                await diarization_obj.diarize(pcm_array)
                
                # Get current state
                state = await self.shared_state.get_current_state()
                tokens = state["tokens"]
                end_attributed_speaker = state["end_attributed_speaker"]
                
                # Update speaker information
                new_end_attributed_speaker = diarization_obj.assign_speakers_to_tokens(
                    end_attributed_speaker, tokens)
                
                await self.shared_state.update_diarization(new_end_attributed_speaker, buffer_diarization)
                
            except Exception as e:
                logger.warning(f"Exception in diarization_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
            finally:
                pcm_queue.task_done()

    async def results_formatter(self, websocket):
        while True:
            try:
                # Get the current state
                state = await self.shared_state.get_current_state()
                tokens = state["tokens"]
                buffer_transcription = state["buffer_transcription"]
                buffer_diarization = state["buffer_diarization"]
                end_attributed_speaker = state["end_attributed_speaker"]
                remaining_time_transcription = state["remaining_time_transcription"]
                remaining_time_diarization = state["remaining_time_diarization"]
                sep = state["sep"]
                
                # If diarization is enabled but no transcription, add dummy tokens periodically
                if (not tokens or tokens[-1].is_dummy) and not self.args.transcription and self.args.diarization:
                    await self.shared_state.add_dummy_token()
                    sleep(0.5)
                    state = await self.shared_state.get_current_state()
                    tokens = state["tokens"]
                # Process tokens to create response
                previous_speaker = -1
                lines = []
                last_end_diarized = 0
                undiarized_text = []
                
                for token in tokens:
                    speaker = token.speaker
                    if self.args.diarization:
                        if (speaker == -1 or speaker == 0) and token.end >= end_attributed_speaker:
                            undiarized_text.append(token.text)
                            continue
                        elif (speaker == -1 or speaker == 0) and token.end < end_attributed_speaker:
                            speaker = previous_speaker
                        if speaker not in [-1, 0]:
                            last_end_diarized = max(token.end, last_end_diarized)

                    if speaker != previous_speaker or not lines:
                        lines.append(
                            {
                                "speaker": speaker,
                                "text": token.text,
                                "beg": format_time(token.start),
                                "end": format_time(token.end),
                                "diff": round(token.end - last_end_diarized, 2)
                            }
                        )
                        previous_speaker = speaker
                    elif token.text:  # Only append if text isn't empty
                        lines[-1]["text"] += sep + token.text
                        lines[-1]["end"] = format_time(token.end)
                        lines[-1]["diff"] = round(token.end - last_end_diarized, 2)
                
                if undiarized_text:
                    combined_buffer_diarization = sep.join(undiarized_text)
                    if buffer_transcription:
                        combined_buffer_diarization += sep
                    await self.shared_state.update_diarization(end_attributed_speaker, combined_buffer_diarization)
                    buffer_diarization = combined_buffer_diarization
                    
                if lines:
                    response = {
                        "lines": lines, 
                        "buffer_transcription": buffer_transcription,
                        "buffer_diarization": buffer_diarization,
                        "remaining_time_transcription": remaining_time_transcription,
                        "remaining_time_diarization": remaining_time_diarization
                    }
                else:
                    response = {
                        "lines": [{
                            "speaker": 1,
                            "text": "",
                            "beg": format_time(0),
                            "end": format_time(tokens[-1].end) if tokens else format_time(0),
                            "diff": 0
                    }],
                        "buffer_transcription": buffer_transcription,
                        "buffer_diarization": buffer_diarization,
                        "remaining_time_transcription": remaining_time_transcription,
                        "remaining_time_diarization": remaining_time_diarization

                    }
                
                response_content = ' '.join([str(line['speaker']) + ' ' + line["text"] for line in lines]) + ' | ' + buffer_transcription + ' | ' + buffer_diarization
                
                if response_content != self.shared_state.last_response_content:
                    if lines or buffer_transcription or buffer_diarization:
                        await websocket.send_json(response)
                        self.shared_state.last_response_content = response_content
                
                # Add a small delay to avoid overwhelming the client
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Exception in results_formatter: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.5)  # Back off on error
