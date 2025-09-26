import asyncio
import numpy as np
from time import time, sleep
import math
import logging
import traceback
from argparse import Namespace
from whisperlivekit.timed_objects import ASRToken, Silence, Line, FrontData, State, Transcript, ChangeSpeaker
from whisperlivekit.core import TranscriptionEngine, online_factory, online_diarization_factory, online_translation_factory
from whisperlivekit.silero_vad_iterator import FixedVADIterator
from whisperlivekit.results_formater import format_output
from whisperlivekit.ffmpeg_manager import FFmpegManager, FFmpegState
# Set up logging once
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SENTINEL = object() # unique sentinel object for end of stream marker


async def get_all_from_queue(queue):
    items = []
    try:
        while True:
            item = queue.get_nowait()
            items.append(item)
    except asyncio.QueueEmpty:
        pass
    return items

class AudioProcessor:
    """
    Processes audio streams for transcription and diarization.
    Handles audio processing, state management, and result formatting.
    """
    
    def __init__(self, **kwargs):
        """Initialize the audio processor with configuration, models, and state."""
        
        if 'transcription_engine' in kwargs and isinstance(kwargs['transcription_engine'], TranscriptionEngine):
            models = kwargs['transcription_engine']
        else:
            models = TranscriptionEngine(**kwargs)

        # Audio processing settings with optional per-connection overrides
        self.base_engine = models
        self.args = Namespace(**vars(models.args))
        override_language = (
            kwargs.get('source_language')
            or kwargs.get('lan')
            or kwargs.get('language')
        )
        if override_language:
            self.args.lan = override_language

        if 'target_language' in kwargs and kwargs['target_language'] is not None:
            self.args.target_language = kwargs['target_language']

        if 'task' in kwargs and kwargs['task']:
            self.args.task = kwargs['task']

        # Audio processing settings
        self.sample_rate = 16000
        self.channels = 1
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size)
        self.bytes_per_sample = 2
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        self.is_pcm_input = self.args.pcm_input
        self.debug = False

        # State management
        self.is_stopping = False
        self.silence = False
        self.silence_duration = 0.0
        self.tokens = []
        self.translated_segments = []
        self.buffer_transcription = Transcript()
        self.end_buffer = 0
        self.end_attributed_speaker = 0
        self.lock = asyncio.Lock()
        self.beg_loop = None #to deal with a potential little lag at the websocket initialization, this is now set in process_audio
        self.sep = " "  # Default separator
        self.last_response_content = FrontData()
        self.last_detected_speaker = None
        self.speaker_languages = {}
        self.cumulative_pcm_len = 0
        self.diarization_before_transcription = False

        # Models and processing
        self.asr = models.asr
        self.tokenizer = models.tokenizer
        self.vac_model = models.vac_model
        if self.args.vac:
            self.vac = FixedVADIterator(models.vac_model)
        else:
            self.vac = None
                         
        self.ffmpeg_manager = None
        self.ffmpeg_reader_task = None
        self._ffmpeg_error = None

        if not self.is_pcm_input:
            self.ffmpeg_manager = FFmpegManager(
                sample_rate=self.sample_rate,
                channels=self.channels
            )
            async def handle_ffmpeg_error(error_type: str):
                logger.error(f"FFmpeg error: {error_type}")
                self._ffmpeg_error = error_type
            self.ffmpeg_manager.on_error_callback = handle_ffmpeg_error
             
        self.transcription_queue = asyncio.Queue() if self.args.transcription else None
        self.diarization_queue = asyncio.Queue() if self.args.diarization else None
        self.translation_queue = asyncio.Queue() if self.args.target_language else None
        self.pcm_buffer = bytearray()

        self.transcription_task = None
        self.diarization_task = None
        self.watchdog_task = None
        self.all_tasks_for_cleanup = []
        self.online_translation = None
        
        if self.args.transcription:
            self.online = online_factory(self.args, models.asr, models.tokenizer)
            self.sep = self.online.asr.sep
        if self.args.diarization:
            self.diarization = online_diarization_factory(self.args, models.diarization_model)
        if self.args.target_language:
            translation_model = None
            if hasattr(models, "get_translation_model"):
                translation_model = models.get_translation_model(self.args.lan)
            else:
                translation_model = getattr(models, 'translation_model', None)

            if translation_model:
                self.online_translation = online_translation_factory(self.args, translation_model)

    def convert_pcm_to_float(self, pcm_buffer):
        """Convert PCM buffer in s16le format to normalized NumPy array."""
        return np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

    async def add_dummy_token(self):
        """Placeholder token when no transcription is available."""
        async with self.lock:
            current_time = time() - self.beg_loop if self.beg_loop else 0
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
                remaining_transcription = max(0, round(current_time - self.beg_loop - self.end_buffer, 1))
                
            remaining_diarization = 0
            if self.tokens:
                latest_end = max(self.end_buffer, self.tokens[-1].end if self.tokens else 0)
                remaining_diarization = max(0, round(latest_end - self.end_attributed_speaker, 1))
                
            return State(
                tokens=self.tokens.copy(),
                translated_segments=self.translated_segments.copy(),
                buffer_transcription=self.buffer_transcription,
                end_buffer=self.end_buffer,
                end_attributed_speaker=self.end_attributed_speaker,
                remaining_time_transcription=remaining_transcription,
                remaining_time_diarization=remaining_diarization
            )
            
    async def reset(self):
        """Reset all state variables to initial values."""
        async with self.lock:
            self.tokens = []
            self.translated_segments = []
            self.buffer_transcription = Transcript()
            self.end_buffer = self.end_attributed_speaker = 0
            self.beg_loop = time()

    async def ffmpeg_stdout_reader(self):
        """Read audio data from FFmpeg stdout and process it into the PCM pipeline."""
        beg = time()
        while True:
            try:
                if self.is_stopping:
                    logger.info("Stopping ffmpeg_stdout_reader due to stopping flag.")
                    break

                state = await self.ffmpeg_manager.get_state() if self.ffmpeg_manager else FFmpegState.STOPPED
                if state == FFmpegState.FAILED:
                    logger.error("FFmpeg is in FAILED state, cannot read data")
                    break
                elif state == FFmpegState.STOPPED:
                    logger.info("FFmpeg is stopped")
                    break
                elif state != FFmpegState.RUNNING:
                    await asyncio.sleep(0.1)
                    continue

                current_time = time()
                elapsed_time = max(0.0, current_time - beg)
                buffer_size = max(int(32000 * elapsed_time), 4096)  # dynamic read
                beg = current_time

                chunk = await self.ffmpeg_manager.read_data(buffer_size)
                if not chunk:
                    # No data currently available
                    await asyncio.sleep(0.05)
                    continue

                self.pcm_buffer.extend(chunk)
                await self.handle_pcm_data()

            except asyncio.CancelledError:
                logger.info("ffmpeg_stdout_reader cancelled.")
                break
            except Exception as e:
                logger.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.2)

        logger.info("FFmpeg stdout processing finished. Signaling downstream processors if needed.")
        if not self.diarization_before_transcription and self.transcription_queue:
            await self.transcription_queue.put(SENTINEL)
        if self.args.diarization and self.diarization_queue:
            await self.diarization_queue.put(SENTINEL)
        if self.online_translation:
            await self.translation_queue.put(SENTINEL)

    async def transcription_processor(self):
        """Process audio chunks for transcription."""
        cumulative_pcm_duration_stream_time = 0.0
        
        while True:
            try:
                item = await self.transcription_queue.get()
                if item is SENTINEL:
                    logger.debug("Transcription processor received sentinel. Finishing.")
                    self.transcription_queue.task_done()
                    break

                asr_internal_buffer_duration_s = len(getattr(self.online, 'audio_buffer', [])) / self.online.SAMPLING_RATE
                transcription_lag_s = max(0.0, time() - self.beg_loop - self.end_buffer)
                asr_processing_logs = f"internal_buffer={asr_internal_buffer_duration_s:.2f}s | lag={transcription_lag_s:.2f}s |"
                if type(item) is Silence:
                    asr_processing_logs += f" + Silence of = {item.duration:.2f}s"
                    if self.tokens:
                        asr_processing_logs += f" | last_end = {self.tokens[-1].end} |"
                    logger.info(asr_processing_logs)
                    cumulative_pcm_duration_stream_time += item.duration
                    self.online.insert_silence(item.duration, self.tokens[-1].end if self.tokens else 0)
                    continue
                elif isinstance(item, ChangeSpeaker):
                    self.online.new_speaker(item)
                elif isinstance(item, np.ndarray):
                    pcm_array = item
                
                logger.info(asr_processing_logs)
                
                duration_this_chunk = len(pcm_array) / self.sample_rate
                cumulative_pcm_duration_stream_time += duration_this_chunk
                stream_time_end_of_current_pcm = cumulative_pcm_duration_stream_time

                self.online.insert_audio_chunk(pcm_array, stream_time_end_of_current_pcm)
                new_tokens, current_audio_processed_upto = await asyncio.to_thread(self.online.process_iter)
                
                _buffer_transcript = self.online.get_buffer()
                buffer_text = _buffer_transcript.text

                if new_tokens:
                    validated_text = self.sep.join([t.text for t in new_tokens])
                    if buffer_text.startswith(validated_text):
                        _buffer_transcript.text = buffer_text[len(validated_text):].lstrip()

                candidate_end_times = [self.end_buffer]

                if new_tokens:
                    candidate_end_times.append(new_tokens[-1].end)
                
                if _buffer_transcript.end is not None:
                    candidate_end_times.append(_buffer_transcript.end)
                
                candidate_end_times.append(current_audio_processed_upto)
                
                async with self.lock:
                    self.tokens.extend(new_tokens)
                    self.buffer_transcription = _buffer_transcript
                    self.end_buffer = max(candidate_end_times)
                
                if self.translation_queue:
                    for token in new_tokens:
                        await self.translation_queue.put(token)
                        
                self.transcription_queue.task_done()
                
            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'pcm_array' in locals() and pcm_array is not SENTINEL : # Check if pcm_array was assigned from queue
                    self.transcription_queue.task_done()
        
        if self.is_stopping:
            logger.info("Transcription processor finishing due to stopping flag.")
            if self.diarization_queue:
                await self.diarization_queue.put(SENTINEL)
            if self.translation_queue:
                await self.translation_queue.put(SENTINEL)

        logger.info("Transcription processor task finished.")


    async def diarization_processor(self, diarization_obj):
        """Process audio chunks for speaker diarization."""
        self.current_speaker = 0
        while True:
            try:
                item = await self.diarization_queue.get()
                if item is SENTINEL:
                    logger.debug("Diarization processor received sentinel. Finishing.")
                    self.diarization_queue.task_done()
                    break
                elif type(item) is Silence:
                    diarization_obj.insert_silence(item.duration)
                    continue
                elif isinstance(item, np.ndarray):
                    pcm_array = item
                else:
                    raise Exception('item should be pcm_array') 
                
                # Process diarization
                await diarization_obj.diarize(pcm_array)
                segments = diarization_obj.get_segments()
                
                if self.diarization_before_transcription:
                    if segments and segments[-1].speaker != self.current_speaker:
                        self.current_speaker = segments[-1].speaker
                        cut_at = int(segments[-1].start*16000 - (self.cumulative_pcm_len))
                        await self.transcription_queue.put(pcm_array[cut_at:])
                        await self.transcription_queue.put(ChangeSpeaker(speaker=self.current_speaker, start=cut_at))                        
                        await self.transcription_queue.put(pcm_array[:cut_at])
                    else:
                        await self.transcription_queue.put(pcm_array)
                else:           
                    async with self.lock:
                        self.tokens = diarization_obj.assign_speakers_to_tokens(
                            self.tokens,
                            use_punctuation_split=self.args.punctuation_split
                        )
                self.cumulative_pcm_len += len(pcm_array)            
                if len(self.tokens) > 0:
                    self.end_attributed_speaker = max(self.tokens[-1].end, self.end_attributed_speaker)
                self.diarization_queue.task_done()
                
            except Exception as e:
                logger.warning(f"Exception in diarization_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'pcm_array' in locals() and pcm_array is not SENTINEL:
                    self.diarization_queue.task_done()
        logger.info("Diarization processor task finished.")

    async def translation_processor(self):
        # the idea is to ignore diarization for the moment. We use only transcription tokens. 
        # And the speaker is attributed given the segments used for the translation
        # in the future we want to have different languages for each speaker etc, so it will be more complex.
        while True:
            try:
                item = await self.translation_queue.get() #block until at least 1 token
                if item is SENTINEL:
                    logger.debug("Translation processor received sentinel. Finishing.")
                    self.translation_queue.task_done()
                    break
                elif type(item) is Silence:
                    self.online_translation.insert_silence(item.duration)
                    continue
                
                # get all the available tokens for translation. The more words, the more precise
                tokens_to_process = [item]
                additional_tokens = await get_all_from_queue(self.translation_queue)
                
                sentinel_found = False
                for additional_token in additional_tokens:
                    if additional_token is SENTINEL:
                        sentinel_found = True
                        break
                    tokens_to_process.append(additional_token)                
                if tokens_to_process:
                    self.online_translation.insert_tokens(tokens_to_process)
                    self.translated_segments = await asyncio.to_thread(self.online_translation.process)
                self.translation_queue.task_done()
                for _ in additional_tokens:
                    self.translation_queue.task_done()
                
                if sentinel_found:
                    logger.debug("Translation processor received sentinel in batch. Finishing.")
                    break
                
            except Exception as e:
                logger.warning(f"Exception in translation_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if 'token' in locals() and item is not SENTINEL:
                    self.translation_queue.task_done()
                if 'additional_tokens' in locals():
                    for _ in additional_tokens:
                        self.translation_queue.task_done()
        logger.info("Translation processor task finished.")

    async def results_formatter(self):
        """Format processing results for output."""
        while True:
            try:
                # If FFmpeg error occurred, notify front-end
                if self._ffmpeg_error:
                    yield FrontData(
                        status="error",
                        error=f"FFmpeg error: {self._ffmpeg_error}"
                    )
                    self._ffmpeg_error = None
                    await asyncio.sleep(1)
                    continue

                # Get current state
                state = await self.get_current_state()
                                
                # Add dummy tokens if needed
                if (not state.tokens or state.tokens[-1].is_dummy) and not self.args.transcription and self.args.diarization:
                    await self.add_dummy_token()
                    sleep(0.5)
                    state = await self.get_current_state()
                
                # Format output
                lines, undiarized_text, end_w_silence = format_output(
                    state,
                    self.silence,
                    current_time = time() - self.beg_loop if self.beg_loop else None,
                    args = self.args,
                    debug = self.debug,
                    sep=self.sep
                )
                if end_w_silence:
                    buffer_transcription = Transcript()
                else:
                    buffer_transcription = state.buffer_transcription

                buffer_diarization = ''
                if undiarized_text:
                    buffer_diarization = self.sep.join(undiarized_text)

                    async with self.lock:
                        self.end_attributed_speaker = state.end_attributed_speaker
                
                response_status = "active_transcription"
                if not state.tokens and not buffer_transcription and not buffer_diarization:
                    response_status = "no_audio_detected"
                    lines = []
                elif not lines:
                    lines = [Line(
                        speaker=1,
                        start=state.end_buffer,
                        end=state.end_buffer
                    )]
                
                response = FrontData(
                    status=response_status,
                    lines=lines,
                    buffer_transcription=buffer_transcription.text.strip(),
                    buffer_diarization=buffer_diarization.strip(),
                    remaining_time_transcription=state.remaining_time_transcription,
                    remaining_time_diarization=state.remaining_time_diarization if self.args.diarization else 0
                )
                                
                should_push = (response != self.last_response_content)
                if should_push and (lines or buffer_transcription or buffer_diarization or response_status == "no_audio_detected"):
                    yield response
                    self.last_response_content = response
                
                # Check for termination condition
                if self.is_stopping:
                    all_processors_done = True
                    if self.args.transcription and self.transcription_task and not self.transcription_task.done():
                        all_processors_done = False
                    if self.args.diarization and self.diarization_task and not self.diarization_task.done():
                        all_processors_done = False
                    
                    if all_processors_done:
                        logger.info("Results formatter: All upstream processors are done and in stopping state. Terminating.")
                        return
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.warning(f"Exception in results_formatter: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.5)
        
    async def create_tasks(self):
        """Create and start processing tasks."""
        self.all_tasks_for_cleanup = []
        processing_tasks_for_watchdog = []

        # If using FFmpeg (non-PCM input), start it and spawn stdout reader
        if not self.is_pcm_input:
            success = await self.ffmpeg_manager.start()
            if not success:
                logger.error("Failed to start FFmpeg manager")
                async def error_generator():
                    yield FrontData(
                        status="error",
                        error="FFmpeg failed to start. Please check that FFmpeg is installed."
                    )
                return error_generator()
            self.ffmpeg_reader_task = asyncio.create_task(self.ffmpeg_stdout_reader())
            self.all_tasks_for_cleanup.append(self.ffmpeg_reader_task)
            processing_tasks_for_watchdog.append(self.ffmpeg_reader_task)

        if self.args.transcription and self.online:
            self.transcription_task = asyncio.create_task(self.transcription_processor())
            self.all_tasks_for_cleanup.append(self.transcription_task)
            processing_tasks_for_watchdog.append(self.transcription_task)
            
        if self.args.diarization and self.diarization:
            self.diarization_task = asyncio.create_task(self.diarization_processor(self.diarization))
            self.all_tasks_for_cleanup.append(self.diarization_task)
            processing_tasks_for_watchdog.append(self.diarization_task)
        
        if self.online_translation:
            self.translation_task = asyncio.create_task(self.translation_processor())
            self.all_tasks_for_cleanup.append(self.translation_task)
            processing_tasks_for_watchdog.append(self.translation_task)
        
        # Monitor overall system health
        self.watchdog_task = asyncio.create_task(self.watchdog(processing_tasks_for_watchdog))
        self.all_tasks_for_cleanup.append(self.watchdog_task)
        
        return self.results_formatter()

    async def watchdog(self, tasks_to_monitor):
        """Monitors the health of critical processing tasks."""
        while True:
            try:
                await asyncio.sleep(10)
                
                for i, task in enumerate(tasks_to_monitor):
                    if task.done():
                        exc = task.exception()
                        task_name = task.get_name() if hasattr(task, 'get_name') else f"Monitored Task {i}"
                        if exc:
                            logger.error(f"{task_name} unexpectedly completed with exception: {exc}")
                        else:
                            logger.info(f"{task_name} completed normally.")
                    
            except asyncio.CancelledError:
                logger.info("Watchdog task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in watchdog task: {e}", exc_info=True)
        
    async def cleanup(self):
        """Clean up resources when processing is complete."""
        logger.info("Starting cleanup of AudioProcessor resources.")
        self.is_stopping = True
        for task in self.all_tasks_for_cleanup:
            if task and not task.done():
                task.cancel()
            
        created_tasks = [t for t in self.all_tasks_for_cleanup if t]
        if created_tasks:
            await asyncio.gather(*created_tasks, return_exceptions=True)
        logger.info("All processing tasks cancelled or finished.")

        if not self.is_pcm_input and self.ffmpeg_manager:
            try:
                await self.ffmpeg_manager.stop()
                logger.info("FFmpeg manager stopped.")
            except Exception as e:
                logger.warning(f"Error stopping FFmpeg manager: {e}")
        if self.args.diarization and hasattr(self, 'dianization') and hasattr(self.diarization, 'close'):
            self.diarization.close()
        logger.info("AudioProcessor cleanup complete.")


    async def process_audio(self, message):
        """Process incoming audio data."""

        if not self.beg_loop:
            self.beg_loop = time()

        if not message:
            logger.info("Empty audio message received, initiating stop sequence.")
            self.is_stopping = True
             
            if self.transcription_queue:
                await self.transcription_queue.put(SENTINEL)

            if not self.is_pcm_input and self.ffmpeg_manager:
                await self.ffmpeg_manager.stop()

            return

        if self.is_stopping:
            logger.warning("AudioProcessor is stopping. Ignoring incoming audio.")
            return

        if self.is_pcm_input:
            self.pcm_buffer.extend(message)
            await self.handle_pcm_data()
        else:
            if not self.ffmpeg_manager:
                logger.error("FFmpeg manager not initialized for non-PCM input.")
                return
            success = await self.ffmpeg_manager.write_data(message)
            if not success:
                ffmpeg_state = await self.ffmpeg_manager.get_state()
                if ffmpeg_state == FFmpegState.FAILED:
                    logger.error("FFmpeg is in FAILED state, cannot process audio")
                else:
                    logger.warning("Failed to write audio data to FFmpeg")

    async def handle_pcm_data(self):
        # Process when enough data
        if len(self.pcm_buffer) < self.bytes_per_sec:
            return

        if len(self.pcm_buffer) > self.max_bytes_per_sec:
            logger.warning(
                f"Audio buffer too large: {len(self.pcm_buffer) / self.bytes_per_sec:.2f}s. "
                f"Consider using a smaller model."
            )

        # Process audio chunk
        pcm_array = self.convert_pcm_to_float(self.pcm_buffer[:self.max_bytes_per_sec])
        self.pcm_buffer = self.pcm_buffer[self.max_bytes_per_sec:]

        res = None
        end_of_audio = False
        silence_buffer = None

        if self.args.vac:
            res = self.vac(pcm_array)

        if res is not None:
            if res.get("end", 0) > res.get("start", 0):
                end_of_audio = True
            elif self.silence: #end of silence
                self.silence = False
                silence_buffer = Silence(duration=time() - self.start_silence)

        if silence_buffer:
            if not self.diarization_before_transcription and self.transcription_queue:
                await self.transcription_queue.put(silence_buffer)
            if self.args.diarization and self.diarization_queue:
                await self.diarization_queue.put(silence_buffer)
            if self.translation_queue:
                await self.translation_queue.put(silence_buffer)

        if not self.silence:
            if not self.diarization_before_transcription and self.transcription_queue:
                await self.transcription_queue.put(pcm_array.copy())

            if self.args.diarization and self.diarization_queue:
                await self.diarization_queue.put(pcm_array.copy())

            self.silence_duration = 0.0

            if end_of_audio:
                self.silence = True
                self.start_silence = time()

        if not self.args.transcription and not self.args.diarization:
            await asyncio.sleep(0.1)
