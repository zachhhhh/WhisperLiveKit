import asyncio
import ffmpeg
import logging
import time
import traceback
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

logger = logging.getLogger(__name__)

class FFmpegState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    RESTARTING = "restarting"
    FAILED = "failed"

class CircuitBreaker:
    """to prevent endless restart loops."""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.is_open = False
    
    def record_success(self):
        self.failure_count = 0
        self.is_open = False
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.error(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_attempt(self) -> bool:
        if not self.is_open:
            return True
        
        if time.time() - self.last_failure_time > self.recovery_timeout:
            logger.info("Circuit breaker recovery timeout reached, attempting reset")
            self.is_open = False
            self.failure_count = 0
            return True
        
        return False

class FFmpegManager:
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, 
                 max_retries: int = 3, restart_delay: float = 1.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_retries = max_retries
        self.restart_delay = restart_delay
        
        self.process: Optional[object] = None
        self.state = FFmpegState.STOPPED
        self.state_lock = asyncio.Lock()
        
        self.circuit_breaker = CircuitBreaker()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ffmpeg")
        
        self._write_queue = asyncio.Queue(maxsize=100)
        self._write_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        
        self.last_activity = time.time()
        self.on_data_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        
        self._restart_lock = asyncio.Lock()
        self._restart_in_progress = False
        
    async def start(self):
        async with self.state_lock:
            if self.state != FFmpegState.STOPPED:
                logger.warning(f"Cannot start FFmpeg, current state: {self.state}")
                return False
            
            self.state = FFmpegState.STARTING
        
        try:
            if not self.circuit_breaker.can_attempt():
                logger.error("CB is open, cannot start FFmpeg")
                async with self.state_lock:
                    self.state = FFmpegState.FAILED
                return False
            
            # Start FFmpeg process
            success = await self._start_process()
            if not success:
                self.circuit_breaker.record_failure()
                async with self.state_lock:
                    self.state = FFmpegState.FAILED
                return False
            
            self.circuit_breaker.record_success()
            
            # Start background tasks
            self._write_task = asyncio.create_task(self._write_worker())
            self._monitor_task = asyncio.create_task(self._monitor_health())
            
            async with self.state_lock:
                self.state = FFmpegState.RUNNING
            
            logger.info("FFmpeg manager started ok")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start FFmpeg manager: {e}")
            logger.error(traceback.format_exc())
            self.circuit_breaker.record_failure()
            async with self.state_lock:
                self.state = FFmpegState.FAILED
            return False
    
    async def stop(self):
        logger.info("Stopping FFmpeg manager")
        
        async with self.state_lock:
            if self.state == FFmpegState.STOPPED:
                return
            self.state = FFmpegState.STOPPED
        
        # Cancel background tasks
        if self._write_task and not self._write_task.done():
            self._write_task.cancel()
            try:
                await self._write_task
            except asyncio.CancelledError:
                pass
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        await self._stop_process()
        while not self._write_queue.empty():
            try:
                self._write_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        logger.info("FFmpeg manager stopped")
    
    async def write_data(self, data: bytes) -> bool:
        current_state = await self.get_state()
        if current_state != FFmpegState.RUNNING:
            logger.warning(f"Cannot write data, FFmpeg state: {current_state}")
            return False
        
        try:
            # Use nowait to avoid blocking
            self._write_queue.put_nowait(data)
            return True
        except asyncio.QueueFull:
            logger.warning("FFmpeg write queue is full, dropping data")
            if self.on_error_callback:
                await self.on_error_callback("write_queue_full")
            return False
    
    async def read_data(self, size: int) -> Optional[bytes]:
        """Read data from FFmpeg stdout (non-blocking)."""
        if not self.process or self.process.poll() is not None:
            return None
        
        try:
            loop = asyncio.get_event_loop()
            data = await asyncio.wait_for(
                loop.run_in_executor(self.executor, self.process.stdout.read, size),
                timeout=5.0
            )
            
            if data:
                self.last_activity = time.time()
            
            return data
            
        except asyncio.TimeoutError:
            logger.warning("FFmpeg read timeout")
            return None
        except Exception as e:
            logger.error(f"Error reading from FFmpeg: {e}")
            return None
    
    async def get_state(self) -> FFmpegState:
        """Get the current FFmpeg state."""
        async with self.state_lock:
            return self.state
    
    async def restart(self, is_external_kill: bool = False) -> bool:
        """Restart the FFmpeg process."""
        async with self._restart_lock:
            if self._restart_in_progress:
                logger.info("Restart already in progress, skipping")
                return False
            
            self._restart_in_progress = True
        
        try:
            logger.warning("Restarting FFmpeg process")
            
            async with self.state_lock:
                if self.state == FFmpegState.RESTARTING:
                    logger.warning("FFmpeg is already restarting")
                    return False
                
                prev_state = self.state
                self.state = FFmpegState.RESTARTING
            
            if is_external_kill:
                logger.warning("External kill detected, resetting circuit breaker. (Check the exit code) ")
                self.circuit_breaker.failure_count = 0
                self.circuit_breaker.is_open = False
            
            logger.warning("Clearing write queue")
            while not self._write_queue.empty():
                try:
                    self._write_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            await self._stop_process()
            await asyncio.sleep(self.restart_delay)            
            for attempt in range(self.max_retries):
                if not self.circuit_breaker.can_attempt():
                    logger.error("Circuit breaker is open, cannot restart")
                    async with self.state_lock:
                        self.state = FFmpegState.FAILED
                    return False
                
                success = await self._start_process()
                if success:
                    self.circuit_breaker.record_success()
                    async with self.state_lock:
                        self.state = FFmpegState.RUNNING
                    logger.info("FFmpeg restarted successfully")
                    return True
                
                self.circuit_breaker.record_failure()
                
                if attempt < self.max_retries - 1:
                    delay = self.restart_delay * (2 ** attempt) # test 
                    logger.warning(f"Restart attempt {attempt + 1} failed, waiting {delay}s")
                    await asyncio.sleep(delay)
            
            logger.error("Failed to restart FFmpeg after all attempts")
            async with self.state_lock:
                self.state = FFmpegState.FAILED
            
            if self.on_error_callback:
                await self.on_error_callback("restart_failed")
            
            return False
            
        finally:
            async with self._restart_lock:
                self._restart_in_progress = False
    
    async def _start_process(self) -> bool:
        """Start the FFmpeg process."""
        try:
            self.process = (
                ffmpeg.input("pipe:0", format="webm")
                .output("pipe:1", format="s16le", acodec="pcm_s16le", 
                       ac=self.channels, ar=str(self.sample_rate))
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
            )
            
            await asyncio.sleep(0.1)
            if self.process.poll() is not None:
                logger.error("FFmpeg process died immediately after starting")
                return False
            
            self.last_activity = time.time()
            logger.info("FFmpeg process started successfully")
            return True
            
        except FileNotFoundError:
            error = """
            FFmpeg is not installed or not found in your system's PATH.
            Please install FFmpeg to enable audio processing.

            Installation instructions:

            # Ubuntu/Debian:
            sudo apt update && sudo apt install ffmpeg

            # macOS (using Homebrew):
            brew install ffmpeg

            # Windows:
            # 1. Download the latest static build from https://ffmpeg.org/download.html
            # 2. Extract the archive (e.g., to C:\\FFmpeg).
            # 3. Add the 'bin' directory (e.g., C:\\FFmpeg\\bin) to your system's PATH environment variable.

            After installation, please restart the application.
            """
            logger.error(error)
            return False
        except Exception as e:
            logger.error(f"Failed to start FFmpeg process: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def _stop_process(self):
        """Stop the FFmpeg process gracefully."""
        if not self.process:
            return
        
        try:
            # Close stdin
            if self.process.stdin and not self.process.stdin.closed:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, self.process.stdin.close)
            
            # Wait for process to terminate
            if self.process.poll() is None:
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self.executor, self.process.wait
                        ),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg did not terminate gracefully, killing")
                    self.process.kill()
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor, self.process.wait
                    )
            
            logger.info("FFmpeg process stopped")
            
        except Exception as e:
            logger.error(f"Error stopping FFmpeg process: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.process = None
    
    async def _write_worker(self):
        """Background worker to write data to FFmpeg."""
        logger.info("FFmpeg write worker started")
        consecutive_failures = 0
        
        while True:
            try:
                # Get data from queue
                data = await self._write_queue.get()
                
                # Check if we're in a restart state
                current_state = await self.get_state()
                if current_state == FFmpegState.RESTARTING:
                    # Drop data during restart
                    continue
                
                if not self.process or self.process.poll() is not None:
                    consecutive_failures += 1
                    
                    # If we have multiple consecutive failures, it might be an external kill
                    if consecutive_failures > 3:
                        logger.warning("Multiple write failures detected, possible external kill")
                        if self.on_error_callback:
                            await self.on_error_callback("process_not_available")
                        
                        # Let the health monitor handle the restart
                        await asyncio.sleep(0.5)
                    continue
                
                # Write data in executor to avoid blocking
                loop = asyncio.get_event_loop()
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(
                            self.executor, 
                            self._write_to_process, 
                            data
                        ),
                        timeout=5.0
                    )
                    self.last_activity = time.time()
                    consecutive_failures = 0  # Reset on success
                    
                except asyncio.TimeoutError:
                    logger.error("FFmpeg write timeout")
                    if self.on_error_callback:
                        await self.on_error_callback("write_timeout")
                    # Trigger restart only if not already restarting
                    if await self.get_state() != FFmpegState.RESTARTING:
                        asyncio.create_task(self.restart())
                    
                except Exception as e:
                    logger.error(f"Error writing to FFmpeg: {e}")
                    if self.on_error_callback:
                        await self.on_error_callback("write_error")
                    # Trigger restart only if not already restarting
                    if await self.get_state() != FFmpegState.RESTARTING:
                        asyncio.create_task(self.restart())
                
            except asyncio.CancelledError:
                logger.info("FFmpeg write worker cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in write worker: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)
    
    def _write_to_process(self, data: bytes):
        if self.process and self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.write(data)
            self.process.stdin.flush()
    
    async def _monitor_health(self):
        logger.info("FFmpeg health monitor started")
        last_known_pid = None
        
        while True:
            try:
                await asyncio.sleep(5)
                
                current_state = await self.get_state()
                if current_state not in [FFmpegState.RUNNING, FFmpegState.RESTARTING]:
                    continue
                
                if not self.process or self.process.poll() is not None:
                    if current_state != FFmpegState.RESTARTING:
                        logger.error(f"FFmpeg process died unexpectedly with exit code: {self.process.poll()}")
                        is_external_kill = last_known_pid is not None
                        
                        if is_external_kill:
                            logger.info("Detected possible external kill (e.g., pkill)")
                        
                        if self.on_error_callback:
                            await self.on_error_callback("process_died")
                        await self.restart(is_external_kill=is_external_kill)
                    continue
                
                if self.process and hasattr(self.process, 'pid'):
                    last_known_pid = self.process.pid
                idle_time = time.time() - self.last_activity
                if idle_time > 30:
                    logger.warning(f"FFmpeg idle for {idle_time:.1f}s")
                    if idle_time > 60 and current_state != FFmpegState.RESTARTING:
                        logger.error("FFmpeg idle timeout, restarting")
                        if self.on_error_callback:
                            await self.on_error_callback("idle_timeout")
                        await self.restart()
                
            except asyncio.CancelledError:
                logger.info("FFmpeg health monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)
    
    def __del__(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=False)
