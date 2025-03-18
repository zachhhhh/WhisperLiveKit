import asyncio
import logging
from time import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from timed_objects import ASRToken

logger = logging.getLogger(__name__)


class SharedState:
    """
    Thread-safe state manager for streaming transcription and diarization.
    Handles coordination between audio processing, transcription, and diarization.
    """
    
    def __init__(self):
        self.tokens: List[ASRToken] = []
        self.buffer_transcription: str = ""
        self.buffer_diarization: str = ""
        self.full_transcription: str = ""
        self.end_buffer: float = 0
        self.end_attributed_speaker: float = 0
        self.lock = asyncio.Lock()
        self.beg_loop: float = time()
        self.sep: str = " "  # Default separator
        self.last_response_content: str = ""  # To track changes in response
        
    async def update_transcription(self, new_tokens: List[ASRToken], buffer: str, 
                                  end_buffer: float, full_transcription: str, sep: str) -> None:
        """Update the state with new transcription data."""
        async with self.lock:
            self.tokens.extend(new_tokens)
            self.buffer_transcription = buffer
            self.end_buffer = end_buffer
            self.full_transcription = full_transcription
            self.sep = sep
            
    async def update_diarization(self, end_attributed_speaker: float, buffer_diarization: str = "") -> None:
        """Update the state with new diarization data."""
        async with self.lock:
            self.end_attributed_speaker = end_attributed_speaker
            if buffer_diarization:
                self.buffer_diarization = buffer_diarization
            
    async def add_dummy_token(self) -> None:
        """Add a dummy token to keep the state updated even without transcription."""
        async with self.lock:
            current_time = time() - self.beg_loop
            dummy_token = ASRToken(
                start=current_time,
                end=current_time + 1,
                text=".",
                speaker=-1,
                is_dummy=True
            )
            self.tokens.append(dummy_token)
            
    async def get_current_state(self) -> Dict[str, Any]:
        """Get the current state with calculated timing information."""
        async with self.lock:
            current_time = time()
            remaining_time_transcription = 0
            remaining_time_diarization = 0
            
            # Calculate remaining time for transcription buffer
            if self.end_buffer > 0:
                remaining_time_transcription = max(0, round(current_time - self.beg_loop - self.end_buffer, 2))
                
            # Calculate remaining time for diarization
            if self.tokens:
                latest_end = max(self.end_buffer, self.tokens[-1].end if self.tokens else 0)
                remaining_time_diarization = max(0, round(latest_end - self.end_attributed_speaker, 2))
                
            return {
                "tokens": self.tokens.copy(),
                "buffer_transcription": self.buffer_transcription,
                "buffer_diarization": self.buffer_diarization,
                "end_buffer": self.end_buffer,
                "end_attributed_speaker": self.end_attributed_speaker,
                "sep": self.sep,
                "remaining_time_transcription": remaining_time_transcription,
                "remaining_time_diarization": remaining_time_diarization
            }
            
    async def reset(self) -> None:
        """Reset the state to initial values."""
        async with self.lock:
            self.tokens = []
            self.buffer_transcription = ""
            self.buffer_diarization = ""
            self.end_buffer = 0
            self.end_attributed_speaker = 0
            self.full_transcription = ""
            self.beg_loop = time()
            self.last_response_content = ""