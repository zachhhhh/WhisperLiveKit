import asyncio
import re
import threading
import numpy as np

from diart import SpeakerDiarization
from diart.inference import StreamingInference
from diart.sources import AudioSource


def extract_number(s: str) -> int:
    m = re.search(r'\d+', s)
    return int(m.group()) if m else None


class WebSocketAudioSource(AudioSource):
    """
    Custom AudioSource that blocks in read() until close() is called.
    Use push_audio() to inject PCM chunks.
    """
    def __init__(self, uri: str = "websocket", sample_rate: int = 16000):
        super().__init__(uri, sample_rate)
        self._closed = False
        self._close_event = threading.Event()

    def read(self):
        self._close_event.wait()

    def close(self):
        if not self._closed:
            self._closed = True
            self.stream.on_completed()
            self._close_event.set()

    def push_audio(self, chunk: np.ndarray):
        if not self._closed:
            self.stream.on_next(np.expand_dims(chunk, axis=0))


class DiartDiarization:
    def __init__(self, sample_rate: int):
        self.processed_time = 0
        self.segment_speakers = []
        self.speakers_queue = asyncio.Queue()
        self.pipeline = SpeakerDiarization()
        self.source = WebSocketAudioSource(uri="websocket_source", sample_rate=sample_rate)
        self.inference = StreamingInference(
            pipeline=self.pipeline,
            source=self.source,
            do_plot=False,
            show_progress=False,
        )
        # Attache la fonction hook et démarre l'inférence en arrière-plan.
        self.inference.attach_hooks(self._diar_hook)
        asyncio.get_event_loop().run_in_executor(None, self.inference)

    def _diar_hook(self, result):
        annotation, audio = result
        if annotation._labels:
            for speaker, label in annotation._labels.items():
                beg = label.segments_boundaries_[0]
                end = label.segments_boundaries_[-1]
                if end > self.processed_time:
                    self.processed_time = end
                asyncio.create_task(self.speakers_queue.put({
                    "speaker": speaker,
                    "beg": beg,
                    "end": end
                }))
        else:
            dur = audio.extent.end
            if dur > self.processed_time:
                self.processed_time = dur

    async def diarize(self, pcm_array: np.ndarray):
        self.source.push_audio(pcm_array)
        self.segment_speakers.clear()
        while not self.speakers_queue.empty():
            self.segment_speakers.append(await self.speakers_queue.get())

    def close(self):
        self.source.close()

    def assign_speakers_to_tokens(self, end_attributed_speaker, tokens: list) -> list:
        for token in tokens:
            for segment in self.segment_speakers:
                if not (segment["end"] <= token.start or segment["beg"] >= token.end):
                    token.speaker = extract_number(segment["speaker"]) + 1
                    end_attributed_speaker = max(token.end, end_attributed_speaker)
        return end_attributed_speaker
