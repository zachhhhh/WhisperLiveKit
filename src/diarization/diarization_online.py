from diart import SpeakerDiarization
from diart.inference import StreamingInference
from diart.sources import AudioSource
from rx.subject import Subject
import threading
import numpy as np
import asyncio

class WebSocketAudioSource(AudioSource):
    """
    Simple custom AudioSource that blocks in read()
    until close() is called.
    push_audio() is used to inject new PCM chunks.
    """
    def __init__(self, uri: str = "websocket", sample_rate: int = 16000):
        super().__init__(uri, sample_rate)
        self._close_event = threading.Event()
        self._closed = False

    def read(self):
        self._close_event.wait()

    def close(self):
        if not self._closed:
            self._closed = True
            self.stream.on_completed()
            self._close_event.set()

    def push_audio(self, chunk: np.ndarray):
        chunk = np.expand_dims(chunk, axis=0)
        if not self._closed:
            self.stream.on_next(chunk)


def create_pipeline(SAMPLE_RATE):
    diar_pipeline = SpeakerDiarization()
    ws_source = WebSocketAudioSource(uri="websocket_source", sample_rate=SAMPLE_RATE)
    inference = StreamingInference(
        pipeline=diar_pipeline,
        source=ws_source,
        do_plot=False,
        show_progress=False,
    )
    return inference, ws_source


def init_diart(SAMPLE_RATE):
    inference, ws_source = create_pipeline(SAMPLE_RATE)

    def diar_hook(result):
        """
        Hook called each time Diart processes a chunk.
        result is (annotation, audio).
        We store the label of the last segment in 'current_speaker'.
        """
        global l_speakers
        l_speakers = []
        annotation, audio = result
        for speaker in annotation._labels:            
            segment = annotation._labels[speaker].__str__()
            asyncio.create_task(
            l_speakers_queue.put({"speaker": speaker, "segment": segment})
        )

    l_speakers_queue = asyncio.Queue()
    inference.attach_hooks(diar_hook)

    # Launch Diart in a background thread
    loop = asyncio.get_event_loop()
    diar_future = loop.run_in_executor(None, inference)
    return inference, l_speakers_queue, ws_source


class DiartDiarization():
    def __init__(self, SAMPLE_RATE):
        self.inference, self.l_speakers_queue, self.ws_source = init_diart(SAMPLE_RATE)

    async def get_speakers(self, pcm_array):
        self.ws_source.push_audio(pcm_array)
        speakers = []
        while not self.l_speakers_queue.empty():
            speakers.append(await self.l_speakers_queue.get())
        return speakers
    
    def close(self):
        self.ws_source.close()
