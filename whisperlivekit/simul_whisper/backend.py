import sys
import numpy as np
import logging
from typing import List, Tuple, Optional
import logging
from whisperlivekit.timed_objects import ASRToken, Transcript
from whisperlivekit.simul_whisper.license_simulstreaming import SIMULSTREAMING_LICENSE
logger = logging.getLogger(__name__)

try:
    import torch
    from whisperlivekit.simul_whisper.config import AlignAttConfig
    from whisperlivekit.simul_whisper.simul_whisper import PaddedAlignAttWhisper
    from whisperlivekit.simul_whisper.whisper import tokenizer
except ImportError as e:
    raise ImportError(
        """SimulStreaming dependencies are not available.
        Please install WhisperLiveKit using pip install "whisperlivekit[simulstreaming]".""")

class SimulStreamingOnlineProcessor:
    SAMPLING_RATE = 16000

    def __init__(
        self,
        asr,
        tokenize_method: Optional[callable] = None,
        buffer_trimming: Tuple[str, float] = ("segment", 15),
        confidence_validation = False,
        logfile=sys.stderr,
    ):        
        self.asr = asr
        self.logfile = logfile
        self.confidence_validation = confidence_validation
        # buffer does not work yet
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming
        self.audio_chunks = []
        self.offset = 0.0
        self.is_last = False
        self.beg = self.offset
        self.end = self.offset
        self.cumulative_audio_duration = 0.0
        self.last_audio_stream_end_time = self.offset
        
        self.committed: List[ASRToken] = []
        self.last_result_tokens: List[ASRToken] = []        
        self.buffer_content = ""
        self.processed_audio_duration = 0.0

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: Optional[float] = None):
        """Append an audio chunk to be processed by SimulStreaming."""
            
        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        self.audio_chunks.append(audio_tensor)
        
        # Update timing
        chunk_duration = len(audio) / self.SAMPLING_RATE
        self.cumulative_audio_duration += chunk_duration
        
        if audio_stream_end_time is not None:
            self.last_audio_stream_end_time = audio_stream_end_time
            self.end = audio_stream_end_time
        else:
            self.end = self.offset + self.cumulative_audio_duration

    def get_buffer(self):
        """
        Get the unvalidated buffer content.
        """
        buffer_end = self.end if hasattr(self, 'end') else None
        return Transcript(
            start=None, 
            end=buffer_end, 
            text=self.buffer_content, 
            probability=None
        )

    def timestamped_text(self, tokens, generation):
        # From the simulstreaming repo. self.model to self.asr.model
        pr = generation["progress"]
        if "result" not in generation:
            split_words, split_tokens = self.asr.model.tokenizer.split_to_word_tokens(tokens)
        else:
            split_words, split_tokens = generation["result"]["split_words"], generation["result"]["split_tokens"]

        frames = [p["most_attended_frames"][0] for p in pr]
        tokens = tokens.copy()
        ret = []
        for sw,st in zip(split_words,split_tokens):
            b = None
            for stt in st:
                t,f = tokens.pop(0), frames.pop(0)
                if t != stt:
                    raise ValueError(f"Token mismatch: {t} != {stt} at frame {f}.")
                if b is None:
                    b = f
            e = f
            out = (b*0.02, e*0.02, sw)
            ret.append(out)
            logger.debug(f"TS-WORD:\t{' '.join(map(str, out))}")
        return ret

    def process_iter(self) -> Tuple[List[ASRToken], float]:
        """
        Process accumulated audio chunks using SimulStreaming.
        
        Returns a tuple: (list of committed ASRToken objects, float representing the audio processed up to time).
        """
        if not self.audio_chunks:
            return [], self.end

        try:
            # concatenate all audio chunks
            if len(self.audio_chunks) == 1:
                audio = self.audio_chunks[0]
            else:
                audio = torch.cat(self.audio_chunks, dim=0)
            
            audio_duration = audio.shape[0] / self.SAMPLING_RATE if audio.shape[0] > 0 else 0
            self.processed_audio_duration += audio_duration
            
            self.audio_chunks = []
            
            logger.debug(f"SimulStreaming processing audio shape: {audio.shape}, duration: {audio_duration:.2f}s")
            logger.debug(f"Current end time: {self.end:.2f}s, last stream time: {self.last_audio_stream_end_time:.2f}s")
            
            self.asr.model.insert_audio(audio)
            tokens, generation_progress = self.asr.model.infer(is_last=self.is_last)
            ts_words = self.timestamped_text(tokens, generation_progress)
            
            new_tokens = []
            for ts_word in ts_words:
                
                start, end, word = ts_word
                token = ASRToken(
                    start=start,
                    end=end,
                    text=word,
                    probability=0.95  # fake prob. Maybe we can extract it from the model?
                )
                new_tokens.append(token)
                self.committed.extend(new_tokens)
            
            return new_tokens, self.end

            
        except Exception as e:
            logger.exception(f"SimulStreaming processing error: {e}")
            return [], self.end

class SimulStreamingASR():
    """SimulStreaming backend with AlignAtt policy."""
    sep = ""

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr, **kwargs):
        logger.warning(SIMULSTREAMING_LICENSE)
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None if lan == "auto" else lan
        
        self.model_path = kwargs.get('model_path', './large-v3.pt')
        self.frame_threshold = kwargs.get('frame_threshold', 25)
        self.audio_max_len = kwargs.get('audio_max_len', 30.0)
        self.audio_min_len = kwargs.get('audio_min_len', 0.0)
        self.segment_length = kwargs.get('segment_length', 0.5)
        self.beams = kwargs.get('beams', 1)
        self.decoder_type = kwargs.get('decoder_type', 'greedy' if self.beams == 1 else 'beam')
        self.task = kwargs.get('task', 'transcribe')
        self.cif_ckpt_path = kwargs.get('cif_ckpt_path', None)
        self.never_fire = kwargs.get('never_fire', False)
        self.init_prompt = kwargs.get('init_prompt', None)
        self.static_init_prompt = kwargs.get('static_init_prompt', None)
        self.max_context_tokens = kwargs.get('max_context_tokens', None)
        
        if model_dir is not None:
            self.model_path = model_dir
        elif modelsize is not None:
            model_mapping = {
                'tiny': './tiny.pt',
                'base': './base.pt',
                'small': './small.pt',
                'medium': './medium.pt',
                'medium.en': './medium.en.pt',
                'large-v1': './large-v1.pt',
                'base.en': './base.en.pt',
                'small.en': './small.en.pt',
                'tiny.en': './tiny.en.pt',
                'large-v2': './large-v2.pt',
                'large-v3': './large-v3.pt',
                'large': './large-v3.pt'
            }
            self.model_path = model_mapping.get(modelsize, f'./{modelsize}.pt')
        
        self.model = self.load_model(modelsize, cache_dir, model_dir)
        
        # Set up tokenizer for translation if needed
        if self.task == "translate":
            self.set_translate_task()

    def load_model(self, modelsize, cache_dir, model_dir):
        try:
            cfg = AlignAttConfig(
                model_path=self.model_path,
                segment_length=self.segment_length,
                frame_threshold=self.frame_threshold,
                language=self.original_language,
                audio_max_len=self.audio_max_len,
                audio_min_len=self.audio_min_len,
                cif_ckpt_path=self.cif_ckpt_path,
                decoder_type="beam",
                beam_size=self.beams,
                task=self.task,
                never_fire=self.never_fire,
                init_prompt=self.init_prompt,
                max_context_tokens=self.max_context_tokens,
                static_init_prompt=self.static_init_prompt,
            )
            
            logger.info(f"Loading SimulStreaming model with language: {self.original_language}")
            model = PaddedAlignAttWhisper(cfg)
            return model
            
        except Exception as e:
            logger.error(f"Failed to load SimulStreaming model: {e}")
            raise

    def set_translate_task(self):
        """Set up translation task."""
        try:
            self.model.tokenizer = tokenizer.get_tokenizer(
                multilingual=True,
                language=self.model.cfg.language,
                num_languages=self.model.model.num_languages,
                task="translate"
            )
            logger.info("SimulStreaming configured for translation task")
        except Exception as e:
            logger.error(f"Failed to configure SimulStreaming for translation: {e}")
            raise

    def warmup(self, audio, init_prompt=""):
        """Warmup the SimulStreaming model."""
        try:
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            self.model.insert_audio(audio)
            self.model.infer(True)
            self.model.refresh_segment(complete=True)
            logger.info("SimulStreaming model warmed up successfully")
        except Exception as e:
            logger.exception(f"SimulStreaming warmup failed: {e}")
