import numpy as np
import torch
import logging
from whisperlivekit.timed_objects import SpeakerSegment

logger = logging.getLogger(__name__)

try:
    from nemo.collections.asr.models import SortformerEncLabelModel
except ImportError:
    raise SystemExit("""Please use `pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"` to use the Sortformer diarization""")