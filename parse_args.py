
import argparse
from whisper_streaming_custom.whisper_online import add_shared_args


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper FastAPI Online Server")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="The host address to bind the server to.",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="The port number to bind the server to."
    )
    parser.add_argument(
        "--warmup-file",
        type=str,
        default=None,
        dest="warmup_file",
        help="""
        The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast.
        If not set, uses https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav.
        If False, no warmup is performed.
        """,
    )

    parser.add_argument(
        "--confidence-validation",
        type=bool,
        default=False,
        help="Accelerates validation of tokens using confidence scores. Transcription will be faster but punctuation might be less accurate.",
    )

    parser.add_argument(
        "--diarization",
        type=bool,
        default=True,
        help="Whether to enable speaker diarization.",
    )

    parser.add_argument(
        "--transcription",
        type=bool,
        default=True,
        help="To disable to only see live diarization results.",
    )

    add_shared_args(parser)
    args = parser.parse_args()
    return args