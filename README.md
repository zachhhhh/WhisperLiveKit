<h1 align="center">WhisperLiveKit</h1>
<p align="center"><b>Real-time, Fully Local Whisper's Speech-to-Text and Speaker Diarization</b></p>


This project is based on [Whisper Streaming](https://github.com/ufal/whisper_streaming) and lets you transcribe audio directly from your browser. Simply launch the local server and grant microphone access. Everything runs locally on your machine ✨

<p align="center">
  <img src="https://raw.githubusercontent.com/QuentinFuxa/WhisperLiveKit/refs/heads/main/demo.png" alt="Demo Screenshot" width="730">
</p>

### Differences from [Whisper Streaming](https://github.com/ufal/whisper_streaming)

#### ⚙️ **Core Improvements**  
- **Buffering Preview** – Displays unvalidated transcription segments
- **Multi-User Support** – Handles multiple users simultaneously by decoupling backend and online asr
- **MLX Whisper Backend** – Optimized for Apple Silicon for faster local processing.  
- **Confidence validation** – Immediately validate high-confidence tokens for faster inference

#### 🎙️ **Speaker Identification**  
- **Real-Time Diarization** – Identify different speakers in real time using [Diart](https://github.com/juanmc2005/diart)

#### 🌐 **Web & API**  
- **Built-in Web UI** – Simple raw html browser interface with no frontend setup required
- **FastAPI WebSocket Server** – Real-time speech-to-text processing with async FFmpeg streaming.  
- **JavaScript Client** – Ready-to-use MediaRecorder implementation for seamless client-side integration.

## Installation

### Via pip

```bash
pip install whisperlivekit
```

### From source

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/QuentinFuxa/WhisperLiveKit
   cd WhisperLiveKit
   pip install -e .
   ```

### System Dependencies

You need to install FFmpeg on your system:

- Install system dependencies:
    ```bash
    # Install FFmpeg on your system (required for audio processing)
    # For Ubuntu/Debian:
    sudo apt install ffmpeg
    
    # For macOS:
    brew install ffmpeg
    
    # For Windows:
    # Download from https://ffmpeg.org/download.html and add to PATH
    ```

- Install required Python dependencies:

    ```bash
    # Whisper streaming required dependencies
    pip install librosa soundfile

    # Whisper streaming web required dependencies
    pip install fastapi ffmpeg-python
    ```
- Install at least one whisper backend among:

    ```
   whisper
   whisper-timestamped
   faster-whisper (faster backend on NVIDIA GPU)
   mlx-whisper (faster backend on Apple Silicon)
   ```
- Optionnal dependencies

    ```
    # If you want to use VAC (Voice Activity Controller). Useful for preventing hallucinations
    torch
   
    # If you choose sentences as buffer trimming strategy
    mosestokenizer
    wtpsplit
    tokenize_uk # If you work with Ukrainian text

    # If you want to run the server using uvicorn (recommended)
    uvicorn

    # If you want to use diarization
    diart
    ```

    Diart uses by default [pyannote.audio](https://github.com/pyannote/pyannote-audio) models from the _huggingface hub_. To use them, please follow the steps described [here](https://github.com/juanmc2005/diart?tab=readme-ov-file#get-access-to--pyannote-models).


3. **Run the FastAPI Server**:

    ```bash
    python whisper_fastapi_online_server.py --host 0.0.0.0 --port 8000
    ```

    **Parameters**
   
    The following parameters are supported:
  
    - `--host` and `--port` let you specify the server's IP/port. 
    - `-min-chunk-size` sets the minimum chunk size for audio processing. Make sure this value aligns with the chunk size selected in the frontend. If not aligned, the system will work but may unnecessarily over-process audio data.
    - `--transcription`: Enable/disable transcription (default: True)
    - `--diarization`: Enable/disable speaker diarization (default: False)
    - `--confidence-validation`: Use confidence scores for faster validation. Transcription will be faster but punctuation might be less accurate (default: True)
    - `--warmup-file`: The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. :
      - If not set, uses https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav.
      - If False, no warmup is performed.
    - `--min-chunk-size` Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.
    - `--model` {_tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large, large-v3-turbo_}
                        Name size of the Whisper model to use (default: tiny). The model is automatically downloaded from the model hub if not present in model cache dir.
    - `--model_cache_dir` Overriding the default model cache dir where models downloaded from the hub are saved
    - `--model_dir` Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.
    - `--lan`, --language Source language code, e.g. en,de,cs, or 'auto' for language detection.
    - `--task` {_transcribe, translate_} Transcribe or translate. If translate is set, we recommend avoiding the _large-v3-turbo_ backend, as it [performs significantly worse](https://github.com/QuentinFuxa/whisper_streaming_web/issues/40#issuecomment-2652816533) than other models for translation.
    - `--backend` {_faster-whisper, whisper_timestamped, openai-api, mlx-whisper_} Load only this backend for Whisper processing.
    - `--vac` Use VAC = voice activity controller. Requires torch.
    - `--vac-chunk-size` VAC sample size in seconds.
    - `--vad` Use VAD = voice activity detection, with the default parameters.
    - `--buffer_trimming` {_sentence, segment_} Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.
    - `--buffer_trimming_sec` Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.

5. **Open the Provided HTML**:

    - By default, the server root endpoint `/` serves a simple `live_transcription.html` page.  
    - Open your browser at `http://localhost:8000` (or replace `localhost` and `8000` with whatever you specified).  
    - The page uses vanilla JavaScript and the WebSocket API to capture your microphone and stream audio to the server in real time.

### How the Live Interface Works

- Once you **allow microphone access**, the page records small chunks of audio using the **MediaRecorder** API in **webm/opus** format.  
- These chunks are sent over a **WebSocket** to the FastAPI endpoint at `/asr`.  
- The Python server decodes `.webm` chunks on the fly using **FFmpeg** and streams them into the **whisper streaming** implementation for transcription.  
- **Partial transcription** appears as soon as enough audio is processed. The “unvalidated” text is shown in **lighter or grey color** (i.e., an ‘aperçu’) to indicate it’s still buffered partial output. Once Whisper finalizes that segment, it’s displayed in normal text.  
- You can watch the transcription update in near real time, ideal for demos, prototyping, or quick debugging.

### Deploying to a Remote Server

If you want to **deploy** this setup:

1. **Host the FastAPI app** behind a production-grade HTTP(S) server (like **Uvicorn + Nginx** or Docker). If you use HTTPS, use "wss" instead of "ws" in WebSocket URL.
2. The **HTML/JS page** can be served by the same FastAPI app or a separate static host.  
3. Users open the page in **Chrome/Firefox** (any modern browser that supports MediaRecorder + WebSocket).  

No additional front-end libraries or frameworks are required. The WebSocket logic in `live_transcription.html` is minimal enough to adapt for your own custom UI or embed in other pages.

## Acknowledgments

This project builds upon the foundational work of the Whisper Streaming project. We extend our gratitude to the original authors for their contributions.
