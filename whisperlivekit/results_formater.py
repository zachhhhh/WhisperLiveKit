
import logging
from datetime import timedelta
from whisperlivekit.remove_silences import handle_silences

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PUNCTUATION_MARKS = {'.', '!', '?'}

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def check_punctuation_nearby(i, tokens):
    if i < len(tokens):
        for ind in range(i, min(len(tokens), i+1)): #we check in the next 1 tokens
            if tokens[ind].text.strip() in PUNCTUATION_MARKS:
                return True
    return False
    
    

def format_output(state, silence, current_time, diarization, debug):
    tokens = state["tokens"]
    buffer_transcription = state["buffer_transcription"]
    buffer_diarization = state["buffer_diarization"]
    end_attributed_speaker = state["end_attributed_speaker"]
    sep = state["sep"]
    
    previous_speaker = -1
    lines = []
    last_end_diarized = 0
    undiarized_text = []
    tokens, buffer_transcription, buffer_diarization = handle_silences(tokens, buffer_transcription, buffer_diarization, current_time, silence)
    for i, token in enumerate(tokens):
        speaker = token.speaker
        
        if len(tokens) == 1 and not diarization:
            if speaker == -1: #Speaker -1 means no attributed by diarization. In the frontend, it should appear under 'Speaker 1'
                speaker = 1
        
        if diarization and not tokens[-1].speaker == -2:
            if (speaker in [-1, 0]) and token.end >= end_attributed_speaker:
                undiarized_text.append(token.text)
                continue
            elif (speaker in [-1, 0]) and token.end < end_attributed_speaker:
                speaker = previous_speaker
            if speaker not in [-1, 0]:
                last_end_diarized = max(token.end, last_end_diarized)

        debug_info = ""
        if debug:
            debug_info = f"[{format_time(token.start)} : {format_time(token.end)}]"
        if speaker != previous_speaker or not lines:
            if speaker != previous_speaker and lines and check_punctuation_nearby(i, tokens): # check if punctuation nearby
                    lines[-1]["text"] += sep + token.text + debug_info
                    lines[-1]["end"] = format_time(token.end)
                    lines[-1]["diff"] = round(token.end - last_end_diarized, 2)
            else:
                lines.append({
                    "speaker": int(speaker),
                    "text": token.text + debug_info,
                    "beg": format_time(token.start),
                    "end": format_time(token.end),
                    "diff": round(token.end - last_end_diarized, 2)
                })
                previous_speaker = speaker
        elif token.text:  # Only append if text isn't empty
            lines[-1]["text"] += sep + token.text + debug_info
            lines[-1]["end"] = format_time(token.end)
            lines[-1]["diff"] = round(token.end - last_end_diarized, 2)

    return lines, undiarized_text, buffer_transcription, ''