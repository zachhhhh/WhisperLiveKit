from typing import Dict, Any, List
from datetime import timedelta

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

def format_response(state: Dict[str, Any], with_diarization: bool = False) -> Dict[str, Any]:
    """
    Format the shared state into a client-friendly response.
    
    Args:
        state: Current shared state dictionary
        with_diarization: Whether to include diarization formatting
        
    Returns:
        Formatted response dictionary ready to send to client
    """
    tokens = state["tokens"]
    buffer_transcription = state["buffer_transcription"]
    buffer_diarization = state["buffer_diarization"]
    end_attributed_speaker = state["end_attributed_speaker"]
    remaining_time_transcription = state["remaining_time_transcription"]
    remaining_time_diarization = state["remaining_time_diarization"]
    sep = state["sep"]
    
    # Default response for empty state
    if not tokens:
        return {
            "lines": [{
                "speaker": 1,
                "text": "",
                "beg": format_time(0),
                "end": format_time(0),
                "diff": 0
            }],
            "buffer_transcription": buffer_transcription,
            "buffer_diarization": buffer_diarization,
            "remaining_time_transcription": remaining_time_transcription,
            "remaining_time_diarization": remaining_time_diarization
        }
    
    # Process tokens to create response
    previous_speaker = -1
    lines = []
    last_end_diarized = 0
    undiarized_text = []
    
    for token in tokens:
        speaker = token.speaker
        
        # Handle diarization logic
        if with_diarization:
            if (speaker == -1 or speaker == 0) and token.end >= end_attributed_speaker:
                undiarized_text.append(token.text)
                continue
            elif (speaker == -1 or speaker == 0) and token.end < end_attributed_speaker:
                speaker = previous_speaker
            
            if speaker not in [-1, 0]:
                last_end_diarized = max(token.end, last_end_diarized)

        # Add new line or append to existing line
        if speaker != previous_speaker or not lines:
            lines.append({
                "speaker": speaker,
                "text": token.text,
                "beg": format_time(token.start),
                "end": format_time(token.end),
                "diff": round(token.end - last_end_diarized, 2)
            })
            previous_speaker = speaker
        elif token.text:  # Only append if text isn't empty
            lines[-1]["text"] += sep + token.text
            lines[-1]["end"] = format_time(token.end)
            lines[-1]["diff"] = round(token.end - last_end_diarized, 2)
    
    # If we have undiarized text, include it in the buffer
    if undiarized_text:
        combined_buffer = sep.join(undiarized_text)
        if buffer_transcription:
            combined_buffer += sep + buffer_transcription
        buffer_diarization = combined_buffer
    
    return {
        "lines": lines,
        "buffer_transcription": buffer_transcription,
        "buffer_diarization": buffer_diarization,
        "remaining_time_transcription": remaining_time_transcription,
        "remaining_time_diarization": remaining_time_diarization
    }