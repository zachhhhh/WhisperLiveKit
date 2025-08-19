import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

try:
    from nemo.collections.asr.models import SortformerEncLabelModel
except ImportError:
    raise SystemExit("""Please use `pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"` to use the Sortformer diarization""")
    

diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2")
diar_model.eval()

if torch.cuda.is_available():
    diar_model.to(torch.device("cuda"))
    
# Set the streaming parameters corresponding to 1.04s latency setup. This will affect the streaming feat loader.
# diar_model.sortformer_modules.chunk_len = 6
# diar_model.sortformer_modules.spkcache_len = 188
# diar_model.sortformer_modules.chunk_right_context = 7
# diar_model.sortformer_modules.fifo_len = 188
# diar_model.sortformer_modules.spkcache_update_period = 144
# diar_model.sortformer_modules.log = False


# here we change the settings for our goal: speed!
# we want batches of around 1 second. one frame is 0.08s, so 1s is 12.5 frames. we take 12.
diar_model.sortformer_modules.chunk_len = 12

# for more speed, we reduce the 'right context'. it's like looking less into the future.
diar_model.sortformer_modules.chunk_right_context = 1

# we keep the rest same for now
diar_model.sortformer_modules.spkcache_len = 188
diar_model.sortformer_modules.fifo_len = 188
diar_model.sortformer_modules.spkcache_update_period = 144
diar_model.sortformer_modules.log = False
diar_model.sortformer_modules._check_streaming_parameters()

batch_size = 1
processed_signal_offset = torch.zeros((batch_size,), dtype=torch.long, device=diar_model.device)

def prepare_audio_signal(signal):
    audio_signal = torch.tensor(signal).unsqueeze(0).to(diar_model.device)
    audio_signal_length = torch.tensor([audio_signal.shape[1]]).to(diar_model.device)
    processed_signal, processed_signal_length = diar_model.preprocessor(input_signal=audio_signal, length=audio_signal_length)
    return processed_signal, processed_signal_length

def create_streaming_loader(processed_signal, processed_signal_length):
    streaming_loader = diar_model.sortformer_modules.streaming_feat_loader(
        feat_seq=processed_signal,
        feat_seq_length=processed_signal_length,
        feat_seq_offset=processed_signal_offset,
    )
    return streaming_loader


def process_diarization(streaming_loader):
    
    streaming_state = diar_model.sortformer_modules.init_streaming_state(
        batch_size = batch_size,
        async_streaming = True,
        device = diar_model.device
)
    total_preds = torch.zeros((batch_size, 0, diar_model.sortformer_modules.n_spk), device=diar_model.device)

    
    chunk_duration_seconds = diar_model.sortformer_modules.chunk_len * diar_model.sortformer_modules.subsampling_factor * diar_model.preprocessor._cfg.window_stride
    print(f"Chunk duration: {chunk_duration_seconds} seconds")

    l_speakers = [
        {'start_time': 0,
        'end_time': 0,
        'speaker': 0
        }
    ]
    len_prediction = None
    for i, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in streaming_loader:
        with torch.inference_mode():
                streaming_state, total_preds = diar_model.forward_streaming_step(
                    processed_signal=chunk_feat_seq_t,
                    processed_signal_length=feat_lengths,
                    streaming_state=streaming_state,
                    total_preds=total_preds,
                    left_offset=left_offset,
                    right_offset=right_offset,
                )
                preds_np = total_preds[0].cpu().numpy()
                active_speakers = np.argmax(preds_np, axis=1)
                if len_prediction is None:
                    len_prediction = len(active_speakers) # we want to get the len of 1 prediction
                frame_duration = chunk_duration_seconds / len_prediction
                active_speakers = active_speakers[-len_prediction:]

                for idx, spk in enumerate(active_speakers):
                    if spk != l_speakers[-1]['speaker']:
                        l_speakers.append(
                            {'start_time': i * chunk_duration_seconds + idx * frame_duration,
                            'end_time': i * chunk_duration_seconds + (idx + 1) * frame_duration,
                            'speaker': spk
                        })                    
                    else:
                        l_speakers[-1]['end_time'] = i * chunk_duration_seconds + (idx + 1) * frame_duration
                    
                print(l_speakers)

if __name__ == '__main__':
    import librosa
    an4_audio = 'new_audio_test.mp3'
    signal, sr = librosa.load(an4_audio,sr=16000) 


    processed_signal, processed_signal_length = prepare_audio_signal(signal)
    streaming_loader = create_streaming_loader(processed_signal, processed_signal_length)
    process_diarization(streaming_loader)