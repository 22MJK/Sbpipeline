import torchaudio
from speechbrain.inference import VAD

# parameters
wav_path = "/opt/data/majikui/audios/smalltest.wav"
threshold = 0.5
min_duration = 0.3
max_duration = 10.0
frame_shift = 0.01  # 10ms

# load model
vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmp_vad")

# directly call get_speech_prob_file to get probability (input is file path)
probs = vad.get_speech_prob_file(wav_path)  

# custom function, generate speech segments from probability
def extract_segments(probs, threshold, frame_shift, min_duration=0.3, max_duration=10.0):
    probs = probs.squeeze()  # (T,)
    speech = probs > threshold
    segments = []
    start = None

    for i, is_speech in enumerate(speech):
        if is_speech and start is None:
            start = i
        elif not is_speech and start is not None:
            end = i
            dur = (end - start) * frame_shift
            if dur >= min_duration:
                segments.append((start * frame_shift, end * frame_shift))
            start = None
    if start is not None:
        end = len(speech)
        dur = (end - start) * frame_shift
        if dur >= min_duration:
            segments.append((start * frame_shift, end * frame_shift))

    # merge close segments
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            last = merged[-1]
            if seg[0] - last[1] < 0.1:
                merged[-1] = (last[0], seg[1])
            else:
                merged.append(seg)

    # split long segments
    final_segments = []
    for seg in merged:
        s, e = seg
        if (e - s) > max_duration:
            n = int((e - s) / max_duration) + 1
            for i in range(n):
                new_s = s + i * max_duration
                new_e = min(new_s + max_duration, e)
                final_segments.append((new_s, new_e))
        else:
            final_segments.append(seg)

    return final_segments

if __name__ == "__main__":
    
    segments = extract_segments(probs, threshold, frame_shift, min_duration, max_duration)

    print("Detected speech segments:")
    for start, end in segments:
        print(f"Start: {start:.2f}s, End: {end:.2f}s")
