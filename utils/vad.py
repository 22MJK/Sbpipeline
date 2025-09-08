import torchaudio
import torch
import librosa
import matplotlib.pyplot as plt
from speechbrain.inference import VAD
from resample import ffmpeg_resample_inplace

def extract_segments(wav_path, threshold, frame_shift, min_duration=0.3, max_duration=10.0):
    vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmp_vad")
    probs = vad.get_speech_prob_file(wav_path)
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
def plot_segments(wav_path,segments,basename):
    savedir = f"pics/vad/{basename}_vad_result.png"
    signal, fs = librosa.load(wav_path,sr=None)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(signal,sr=fs)
    for i,(start,end) in enumerate(segments):
        plt.axvspan(start,end,color="red",alpha=0.3,label="Speech"if i == 0 else None)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("VAD Detected Speech Segments")
    plt.legend(loc="upper right")
    plt.subplots_adjust(bottom=0.15,top=0.9)
    plt.savefig(savedir,dpi=100)
    plt.close()
if __name__ == "__main__":
    # parameters
    wav_path = "/opt/data/majikui/audios/bigtest_vocals.wav"
    threshold = 0.5 #500ms
    min_duration = 0.3
    max_duration = 10.0
    frame_shift = 0.01  
    ffmpeg_resample_inplace(wav_path,16000)
    signal, fs = torchaudio.load(wav_path)
    segments = extract_segments(wav_path, threshold, frame_shift, min_duration, max_duration)
    plot_segments(wav_path,segments,"bigtest_vocals")
    print("Detected speech segments:")
    for start, end in segments:
        print(f"Start: {start:.2f}s, End: {end:.2f}s")