import torchaudio
import torch
import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
from speechbrain.inference import VAD
from resample import ffmpeg_resample_inplace

def asr_extract_segments(vad,wav_path, threshold, frame_shift, min_duration=0.3, max_duration=10.0):
    
    probs = vad.get_speech_prob_file(wav_path)
    if isinstance(probs, torch.Tensor):    
        probs = probs.cpu().numpy()
    probs = probs.squeeze()
    
    speech = probs > threshold
    changes = np.diff(speech.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if speech[0]:
        starts = np.r_(0, starts)
    if speech[-1]:
        ends = np.r_[ends, len(speech)]
    
    segments = [(s*frame_shift, e*frame_shift) 
                for s, e in zip(starts, ends)
                if (e - s)*frame_shift >= min_duration]

    merged =   []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            last = merged[-1]
            min_gap =1.0
            if seg[0] - last[1] <= min_gap:
                merged[-1] = (last[0], seg[1])
            else:
                merged.append(seg)
    final_segments = []
    if merged:
        temp_start, temp_end = merged[0]
        for seg in merged[1:]:
            start, end = seg
            if end - temp_start <= max_duration:
                temp_end = end
            else:
                final_segments.append((temp_start, temp_end))
                temp_start, temp_end = start, end
        final_segments.append((temp_start, temp_end))
    

    return final_segments
def plot_segments(wav_path,segments):
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    os.makedirs("/data/majikui/Sbpipeline/pics/split",exist_ok=True)
    savedir = f"/data/majikui/Sbpipeline/pics/split/{basename}_asr_segments.png"

    signal, fs = librosa.load(wav_path,sr=None)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(signal,sr=fs,alpha=0.8)

    for i,(start,end) in enumerate(segments):
        plt.axvspan(start,end,color="red",alpha=0.3,label="Speech"if i == 0 else None)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Detected Speech Segments for ASR")
    if segments:
        plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(savedir,dpi=100)
    plt.close()
if __name__ == "__main__":
    # parameters
    wav_path = "/tmpdata01/majikui/audios/bigtest.wav"
    threshold = 0.5 #probability threshold
    min_duration = 0.3
    max_duration = 30.0
    frame_shift = 0.01  
    ffmpeg_resample_inplace(wav_path,16000)
    vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="/data/majikui/Sbpipeline/tmp_vad")
    segments = asr_extract_segments(vad,wav_path, threshold, frame_shift, min_duration, max_duration)
    plot_segments(wav_path,segments)
    print("Detected speech segments:")
    for start, end in segments:
        print(f"Start: {start:.2f}s, End: {end:.2f}s")