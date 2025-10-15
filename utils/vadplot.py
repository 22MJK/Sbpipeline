import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from speechbrain.pretrained import VAD
from resample import ffmpeg_resample_inplace

def plot_vad(vad, wav_path, frame_shift=0.01, basename=None):
    """
    """
    os.makedirs("pics/vad", exist_ok=True)
    if basename is None:
        basename = os.path.splitext(os.path.basename(wav_path))[0]
    savedir = f"/data/majikui/Sbpipeline/pics/vad/{basename}_vad_only.png"

    # 读取音频
    signal, fs = librosa.load(wav_path, sr=None)
    duration = len(signal) / fs
    times = np.linspace(0, duration, num=len(signal))

    # 获取 VAD 概率
    probs = vad.get_speech_prob_file(wav_path).squeeze()
    print(probs)
    probs_np = probs.detach().cpu().float().numpy()
    vad_times = np.arange(len(probs_np)) * frame_shift

    # 绘图
    plt.figure(figsize=(12, 4))

    # 波形
    plt.plot(times, signal, alpha=0.8, label="Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # VAD 概率
    plt.plot(vad_times, probs, color="red", label="VAD Probability")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude / Probability")
    plt.title("Waveform and VAD Probability")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(savedir, dpi=100)
    plt.close()
    print(f"Saved VAD plot to {savedir}")


if __name__ == "__main__":
    # parameters
    wav_path = "/tmpdata01/majikui/audios/bigtest.wav"
    ffmpeg_resample_inplace(wav_path,16000)

    vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmp_vad")
    plot_vad(vad,wav_path,frame_shift=0.01,basename="bigtest")