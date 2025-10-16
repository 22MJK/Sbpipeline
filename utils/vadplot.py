import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from speechbrain.pretrained import VAD
from resample import ffmpeg_resample_inplace

def plot_vad(vad, wav_path, frame_shift=0.01, threshold=0.5):
    """
    just plot vad probabilities and amplitude
    Args:
        vad: VAD object from speechbrain
        wav_path: path to the audio file
        frame_shift: frame shift in seconds (default: 0.01s)
        basename: base name for saving the plot (if None, use wav file name)
    Returns:
        None (saves the plot to pics/vad directory)
    """
    os.makedirs("/data/majikui/Sbpipeline/pics/vad", exist_ok=True)
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    savedir = f"/data/majikui/Sbpipeline/pics/vad/{basename}_vadplot.png"

    # load audio
    signal, fs = librosa.load(wav_path, sr=None)
    duration = len(signal) / fs
    times = np.linspace(0, duration, num=len(signal))

    # get VAD probabilities
    probs = vad.get_speech_prob_file(wav_path).squeeze()
    # print(probs)
    probs_np = probs.detach().cpu().float().numpy()
    vad_times = np.arange(len(probs_np)) * frame_shift

    # plot
    plt.figure(figsize=(12, 4))

    # waveform
    plt.plot(times, signal, alpha=0.8, label="Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # VAD probabilities
    plt.plot(vad_times, probs, color="red", label="VAD Probability")
    plt.axhline(y=threshold, color="green", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}")
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

    vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="/data/majikui/Sbpipeline/tmp_vad")
    plot_vad(vad,wav_path,frame_shift=0.01)