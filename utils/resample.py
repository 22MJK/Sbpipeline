import os
import subprocess
import torchaudio

def ffmpeg_resample_inplace(filepath, target_sr=16000):

    info = torchaudio.info(filepath)
    original_sr = info.sample_rate

    if original_sr == target_sr:
        print(f"Sampling rate is already {target_sr}Hz")
        return


    dir_path = os.path.dirname(filepath)
    base_name = os.path.basename(filepath)
    temp_path = os.path.join(dir_path, "tmp_" + base_name)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", filepath,
        "-ar", str(target_sr),
        "-ac", "1",
        temp_path
    ]
    subprocess.run(cmd, check=True)

    os.replace(temp_path, filepath)
    print(f"Resample to {target_sr}Hz")

if __name__ == "__main__":
    wav_path = "/opt/data/majikui/audios/smalltest.wav"
    ffmpeg_resample_inplace(wav_path, 16000)
    info = torchaudio.info(wav_path)
    print(info)
