import os
import whisper
import torchaudio
import numpy as np
from speechbrain.inference import VAD
from asr_split import asr_extract_segments
from split import extract_segments

# parameters
wav_path = "/tmpdata01/majikui/audios/smalltest.wav"
basename = os.path.splitext(os.path.basename(wav_path))[0]
threshold = 0.5
frame_shift = 0.01  # 10ms
min_duration = 0.3
max_duration = 30.0  # seconds

# initialize VAD
vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmp_vad")


# extract speech segments
segments = asr_extract_segments(vad, wav_path, threshold, frame_shift, min_duration, max_duration)
# segments = extract_segments(vad, wav_path, threshold, frame_shift, min_duration, max_duration)
signals,fs = torchaudio.load(wav_path)

# initialize Whisper
model = whisper.load_model("large")
for i, (start, end) in enumerate(segments):
    start_sample = int(start * fs)
    end_sample = int(end * fs)
    seg_signal = signals[:, start_sample:end_sample]  # 
    seg_signal = seg_signal.mean(dim=0)  # 
    seg_signal = seg_signal.numpy().astype(np.float32)
    result = model.transcribe(seg_signal,language="zh")
    print(f"[Segment {i}] {start:.2f}-{end:.2f}: {result['text']}")
