import torchaudio
import whisper
import json
import os
from speechbrain.inference import VAD
from speechbrain.inference.speaker import EncoderClassifier
from utils.vad import extract_segments
from utils.cluster import cluster_and_visualize,plot_cluster
from utils.resample import ffmpeg_resample_inplace
import torch
import numpy as np

from sklearn.cluster import AgglomerativeClustering

# 1. load audio
wav_path = "/opt/data/majikui/audios/bigtest.wav"
basename = os.path.splitext(os.path.basename(wav_path))[0]
ffmpeg_resample_inplace(wav_path,16000)
signal, fs = torchaudio.load(wav_path)

# if multi-channel, take the first channel
if signal.shape[0] > 1:
    signal = signal[0:1, :]

# 2. VAD Detection
vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmp_vad")
boundaries = extract_segments(wav_path, threshold=0.5, frame_shift=0.01)

# 3. speaker embedding (ECAPA-TDNN)
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
MIN_DURATION = 1.0

speaker_embeddings = []
valid_segments = []

for seg in boundaries:
    start, end = seg[0], seg[1]
    if end - start < MIN_DURATION:
        continue
    segment = signal[:, int(start * fs):int(end * fs)]
    embedding = classifier.encode_batch(segment).squeeze(0)
    speaker_embeddings.append(embedding)
    valid_segments.append((start, end))
embeddings_np = np.vstack([emb.cpu().numpy() for emb in speaker_embeddings])
# 4. clustering

labels = cluster_and_visualize(embeddings_np=embeddings_np)
plot_cluster(embeddings_np,labels,basename)

# 5. initialize Whisper
asr_model = whisper.load_model("large")

# 6. transcribe each segment in memory
results = []

for i, ((start, end), label) in enumerate(zip(valid_segments, labels)):
    segment_audio = signal[:, int(start * fs):int(end * fs)].squeeze(0).numpy()
    segment_audio = segment_audio.astype(np.float32)
    if fs != 16000:
        segment_audio = torchaudio.functional.resample(
            torch.from_numpy(segment_audio),orig_freq=fs, new_freq=16000
            ).numpy()
    segment_tensor = torch.from_numpy(segment_audio)
    result = asr_model.transcribe(segment_tensor, language="zh")
    text = result["text"].strip()
    
    results.append({
        "start": round(start, 2),
        "end": round(end, 2),
        "speaker": f"Speaker_{label}",
        "text": text
    })

# 7. output structured text
for r in results:
    print(f"[{r['start']:.2f}s - {r['end']:.2f}s] {r['speaker']}: {r['text']}")
savedir = f"transcribe_results/{basename}_results.json"
with open(savedir,"w",encoding="utf-8") as f:
    json.dump(results,f,ensure_ascii=False,indent=4)

